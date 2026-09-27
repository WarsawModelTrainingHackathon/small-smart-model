# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.0",
#   "torchvision==0.29.0",
#   "transformers==5.17.0",
#   "peft==0.21.0",
#   "accelerate==1.15.0",
#   "bitsandbytes==0.50.2; sys_platform == 'linux'",
#   "pillow==12.3.0",
# ]
# ///
"""QLoRA fine-tuning of Gemma 4 (default google/gemma-4-12B-it) for the matura z historii.

* Base in 4-bit bitsandbytes nf4 (bf16 compute) by default; --no-4bit for bf16 LoRA.
* PEFT LoRA on the language-model linear layers only (vision/audio embedders untouched).
* Assistant-only loss: prompt tokens are masked; only the target tokens (+ end of turn) are learned.
* Prompts come from scripts/matura_format.py, exactly as in scripts/eval_matura.py.
  Targets: closed items -> short reasoning + final 'Odpowiedź:' line; open items -> CKE example
  answer; essays skipped by default (no example answers).
* After every epoch: dev eval on the auto-gradable items with the SAME generation + parser as the
  benchmark, plus teacher-forced dev loss on all dev targets; the best adapter is kept.
* Only the train and dev splits are ever loaded (never test).

  uv run scripts/train_lora.py --data /tmp/mh/data/matura-historia/data.json --output runs/train/qlora-r16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matura_format as mf  # noqa: E402
import eval_matura as em  # noqa: E402

DEFAULT_TARGETS = 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'
NON_LM = ('vision', 'audio', 'embed_vision', 'embed_audio', 'multi_modal', 'lm_head')


def lora_target_names(model, leaf_names, scope='language'):
    """Full module names of the Linear / Linear4bit layers LoRA should wrap."""
    import torch
    leafs = set(leaf_names)
    out = []
    for name, module in model.named_modules():
        if name.rsplit('.', 1)[-1] not in leafs:
            continue
        if not (isinstance(module, torch.nn.Linear) or type(module).__name__ in ('Linear4bit', 'Linear8bitLt')):
            continue
        if scope == 'language' and any(x in name for x in NON_LM):
            continue
        out.append(name)
    if not out:
        raise ValueError(f'no LoRA targets found for {sorted(leafs)}')
    return out


def end_of_turn(tok):
    for t in ('<turn|>', '<end_of_turn>'):
        if t in tok.get_vocab():
            return t
    return tok.eos_token


def select_examples(items, include_open=True, include_essays=False):
    rows = []
    for it in items:
        if it['type'] == mf.ESSAY and not include_essays:
            continue
        if it['type'] in mf.OPEN_TYPES and not include_open:
            continue
        target = mf.build_target(it)
        if target:
            rows.append((it, target))
    return rows


def encode_example(processor, item, target, a, device):
    """input_ids = prompt (+ images) + target + end-of-turn; returns (enc, n_target) or None if too long."""
    import torch
    tok = em.tokenizer_of(processor)
    enc, _ = em.encode(processor, [item], text_only=a.text_only, thinking=False,
                       image_max_soft_tokens=a.image_max_soft_tokens, device='cpu')
    target_ids = tok(target, add_special_tokens=False, return_tensors='pt')['input_ids']
    end_ids = tok(end_of_turn(tok) + '\n', add_special_tokens=False, return_tensors='pt')['input_ids']
    n_prompt = enc['input_ids'].shape[1]
    if n_prompt + 8 > a.max_len:
        return None
    # Keep the end-of-turn token even when a long self-distilled answer must be clipped.
    # Without it, the model is trained on an unfinished continuation and may not learn to stop.
    target_budget = a.max_len - n_prompt
    target_ids = target_ids[:, :max(0, target_budget - end_ids.shape[1])]
    tgt = torch.cat([target_ids, end_ids], dim=1)
    enc['input_ids'] = torch.cat([enc['input_ids'], tgt], 1)
    enc['attention_mask'] = torch.cat([enc['attention_mask'], torch.ones_like(tgt)], 1)
    if 'mm_token_type_ids' in enc:
        enc['mm_token_type_ids'] = torch.cat([enc['mm_token_type_ids'], torch.zeros_like(tgt)], 1)
    if 'token_type_ids' in enc:
        enc['token_type_ids'] = torch.cat([enc['token_type_ids'], torch.zeros_like(tgt)], 1)
    return enc.to(device), tgt.shape[1]


def target_loss(model, enc, n_target):
    """Mean CE over the target tokens only (logits only for the last n_target+1 positions)."""
    import torch
    out = model(**enc, use_cache=False, logits_to_keep=n_target + 1)
    logits = out.logits[:, :-1, :].float()
    labels = enc['input_ids'][:, -n_target:]
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def dev_loss(model, processor, examples, a, device, autocast):
    import torch
    model.eval()
    tot = n = 0
    with torch.no_grad():
        for it, target in examples:
            ex = encode_example(processor, it, target, a, device)
            if ex is None:
                continue
            with autocast():
                tot += target_loss(model, *ex).item()
            n += 1
    return round(tot / max(1, n), 5)


def run(a):
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import get_cosine_schedule_with_warmup

    started = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    data = mf.load_data(a.data, splits=['train', 'dev'])  # never the test split
    assert 'test' not in data
    if a.targets_jsonl:
        # self-distillation: train on our own graded answers instead of the CKE example answers
        tmap = {r['id']: r['target'] for r in map(json.loads, Path(a.targets_jsonl).read_text(encoding='utf-8').splitlines()) if r.get('target')}
        train = [(it, tmap[it['id']]) for it in data['train'] if it['id'] in tmap]
    else:
        train = select_examples(data['train'], not a.closed_only, a.include_essays)
    dev_items = data['dev']
    dev_closed = [it for it in dev_items if mf.auto_gradable(it)]
    dev_targets = select_examples(dev_items, not a.closed_only, False)
    if a.limit_train:
        train = train[:a.limit_train]
    if a.dev_limit:
        dev_closed, dev_targets = dev_closed[:a.dev_limit], dev_targets[:a.dev_limit]
    print(f'train examples: {len(train)} | dev closed: {len(dev_closed)} | dev loss targets: {len(dev_targets)}', flush=True)

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    use_4bit = not a.no_4bit
    model, processor = em.load_model(a.model, load_4bit=use_4bit, device=device, attn=a.attn,
                                     dtype=torch.bfloat16 if device == 'cuda' else torch.float32)
    tok = em.tokenizer_of(processor)
    for p in model.parameters():
        p.requires_grad_(False)
    targets = lora_target_names(model, a.target_modules.split(','), a.lora_scope)
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, target_modules=targets,
                     bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, cfg)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'LoRA on {len(targets)} modules, {n_train / 1e6:.2f}M trainable params', flush=True)

    autocast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if device == 'cuda' else \
        (lambda: torch.autocast('cpu', enabled=False))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(train) / a.grad_accum)
    total_steps = max(1, steps_per_epoch * a.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(a.warmup_ratio * total_steps), total_steps)

    def state():
        return {k: v.detach().to('cpu', copy=True) for k, v in get_peft_model_state_dict(model).items()}

    def dev_eval(tag):
        t0 = time.time()
        model.eval()
        gens = em.generate(model, processor, dev_closed, a.dev_max_new_tokens, a.dev_batch_size, a.text_only,
                           False, a.image_max_soft_tokens) if dev_closed else []
        closed = em.closed_dev_score(dev_closed, gens) if dev_closed else dict(points=0, max=0, pct=None, unparseable=0, n=0)
        loss = dev_loss(model, processor, dev_targets, a, device, autocast) if dev_targets else None
        rec = dict(tag=tag, dev_closed=closed, dev_loss=loss, seconds=round(time.time() - t0, 1),
                   samples=[dict(id=g['id'], output=g['output'][-300:]) for g in gens[:3]])
        print(f"[dev {tag}] closed {closed['points']}/{closed['max']} unparseable {closed['unparseable']} "
              f"| dev loss {loss} ({rec['seconds']}s)", flush=True)
        return rec

    def score(rec):
        if a.select_by == 'loss':
            return (-(rec['dev_loss'] if rec['dev_loss'] is not None else 1e9),)
        return (rec['dev_closed']['points'], -(rec['dev_loss'] if rec['dev_loss'] is not None else 1e9))

    report = dict(config=vars(a), model=a.model, quantization='nf4-4bit' if use_4bit else 'bf16', lora_targets=len(targets),
                  trainable_params=n_train, n_train=len(train), n_dev_closed=len(dev_closed), n_dev_targets=len(dev_targets),
                  split_policy='train only for training, dev for selection; test never loaded', history=[], dev=[])
    report['dev'].append(dev_eval('epoch0-base') if not a.skip_base_eval else dict(tag='epoch0-base', skipped=True))
    best, best_state, best_tag = None, None, None
    rng = random.Random(a.seed)
    step = 0
    skipped = 0
    t_train = time.time()
    stop = False
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        acc = acc_loss = 0
        for idx_pos, i in enumerate(order):
            it, target = train[i]
            ex = encode_example(processor, it, target, a, device)
            if ex is None:
                skipped += 1
                continue
            with autocast():
                loss = target_loss(model, *ex)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss on {it["id"]}')
            (loss / a.grad_accum).backward()
            acc += 1
            acc_loss += loss.item()
            last = idx_pos == len(order) - 1
            if acc == a.grad_accum or last:
                gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                rec = dict(step=step, epoch=epoch, loss=round(acc_loss / acc, 5), grad_norm=round(float(gn), 4),
                           lr=scheduler.get_last_lr()[0], elapsed=round(time.time() - t_train, 1))
                report['history'].append(rec)
                if step % a.log_every == 0 or last:
                    print(json.dumps(rec), flush=True)
                acc = acc_loss = 0
                if a.max_seconds and time.time() - t_train >= a.max_seconds:
                    print(f'--max-seconds {a.max_seconds} reached at step {step}', flush=True)
                    stop = True
                    break
        rec = dev_eval(f'epoch{epoch}')
        rec.update(epoch=epoch, step=step)
        report['dev'].append(rec)
        if a.save_every_epoch:
            model.save_pretrained(out / f'epoch{epoch}_adapter')
        if best is None or score(rec) > best:
            best, best_state, best_tag = score(rec), state(), rec['tag']
            model.save_pretrained(out / 'best_adapter')
            processor.save_pretrained(out / 'best_adapter')
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        if stop:
            break
    model.save_pretrained(out / 'final_adapter')
    base = report['dev'][0]
    report.update(best_epoch=best_tag, best_score=list(best) if best else None, steps=step, skipped_too_long=skipped,
                  train_seconds=round(time.time() - t_train, 1), total_seconds=round(time.time() - started, 1),
                  beats_base_on_dev_closed=(None if base.get('skipped') or best_state is None else
                                            next(r for r in report['dev'] if r['tag'] == best_tag)['dev_closed']['points']
                                            > base['dev_closed']['points']),
                  peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
                  best_adapter=str(out / 'best_adapter'), final_adapter=str(out / 'final_adapter'))
    if best_state is not None:
        set_peft_model_state_dict(model, best_state)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f"done: best {best_tag} {best} -> {out / 'best_adapter'} ({report['total_seconds']}s)", flush=True)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='google/gemma-4-12B-it')
    p.add_argument('--data', default=mf.DEFAULT_DATA)
    p.add_argument('--output', required=True)
    p.add_argument('--no-4bit', action='store_true', help='bf16 LoRA instead of QLoRA')
    p.add_argument('--lora-r', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    p.add_argument('--target-modules', default=DEFAULT_TARGETS, help='comma list of Linear leaf names')
    p.add_argument('--lora-scope', default='language', choices=['language', 'all'],
                   help='language = only language-model layers (default)')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--warmup-ratio', type=float, default=0.05)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--grad-accum', type=int, default=8, help='micro-batch is 1 example (images differ in size)')
    p.add_argument('--max-len', type=int, default=4096)
    p.add_argument('--max-seconds', type=int, default=0, help='stop training after this many seconds (0 = off)')
    p.add_argument('--text-only', action='store_true')
    p.add_argument('--image-max-soft-tokens', type=int, default=560, choices=em.IMAGE_SOFT_TOKENS)
    p.add_argument('--closed-only', action='store_true', help='train on closed items only')
    p.add_argument('--include-essays', action='store_true', help='(no example answers exist: currently a no-op)')
    p.add_argument('--targets-jsonl', help='JSONL {id, target}: train only on these train items with these targets (self-distillation)')
    p.add_argument('--no-gradient-checkpointing', dest='gradient_checkpointing', action='store_false')
    p.add_argument('--select-by', default='closed_then_loss', choices=['closed_then_loss', 'loss'])
    p.add_argument('--dev-max-new-tokens', type=int, default=512)
    p.add_argument('--dev-batch-size', type=int, default=4)
    p.add_argument('--dev-limit', type=int)
    p.add_argument('--limit-train', type=int, help='debug: first N training examples')
    p.add_argument('--skip-base-eval', action='store_true')
    p.add_argument('--save-every-epoch', action='store_true')
    p.add_argument('--log-every', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--attn', default='sdpa')
    p.add_argument('--device')
    return p.parse_args(argv)


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    run(parse_args())
