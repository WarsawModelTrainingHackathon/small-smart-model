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
"""Evaluate a (Gemma 4) model on the matura z historii benchmark.

Examples (see docs/eval.md):
  uv run scripts/eval_matura.py --model google/gemma-4-12B-it --load-4bit --split test \
      --data /tmp/mh/data/matura-historia/data.json --label base-4bit --judge openai
  uv run scripts/eval_matura.py --output runs/eval/base-4bit --regrade --judge openai

Outputs in --output: run.json (config), generations.jsonl (resumable), graded.jsonl,
report.json, report.md.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matura_format as mf  # noqa: E402
import matura_grading as mg  # noqa: E402
import wiki_bm25  # noqa: E402

IMAGE_SOFT_TOKENS = (70, 140, 280, 560, 1120)


# --------------------------------------------------------------------------- model

def bnb_config(torch):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              llm_int8_skip_modules=SKIP_QUANT)


# Keep the vision / audio embedders and the LM head in bf16 (tiny, and quantizing them hurts).
# (transformers matches these with re.match on the full module name, hence the '.*' prefixes.)
SKIP_QUANT = ['lm_head', r'.*embed_vision', r'.*embed_audio', r'.*vision_tower', r'.*audio_tower', r'.*multi_modal_projector']


def is_prequantized(model_id):
    """True if the checkpoint already carries a quantization_config (e.g. our 4-bit export)."""
    cfg = Path(model_id) / 'config.json'
    if cfg.exists():
        return 'quantization_config' in json.loads(cfg.read_text(encoding='utf-8'))
    try:
        from transformers import AutoConfig
        return getattr(AutoConfig.from_pretrained(model_id), 'quantization_config', None) is not None
    except Exception:  # noqa: BLE001
        return False


def load_model(model_id, adapter=None, load_4bit=False, device=None, dtype=None, attn='sdpa', trainable_adapter=False):
    """(model, processor). 4-bit = bitsandbytes nf4 exactly as the submitted checkpoint."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = dtype or (torch.bfloat16 if device == 'cuda' else torch.float32)
    kw = dict(dtype=dtype, attn_implementation=attn)
    prequant = is_prequantized(model_id)
    if load_4bit and not prequant:
        kw['quantization_config'] = bnb_config(torch)
    if load_4bit or prequant:
        kw['device_map'] = {'': 0} if device == 'cuda' else {'': 'cpu'}
    processor = AutoProcessor.from_pretrained(adapter if adapter and (Path(adapter) / 'processor_config.json').exists()
                                              else model_id)
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if 'device_map' not in kw:
        model.to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable_adapter)
    model.eval()
    tok = tokenizer_of(processor)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, processor


def tokenizer_of(processor):
    return getattr(processor, 'tokenizer', processor)


def supports_images(processor):
    return getattr(processor, 'image_processor', None) is not None and getattr(processor, 'image_token', None)


def load_images(item):
    from PIL import Image
    out = []
    for p in mf.image_paths(item):
        with Image.open(p) as im:
            out.append(im.convert('RGB'))
    return out


def encode(processor, items, text_only=False, thinking=False, image_max_soft_tokens=560, device='cpu'):
    """Batch-encode prompts (left padding). All items in a batch must agree on having images."""
    text_only = text_only or not supports_images(processor)
    prompts = [mf.render_prompt(processor, it, text_only=text_only, thinking=thinking) for it in items]
    with_images = not text_only and all(it.get('images') for it in items)
    if not text_only and any(it.get('images') for it in items) and not with_images:
        raise ValueError('mixed image / no-image batch; group items with batches()')
    if with_images:
        enc = processor(text=prompts, images=[load_images(it) for it in items], return_tensors='pt', padding=True,
                        add_special_tokens=False, images_kwargs={'max_soft_tokens': image_max_soft_tokens})
    else:
        enc = tokenizer_of(processor)(prompts, return_tensors='pt', padding=True, add_special_tokens=False)
    enc.pop('num_soft_tokens_per_image', None)
    return enc.to(device), prompts


def batches(items, batch_size, text_only, key=None):
    """Group items so a batch never mixes image/no-image prompts or generation budgets."""
    groups = {}
    for it in items:
        g = (bool(it.get('images')) and not text_only, key(it) if key else 0)
        groups.setdefault(g, []).append(it)
    for g in sorted(groups, key=str):
        rows = sorted(groups[g], key=lambda it: len(it.get('context') or '') + len(it.get('question') or ''))
        for i in range(0, len(rows), batch_size):
            yield rows[i:i + batch_size]


def decode(processor, seq):
    tok = tokenizer_of(processor)
    text = tok.decode(seq, skip_special_tokens=False)
    for stop in ('<turn|>', '<end_of_turn>', tok.eos_token or '<eos>'):
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    for t in (tok.pad_token, tok.bos_token):
        if t:
            text = text.replace(t, '')
    return text.strip()


def budget(item, a):
    base = a.essay_max_new_tokens if item['type'] == mf.ESSAY else a.max_new_tokens
    return base + (a.thinking_budget if a.thinking == 'on' else 0)


def generate(model, processor, items, max_new_tokens, batch_size=4, text_only=False, thinking=False,
             image_max_soft_tokens=560, on_batch=None):
    """Greedy generation for items -> [{id, output, new_tokens, seconds}] (also used by training dev eval)."""
    import torch
    device = next(p.device for p in model.parameters() if p.device.type != 'meta')
    budget_of = max_new_tokens if callable(max_new_tokens) else (lambda it: max_new_tokens)
    results = []
    tok = tokenizer_of(processor)
    eos = [tok.convert_tokens_to_ids(t) for t in ('<turn|>', '<end_of_turn>') if t in tok.get_vocab()]
    eos += [tok.eos_token_id] if tok.eos_token_id is not None else []
    for batch in batches(items, batch_size, text_only, key=budget_of):
        t0 = time.time()
        enc, _ = encode(processor, batch, text_only, thinking, image_max_soft_tokens, device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=budget_of(batch[0]), do_sample=False,
                                 eos_token_id=sorted(set(eos)) or None, pad_token_id=tok.pad_token_id)
        new = out[:, enc['input_ids'].shape[1]:]
        dt = time.time() - t0
        rows = []
        for it, seq in zip(batch, new):
            n = int((seq != tok.pad_token_id).sum())
            rows.append(dict(id=it['id'], output=decode(processor, seq), new_tokens=n,
                             prompt_tokens=int(enc['input_ids'].shape[1]), seconds=round(dt / len(batch), 2)))
        results += rows
        if on_batch:
            on_batch(rows)
    return results


def closed_dev_score(items, generations):
    """Points on auto-gradable items with the benchmark parser (training model selection)."""
    by_id = {it['id']: it for it in items}
    pts = mx = unparsed = 0
    for g in generations:
        r = mg.grade_closed(by_id[g['id']], mf.strip_thinking(g['output']))
        pts += r['points']
        mx += r['max_points']
        unparsed += not r['parse_ok']
    return dict(points=pts, max=mx, pct=round(100 * pts / mx, 2) if mx else None, unparseable=unparsed, n=len(generations))


# --------------------------------------------------------------------------- io

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def select_items(a):
    data = mf.load_data(a.data, splits=[a.split])
    if a.split not in data:
        raise SystemExit(f'split {a.split} not in {a.data}')
    items = data[a.split]
    if a.types:
        keep = set(a.types.split(','))
        items = [it for it in items if it['type'] in keep]
    if a.essays == 'none':
        items = [it for it in items if it['type'] != mf.ESSAY]
    elif a.essays == 'one':
        seen = set()
        items = [it for it in items if it['type'] != mf.ESSAY or not (mf.essay_group(it) in seen or seen.add(mf.essay_group(it)))]
    if a.ids:
        keep = set(a.ids.split(','))
        items = [it for it in items if it['id'] in keep]
    if a.limit:
        items = items[:a.limit]
    return items


def grade_and_report(a, items, out):
    gens = {g['id']: g for g in read_jsonl(out / 'generations.jsonl')}
    gens = [gens[it['id']] for it in items if it['id'] in gens]
    judge = mg.make_judge(a.judge, a.judge_model, a.judge_4bit)
    by_id = {it['id']: it for it in items}
    graded = mg.grade_records(by_id, gens, judge)
    with open(out / 'graded.jsonl', 'w', encoding='utf-8') as f:
        for g in graded:
            f.write(json.dumps(g, ensure_ascii=False, default=str) + '\n')
    run = json.loads((out / 'run.json').read_text(encoding='utf-8')) if (out / 'run.json').exists() else {}
    summary = mg.summarize(graded, by_id, essay_policy=a.essay_policy)
    meta = dict(run, judge=judge.name, n_items=len(items), n_generated=len(gens),
                mean_new_tokens=round(sum(g.get('new_tokens', 0) for g in gens) / max(1, len(gens)), 1),
                generation_seconds=round(sum(g.get('seconds', 0) for g in gens), 1))
    report = dict(meta=meta, summary=summary)
    write_json(out / 'report.json', report)
    (out / 'report.md').write_text(mg.report_markdown(summary, meta), encoding='utf-8')
    t = summary['total']
    print(f"[{meta.get('label')}] total {t['points']}/{t['max']} ({t['pct']}%), paper max {summary['exam_max']}, "
          f"ungraded {t['ungraded']}, closed {summary['closed_items']['pct']}%, "
          f"unparseable {summary['closed_unparseable']['n']}/{summary['closed_unparseable']['of']}", flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='google/gemma-4-12B-it', help='HF id or local dir (e.g. our 4-bit export)')
    p.add_argument('--adapter', help='PEFT LoRA adapter dir')
    p.add_argument('--load-4bit', action='store_true', help='bitsandbytes nf4 (evaluate like the submitted model)')
    p.add_argument('--data', default=mf.DEFAULT_DATA)
    p.add_argument('--split', default='dev', choices=['train', 'dev', 'test'],
                   help='train is for self-distill teacher runs only; Sunday exam uses test')
    p.add_argument('--limit', type=int)
    p.add_argument('--types', help='comma list of item types to keep')
    p.add_argument('--ids', help='comma list of item ids to keep')
    p.add_argument('--essays', default='all', choices=['all', 'one', 'none'],
                   help='all topics (score = policy over topics), only the first topic per session, or skip')
    p.add_argument('--essay-policy', default='mean', choices=['mean', 'best', 'first'])
    p.add_argument('--text-only', action='store_true', help='no images; use adapted_660_text where available')
    p.add_argument('--thinking', default='off', choices=['on', 'off'])
    p.add_argument('--image-max-soft-tokens', type=int, default=560, choices=IMAGE_SOFT_TOKENS)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=768)
    p.add_argument('--essay-max-new-tokens', type=int, default=3072)
    p.add_argument('--thinking-budget', type=int, default=4096, help='extra new tokens when --thinking on')
    p.add_argument('--label')
    p.add_argument('--output', help='run dir (default runs/eval/<label>)')
    p.add_argument('--regrade', action='store_true', help='skip generation; grade saved generations.jsonl')
    p.add_argument('--judge', default='none', choices=['none', 'hf', 'openai'])
    p.add_argument('--judge-model', help='hf: model id/dir; openai: model name (else $MATURA_JUDGE_MODEL)')
    p.add_argument('--judge-4bit', action='store_true')
    p.add_argument('--attn', default='sdpa')
    p.add_argument('--device')
    p.add_argument('--wiki', nargs='?', const=str(wiki_bm25.DEFAULT_CORPUS),
                   help='BM25 Polish-Wikipedia jsonl; flag alone uses harness/wiki/minicorpus.jsonl')
    p.add_argument('--wiki-k', type=int, default=2)
    p.add_argument('--wiki-essays', action='store_true', help='also retrieve for essay items (default: skip)')
    a = p.parse_args(argv)

    label = a.label or '-'.join(x for x in [Path(a.model).name, Path(a.adapter).name if a.adapter else None,
                                             '4bit' if a.load_4bit else None, a.split,
                                             'text' if a.text_only else None, 'think' if a.thinking == 'on' else None] if x)
    out = Path(a.output or f'runs/eval/{label}')
    out.mkdir(parents=True, exist_ok=True)
    items = select_items(a)
    wiki_hits = None
    if a.wiki:
        wiki_bm25.attach_rag(items, corpus=a.wiki, k=a.wiki_k, skip_essays=not a.wiki_essays)
        wiki_hits = {it['id']: it.get('rag_passage_ids', []) for it in items}
    run = dict(label=label, model=a.model, adapter=a.adapter, load_4bit=a.load_4bit, split=a.split,
               text_only=a.text_only, thinking=a.thinking, image_max_soft_tokens=a.image_max_soft_tokens,
               max_new_tokens=a.max_new_tokens, essay_max_new_tokens=a.essay_max_new_tokens, essays=a.essays,
               data=str(a.data), data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
               wiki=a.wiki, wiki_k=a.wiki_k if a.wiki else None,
               item_ids=[it['id'] for it in items], wiki_hits=wiki_hits)
    if a.regrade:
        if not (out / 'generations.jsonl').exists():
            raise SystemExit(f'--regrade: no {out}/generations.jsonl')
        return grade_and_report(a, items, out)

    old = json.loads((out / 'run.json').read_text(encoding='utf-8')) if (out / 'run.json').exists() else None
    if old:
        for k in ('model', 'adapter', 'load_4bit', 'text_only', 'thinking', 'split', 'wiki'):
            if old.get(k) != run[k]:
                raise SystemExit(f'{out} holds a run with {k}={old.get(k)!r}, not {run[k]!r}; use another --output')
    write_json(out / 'run.json', run)
    done = {g['id'] for g in read_jsonl(out / 'generations.jsonl')}
    todo = [it for it in items if it['id'] not in done]
    print(f'{len(items)} items, {len(done & set(run["item_ids"]))} already generated, {len(todo)} to go -> {out}', flush=True)
    if todo:
        import torch
        t0 = time.time()
        model, processor = load_model(a.model, a.adapter, a.load_4bit, device=a.device, attn=a.attn)
        print(f'model loaded in {time.time() - t0:.0f}s', flush=True)
        if a.text_only is False and not supports_images(processor):
            print('processor has no image support -> text-only prompts', flush=True)
        f = open(out / 'generations.jsonl', 'a', encoding='utf-8')
        n = [len(done)]

        def save(rows):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            n[0] += len(rows)
            print(f'  {n[0]}/{len(items)} generated ({time.time() - t0:.0f}s)', flush=True)

        generate(model, processor, todo, lambda it: budget(it, a), a.batch_size, a.text_only, a.thinking == 'on',
                 a.image_max_soft_tokens, on_batch=save)
        f.close()
        run['peak_vram_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
        write_json(out / 'run.json', run)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return grade_and_report(a, items, out)


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
