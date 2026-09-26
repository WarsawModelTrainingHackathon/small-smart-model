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
#   "tantivy==0.26.2",
# ]
# ///
"""Evaluate a (Gemma 4) model on the matura z historii benchmark.

Examples (see docs/eval.md):
  uv run scripts/eval_matura.py --model google/gemma-4-12B-it --load-4bit --split test \
      --data /tmp/mh/data/matura-historia/data.json --label base-4bit --judge openai
  uv run scripts/eval_matura.py --output runs/eval/base-4bit --regrade --judge openai
  # optional Wikipedia RAG (local index built by scripts/wiki_rag.py; label gets a -rag5 suffix):
  uv run scripts/eval_matura.py --model google/gemma-4-12B-it --load-4bit --split test \
      --data /tmp/mh/data/matura-historia/data.json --rag-index /team/wiki/index --rag-k 5 --rag-max-chars 3000

Outputs in --output: run.json (config), generations.jsonl (resumable), graded.jsonl,
report.json, report.md; with --rag-index also retrieval.jsonl (query + passages per item).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matura_format as mf  # noqa: E402
import matura_grading as mg  # noqa: E402

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


def encode(processor, items, text_only=False, thinking=False, image_max_soft_tokens=560, device='cpu',
           passages=None, rag_max_chars=mf.DEFAULT_RAG_MAX_CHARS):
    """Batch-encode prompts (left padding). All items in a batch must agree on having images.

    `passages`: optional {item id: retrieval hits} -> Wikipedia block in the prompt (see mf.user_text).
    """
    text_only = text_only or not supports_images(processor)
    passages = passages or {}
    prompts = [mf.render_prompt(processor, it, text_only=text_only, thinking=thinking,
                                passages=passages.get(it['id']), rag_max_chars=rag_max_chars) for it in items]
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
             image_max_soft_tokens=560, on_batch=None, passages=None, rag_max_chars=mf.DEFAULT_RAG_MAX_CHARS):
    """Greedy generation for items -> [{id, output, new_tokens, seconds}] (also used by training dev eval).

    `passages`: optional {item id: retrieval hits} prepended to the prompts (RAG); None = plain prompts.
    """
    import torch
    device = next(p.device for p in model.parameters() if p.device.type != 'meta')
    budget_of = max_new_tokens if callable(max_new_tokens) else (lambda it: max_new_tokens)
    results = []
    tok = tokenizer_of(processor)
    eos = [tok.convert_tokens_to_ids(t) for t in ('<turn|>', '<end_of_turn>') if t in tok.get_vocab()]
    eos += [tok.eos_token_id] if tok.eos_token_id is not None else []
    for batch in batches(items, batch_size, text_only, key=budget_of):
        t0 = time.time()
        enc, _ = encode(processor, batch, text_only, thinking, image_max_soft_tokens, device, passages, rag_max_chars)
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


# --------------------------------------------------------------------------- retrieval (optional RAG)

def fallback_query(item, max_chars=1000):
    """Query used when scripts/wiki_rag.py has no build_query: the question (names the topic), then the sources."""
    text = ' '.join(x for x in (item.get('question'), item.get('context')) if x)
    return re.sub(r'\s+', ' ', text).strip()[:max_chars]


def open_rag_index(index_dir):
    """(index, build_query, query_source). scripts/wiki_rag.py is imported only here, i.e. only with --rag-index."""
    try:
        import wiki_rag
    except ImportError as e:
        raise SystemExit(f'--rag-index needs scripts/wiki_rag.py and its dependencies: {e}') from e
    index = wiki_rag.WikiIndex(str(index_dir))
    for owner, name in ((index, 'WikiIndex.build_query'), (wiki_rag, 'wiki_rag.build_query')):
        fn = getattr(owner, 'build_query', None)
        if callable(fn):
            return index, fn, name
    return index, fallback_query, 'fallback'


def _hit(h):
    get = h.get if isinstance(h, dict) else (lambda key, default=None: getattr(h, key, default))
    score = get('score')
    return dict(title=str(get('title') or ''), text=str(get('text') or ''),
                score=None if score is None else round(float(score), 4))


def retrieve(items, index_dir, k, out):
    """Retrieve once per item, before any generation -> {id: [{title, text, score}, ...]} (rank order).

    Records (query + full passages) are appended to out/retrieval.jsonl and reused on resume, so an item is
    never retrieved twice. The index is released before the LLM is loaded.
    """
    path = out / 'retrieval.jsonl'
    cache = {r['id']: r for r in read_jsonl(path)}
    need = [it for it in items if it['id'] not in cache]
    if need:
        t0 = time.time()
        index, build_query, source = open_rag_index(index_dir)
        print(f'rag: index {index_dir} opened in {time.time() - t0:.0f}s (query: {source}, k={k})', flush=True)
        with open(path, 'a', encoding='utf-8') as f:
            for it in need:
                t1 = time.time()
                query = build_query(it)
                hits = [_hit(h) for h in (index.search(query, k=k) or [])][:k]
                rec = dict(id=it['id'], query=query, query_source=source, k=k, seconds=round(time.time() - t1, 3),
                           passages=hits)
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
                cache[it['id']] = rec
        del index, build_query
        gc.collect()
        print(f'rag: retrieved passages for {len(need)} items in {time.time() - t0:.0f}s -> {path}', flush=True)
    return {it['id']: cache[it['id']]['passages'] for it in items}


def rag_row(passages, max_chars):
    """What generations.jsonl records per item: retrieved titles + scores, how many passages / chars made it in."""
    return dict(hits=[dict(title=p['title'], score=p['score']) for p in passages],
                used=len(mf.fit_passages(passages, max_chars)), chars=len(mf.passages_block(passages, max_chars)))


def rag_stats(out, item_ids, max_chars):
    keep = set(item_ids)
    recs = [r for r in read_jsonl(out / 'retrieval.jsonl') if r['id'] in keep]
    if not recs:
        return None
    rows = [rag_row(r['passages'], max_chars) for r in recs]
    top = [r['passages'][0]['score'] for r in recs if r['passages'] and r['passages'][0].get('score') is not None]
    return dict(n_items=len(recs), query_source=','.join(sorted({str(r.get('query_source')) for r in recs})),
                mean_hits=round(sum(len(r['passages']) for r in recs) / len(recs), 2),
                mean_used=round(sum(r['used'] for r in rows) / len(rows), 2),
                mean_block_chars=round(sum(r['chars'] for r in rows) / len(rows), 1),
                mean_top_score=round(sum(top) / len(top), 4) if top else None,
                retrieval_seconds=round(sum(r.get('seconds', 0) for r in recs), 1))


def rag_config_key(rag):
    """What must match to resume a run dir: RAG on/off, k, char budget, index (by folder name)."""
    return None if not rag else (Path(rag['index']).name, rag['k'], rag['max_chars'])


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
    p.add_argument('--split', default='dev', choices=['dev', 'test', 'train'])
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
    p.add_argument('--rag-index', help='local Wikipedia index dir (scripts/wiki_rag.py WikiIndex): retrieve passages once '
                                       'per item and prepend them to the prompt; the label gets a -rag<k> suffix')
    p.add_argument('--rag-k', type=int, default=5, help='passages retrieved per item')
    p.add_argument('--rag-max-chars', type=int, default=mf.DEFAULT_RAG_MAX_CHARS,
                   help='max characters of the Wikipedia block in the prompt (header included)')
    a = p.parse_args(argv)
    if a.rag_index:
        if a.rag_k < 1:
            p.error('--rag-k must be >= 1')
        if a.rag_max_chars < len(mf.RAG_HEADER) + mf.MIN_RAG_TRUNCATED_CHARS:
            p.error(f'--rag-max-chars must be >= {len(mf.RAG_HEADER) + mf.MIN_RAG_TRUNCATED_CHARS}')
        if not a.regrade and not Path(a.rag_index).exists():
            raise SystemExit(f'--rag-index {a.rag_index} does not exist')

    rag_tag = None
    if a.rag_index:
        rag_tag = f'rag{a.rag_k}' + ('' if a.rag_max_chars == mf.DEFAULT_RAG_MAX_CHARS else f'c{a.rag_max_chars}')
    label = a.label or '-'.join(x for x in [Path(a.model).name, Path(a.adapter).name if a.adapter else None,
                                             '4bit' if a.load_4bit else None, a.split,
                                             'text' if a.text_only else None, 'think' if a.thinking == 'on' else None,
                                             rag_tag] if x)
    if rag_tag and a.label and 'rag' not in a.label.lower():
        label = f'{a.label}-{rag_tag}'  # RAG runs are always distinguishable by label
    out = Path(a.output or f'runs/eval/{label}')
    out.mkdir(parents=True, exist_ok=True)
    items = select_items(a)
    rag = dict(index=str(Path(a.rag_index).resolve()), k=a.rag_k, max_chars=a.rag_max_chars) if a.rag_index else None
    run = dict(label=label, model=a.model, adapter=a.adapter, load_4bit=a.load_4bit, split=a.split,
               text_only=a.text_only, thinking=a.thinking, image_max_soft_tokens=a.image_max_soft_tokens,
               max_new_tokens=a.max_new_tokens, essay_max_new_tokens=a.essay_max_new_tokens, essays=a.essays,
               data=str(a.data), data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
               item_ids=[it['id'] for it in items], rag=rag)
    if a.regrade:
        if not (out / 'generations.jsonl').exists():
            raise SystemExit(f'--regrade: no {out}/generations.jsonl')
        if a.rag_index:
            print('--regrade: grading saved generations; no retrieval (RAG config is read from run.json)', flush=True)
        return grade_and_report(a, items, out)

    old = json.loads((out / 'run.json').read_text(encoding='utf-8')) if (out / 'run.json').exists() else None
    if old:
        for k in ('model', 'adapter', 'load_4bit', 'text_only', 'thinking', 'split'):
            if old.get(k) != run[k]:
                raise SystemExit(f'{out} holds a run with {k}={old.get(k)!r}, not {run[k]!r}; use another --output')
        if rag_config_key(old.get('rag')) != rag_config_key(rag):
            raise SystemExit(f'{out} holds a run with rag={old.get("rag")!r}, not {rag!r}; use another --output')
    write_json(out / 'run.json', run)
    done = {g['id'] for g in read_jsonl(out / 'generations.jsonl')}
    todo = [it for it in items if it['id'] not in done]
    print(f'{len(items)} items, {len(done & set(run["item_ids"]))} already generated, {len(todo)} to go -> {out}', flush=True)
    passages = None
    if rag:
        if todo:
            passages = retrieve(todo, a.rag_index, a.rag_k, out)  # all retrieval happens before the LLM is loaded
        run['rag_stats'] = rag_stats(out, run['item_ids'], a.rag_max_chars)
        write_json(out / 'run.json', run)
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
                if passages is not None:
                    r['rag'] = rag_row(passages[r['id']], a.rag_max_chars)
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            n[0] += len(rows)
            print(f'  {n[0]}/{len(items)} generated ({time.time() - t0:.0f}s)', flush=True)

        generate(model, processor, todo, lambda it: budget(it, a), a.batch_size, a.text_only, a.thinking == 'on',
                 a.image_max_soft_tokens, on_batch=save, passages=passages, rag_max_chars=a.rag_max_chars)
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
