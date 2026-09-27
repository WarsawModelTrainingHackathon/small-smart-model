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
"""Turn an organizer exam pack (exam.json + images/) into answers.json for submissions.html.

Does not upload. Does not put TEAM_KEY in the file. Exam packs stay off git.

  bash scripts/run_exam.sh /path/to/unpacked-exam answers.json
  # 4-bit google/gemma-4-12B-it + plwiki RAG, essays without wiki. No LoRA.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_matura as em  # noqa: E402
import matura_format as mf  # noqa: E402

_ALLOWED_TOP = {'exam_id', 'answers'}
_ALLOWED_ANS = {'id', 'answer'}
MAX_CHARS = 100_000
MAX_BYTES = 1_000_000
DEFAULT_RAG_SCRIPTS = '/team/rag-wt/scripts'


def attach_wiki_rag(items, index_dir, k=5, max_chars=3000, skip_types=('essay',), scripts=DEFAULT_RAG_SCRIPTS):
    """Offline plwiki BM25 (tantivy). Same index as the 98.33 base-test-rag run. Skips essays by default."""
    scripts = str(scripts)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import wiki_rag  # noqa: WPS433
    index = wiki_rag.WikiIndex(str(index_dir))
    build_query = getattr(wiki_rag, 'build_query', None)
    skip = set(skip_types)
    n = 0
    for it in items:
        if it.get('type') in skip:
            continue
        query = build_query(it) if callable(build_query) else f"{it.get('question') or ''} {it.get('context') or ''}"
        hits = list(index.search(query, k=k) or [])[:k]
        lines = []
        for i, h in enumerate(hits, 1):
            get = h.get if isinstance(h, dict) else lambda key, default='': getattr(h, key, default)
            title, text = get('title') or '', get('text') or ''
            lines.append(f'[{i}] {title}: {text}' if title else f'[{i}] {text}')
        if not lines:
            continue
        header = (
            'Fragmenty z Wikipedii (pomocnicze, CC BY-SA; to NIE jest źródło z arkusza — '
            'jeśli hałasują, ignoruj je i trzymaj się materiałów CKE):'
        )
        block = header + '\n' + '\n'.join(lines)
        it['rag_context'] = block[:max_chars]
        n += 1
    print(f'rag: attached to {n}/{len(items)} items (skip {sorted(skip)})', flush=True)


def load_pack(exam_dir):
    exam_dir = Path(exam_dir)
    exam = json.loads((exam_dir / 'exam.json').read_text(encoding='utf-8'))
    if not exam.get('exam_id') or not isinstance(exam.get('items'), list):
        raise SystemExit(f'bad exam.json in {exam_dir}')
    return exam_dir, exam


def classify(fmt, question=''):
    f = (fmt or '').strip()
    q = question or ''
    if 'wypracowanie' in f.lower() or 'trzy tematy' in q.lower():
        return mf.ESSAY, None
    lines = [ln.strip() for ln in f.splitlines() if ln.strip()]
    if len(lines) == 1 and re.fullmatch(r'[A-D]', lines[0]):
        return 'closed_abcd', lines[0]
    pairs = []
    for ln in lines:
        m = re.match(r'^([A-Za-z0-9.]+)\s*[:=]\s*(.+)$', ln)
        if not m:
            return 'open_short', None
        pairs.append((m.group(1), m.group(2).strip()))
    if not pairs:
        return 'open_short', None
    vals = {v.upper() for _, v in pairs}
    if vals <= {'P', 'F'}:
        return 'true_false', {k: v[0].upper() for k, v in pairs}
    if all(re.fullmatch(r'[A-D]', v, re.I) for _, v in pairs):
        return 'closed_abcd', {k: v[0].upper() for k, v in pairs}
    return 'matching', {k: v for k, v in pairs}


def to_item(raw, root):
    typ, key = classify(raw.get('answer_format') or '', raw.get('question') or '')
    imgs = []
    for im in raw.get('images') or []:
        p = im.get('path') if isinstance(im, dict) else im
        if p:
            imgs.append(p)
    it = {
        'id': str(raw['id']),
        'type': typ,
        'task': str(raw['id']),
        'group': raw.get('group'),
        'session_key': 'submit',
        'max_points': raw.get('max_points', 1),
        'question': raw.get('question') or '',
        'context': raw.get('source_text') or '',
        'images': imgs,
        'needs_visual': bool(imgs),
        'auto_gradable': False,
        'requires_justification': True,
        '_root': str(root),
        'answer_format_raw': raw.get('answer_format') or '',
    }
    if key is not None:
        it['answer_key'] = key
    return it


def remap_closed(value, fmt):
    """Turn '1-P; 2-F' / 'A' into the syntax sketched in answer_format (not a gold key)."""
    value = (value or '').strip()
    if not value:
        return ''
    fmt = (fmt or '').strip()
    if re.fullmatch(r'[A-D]', fmt, re.I):
        m = re.search(r'\b([A-Da-d])\b', value)
        return m.group(1).upper() if m else value[:1].upper()
    parts = [p.strip() for p in re.split(r'[;\n]+', value) if p.strip()]
    mapped = []
    for p in parts:
        m = re.match(r'^([^:=\s]+)\s*[-:=]\s*(.+)$', p)
        if m:
            mapped.append(f'{m.group(1)}: {m.group(2).strip()}')
        else:
            mapped.append(p)
    return '\n'.join(mapped)


def to_submit_answer(item, output):
    text = mf.strip_thinking(output)
    if item['type'] == mf.ESSAY:
        return text[:MAX_CHARS]
    if mf.is_closed(item):
        line = mf.final_answer(text) or text
        return remap_closed(line, item.get('answer_format_raw'))[:MAX_CHARS]
    return text[:MAX_CHARS]


def validate(payload, required_ids):
    extra = set(payload) - _ALLOWED_TOP
    if extra:
        raise SystemExit(f'illegal top-level keys: {extra}')
    ids = [a['id'] for a in payload['answers']]
    if sorted(ids) != sorted(required_ids):
        raise SystemExit(f'id mismatch: missing {set(required_ids)-set(ids)} extra {set(ids)-set(required_ids)}')
    if len(ids) != len(set(ids)):
        raise SystemExit('duplicate ids')
    for a in payload['answers']:
        if set(a) != _ALLOWED_ANS or not isinstance(a['answer'], str):
            raise SystemExit(f'bad answer row {a.get("id")}')
    raw = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    if len(raw) > MAX_BYTES:
        raise SystemExit(f'answers.json {len(raw)} bytes > 1 MiB')
    return len(raw)


def write_answers(exam, gens, path):
    by_id = {g['id']: g['output'] for g in gens}
    items = {str(it['id']): to_item(it, '.') for it in exam['items']}
    answers = []
    for it in exam['items']:
        iid = str(it['id'])
        answers.append({'id': iid, 'answer': to_submit_answer(items[iid], by_id.get(iid, ''))})
    payload = {'exam_id': exam['exam_id'], 'answers': answers}
    n = validate(payload, [str(it['id']) for it in exam['items']])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    blanks = sum(1 for a in answers if not a['answer'].strip())
    print(f'wrote {path} ({n} bytes, {blanks} blank)', flush=True)
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exam-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='google/gemma-4-12B-it')
    p.add_argument('--adapter')
    p.add_argument('--load-4bit', action='store_true')
    p.add_argument('--text-only', action='store_true')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=768)
    p.add_argument('--essay-max-new-tokens', type=int, default=3072)
    p.add_argument('--image-max-soft-tokens', type=int, default=560)
    p.add_argument('--limit', type=int)
    p.add_argument('--rag-index', help='local tantivy Wikipedia index (wiki_rag.WikiIndex)')
    p.add_argument('--rag-k', type=int, default=5)
    p.add_argument('--rag-max-chars', type=int, default=3000)
    p.add_argument('--rag-skip-types', default='essay')
    p.add_argument('--rag-scripts', default=DEFAULT_RAG_SCRIPTS, help='dir with wiki_rag.py')
    a = p.parse_args(argv)

    root, exam = load_pack(a.exam_dir)
    items = [to_item(it, root) for it in exam['items']]
    if a.limit:
        items = items[: a.limit]
    if a.rag_index:
        skip = tuple(t.strip() for t in a.rag_skip_types.split(',') if t.strip())
        attach_wiki_rag(items, a.rag_index, a.rag_k, a.rag_max_chars, skip, a.rag_scripts)
    out = Path(a.output)
    gen_path = out.with_suffix('.generations.jsonl')
    done = {g['id'] for g in em.read_jsonl(gen_path)}
    todo = [it for it in items if it['id'] not in done]
    print(f'{len(items)} items, {len(done)} cached, {len(todo)} to go', flush=True)
    if todo:
        import time
        t0 = time.time()
        model, processor = em.load_model(a.model, a.adapter, a.load_4bit)
        print(f'model loaded in {time.time()-t0:.0f}s', flush=True)
        f = open(gen_path, 'a', encoding='utf-8')

        def save(rows):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            print(f'  {sum(1 for _ in em.read_jsonl(gen_path))}/{len(items)} generated', flush=True)

        def budget(it):
            return a.essay_max_new_tokens if it['type'] == mf.ESSAY else a.max_new_tokens

        em.generate(model, processor, todo, budget, a.batch_size, a.text_only, False,
                    a.image_max_soft_tokens, on_batch=save)
        f.close()
    gens = em.read_jsonl(gen_path)
    write_answers(exam, gens, out)


if __name__ == '__main__':
    main()
