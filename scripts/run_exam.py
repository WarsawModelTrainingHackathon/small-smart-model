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
"""Sunday/mock pack -> answers.json. Reuses eval_matura generate. No training.

  uv run scripts/run_exam.py --exam /team/exam/history-2023-mock-v1 --load-4bit --limit 1
  uv run scripts/run_exam.py --exam ~/Downloads/history-2023-mock-v1 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_PAIR = re.compile(r'^([A-Za-z0-9.]+)\s*:\s*(.+)$')


def _pairs(fmt: str) -> dict[str, str]:
    out = {}
    for line in (fmt or '').splitlines():
        m = _PAIR.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def exam_item_to_cke(raw: dict, root: Path) -> dict:
    fmt = (raw.get('answer_format') or '').strip()
    pairs = _pairs(fmt)
    q = raw.get('question') or ''
    if raw.get('id') == '26' or 'minimum 300' in q.lower() or 'wypracowanie' in fmt.lower():
        typ, key = 'essay', None
    elif pairs and all(v.upper() in ('P', 'F') for v in pairs.values()):
        typ, key = 'true_false', {k: 'P' for k in pairs}
    elif pairs and fmt[:1].isalpha() and not fmt.startswith('Tekst'):
        typ, key = 'matching', {k: '…' for k in pairs}
    elif pairs:
        typ, key = 'closed_abcd', {k: 'A' for k in pairs}
    else:
        typ, key = ('open_long' if (raw.get('max_points') or 1) >= 2 else 'open_short'), None
    images = [im['path'] for im in (raw.get('images') or [])]
    item = dict(
        id=str(raw['id']),
        type=typ,
        task=str(raw['id']),
        group=raw.get('group'),
        max_points=raw.get('max_points') or 1,
        question=q,
        context=(raw.get('source_text') or '').strip(),
        images=images,
        _root=str(root.resolve()),
        session_key='mock',
        auto_gradable=False,
        requires_justification=True,
    )
    if key:
        item['answer_key'] = key
    return item


def load_exam(exam_dir: Path) -> tuple[dict, list[dict], dict]:
    exam_path = exam_dir / 'exam.json'
    exam = json.loads(exam_path.read_text(encoding='utf-8'))
    template = json.loads((exam_dir / 'answers-template.json').read_text(encoding='utf-8'))
    if exam['exam_id'] != template['exam_id']:
        raise SystemExit('exam_id mismatch between exam.json and answers-template.json')
    items = [exam_item_to_cke(it, exam_dir) for it in exam['items']]
    return exam, items, template


def verify_images(items: list[dict], exam: dict, exam_dir: Path) -> None:
    by_id = {str(it['id']): it for it in exam['items']}
    for it in items:
        raw = by_id[it['id']]
        for im in raw.get('images') or []:
            path = exam_dir / im['path']
            if not path.is_file():
                raise SystemExit(f'missing image {path}')
            want = (im.get('sha256') or '').lower()
            if want:
                got = hashlib.sha256(path.read_bytes()).hexdigest()
                if got != want:
                    raise SystemExit(f'sha256 mismatch {path}: {got} != {want}')


def to_answer_text(item: dict, output: str) -> str:
    import matura_format as mf
    text = mf.strip_thinking(output)
    if item['type'] in mf.CLOSED_TYPES:
        parsed = mf.final_answer(text)
        if parsed:
            return parsed
    return text.strip()


def write_answers(template: dict, by_id: dict[str, str], dest: Path) -> None:
    answers = []
    for row in template['answers']:
        i = str(row['id'])
        answers.append({'id': i, 'answer': by_id.get(i, row.get('answer') or '')})
    dest.write_text(
        json.dumps({'exam_id': template['exam_id'], 'answers': answers}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exam', required=True, type=Path, help='unzipped pack (exam.json + images/ + template)')
    p.add_argument('--output', type=Path, help='answers.json path (default: <exam>/answers.json)')
    p.add_argument('--model', default='google/gemma-4-12B-it')
    p.add_argument('--adapter', help='omit for base model (no extra training)')
    p.add_argument('--load-4bit', action='store_true')
    p.add_argument('--limit', type=int, help='first N items (smoke)')
    p.add_argument('--ids', help='comma list of mock ids, e.g. 1,2.1')
    p.add_argument('--text-only', action='store_true')
    p.add_argument('--thinking', default='off', choices=['on', 'off'])
    p.add_argument('--image-max-soft-tokens', type=int, default=560, choices=(70, 140, 280, 560, 1120))
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-new-tokens', type=int, default=768)
    p.add_argument('--essay-max-new-tokens', type=int, default=3072)
    p.add_argument('--attn', default='sdpa')
    p.add_argument('--device')
    p.add_argument('--dry-run', action='store_true', help='load JSON+PNGs, write empty answers.json, no GPU')
    a = p.parse_args(argv)

    exam_dir = a.exam.expanduser().resolve()
    exam, items, template = load_exam(exam_dir)
    verify_images(items, exam, exam_dir)
    if a.ids:
        keep = set(a.ids.split(','))
        items = [it for it in items if it['id'] in keep]
    if a.limit:
        items = items[: a.limit]
    dest = a.output or (exam_dir / 'answers.json')
    print(f'{exam["exam_id"]}: {len(exam["items"])} items, running {len(items)} -> {dest}', flush=True)
    if a.dry_run:
        n_img = sum(len(it['images']) for it in items)
        print(f'dry-run ok, {n_img} image refs, types={ {it["type"] for it in items} }', flush=True)
        write_answers(template, {}, dest)
        print(f'wrote empty template copy {dest}', flush=True)
        return

    import eval_matura as ev
    import matura_format as mf
    model, processor = ev.load_model(a.model, a.adapter, a.load_4bit, device=a.device, attn=a.attn)
    gens = ev.generate(
        model, processor, items,
        lambda it: (a.essay_max_new_tokens if it['type'] == 'essay' else a.max_new_tokens),
        a.batch_size, a.text_only, a.thinking == 'on', a.image_max_soft_tokens,
        on_batch=lambda rows: print(f'  {rows[-1]["id"]} +{rows[-1]["new_tokens"]} tok', flush=True),
    )
    filled = {g['id']: to_answer_text(next(it for it in items if it['id'] == g['id']), g['output']) for g in gens}
    write_answers(template, filled, dest)
    nonempty = sum(1 for v in filled.values() if v.strip())
    print(f'wrote {dest} ({nonempty}/{len(template["answers"])} non-empty slots)', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
