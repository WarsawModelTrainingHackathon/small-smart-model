# /// script
# requires-python = ">=3.11"
# ///
"""Keep full-score Gemma (or regraded) answers as SFT targets.

  python scripts/build_self_distill.py \
    --eval-dir /team/runs/eval/base-train \
    --output /team/angel/sft/self_distill.jsonl

Closed items: points == max_points. Essays: points >= 0.8 * max_points.
Does not read CKE `answer` fields. Target text is the saved generation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def keep(rec, essay_frac):
    pts, mx = rec.get('points'), rec.get('max_points')
    if pts is None or not mx:
        return False
    if rec.get('type') == 'essay':
        return pts >= essay_frac * mx
    return pts >= mx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval-dir', required=True, help='run dir with graded.jsonl + generations.jsonl')
    p.add_argument('--output', required=True)
    p.add_argument('--essay-frac', type=float, default=0.8)
    a = p.parse_args()
    ev = Path(a.eval_dir)
    gens = {g['id']: g.get('output') or '' for g in load_jsonl(ev / 'generations.jsonl')}
    kept, skipped = [], Counter()
    by_type = Counter()
    for rec in load_jsonl(ev / 'graded.jsonl'):
        if not keep(rec, a.essay_frac):
            skipped[rec.get('type') or '?'] += 1
            continue
        text = gens.get(rec['id'], '').strip()
        if not text:
            skipped['no_generation'] += 1
            continue
        kept.append(dict(id=rec['id'], type=rec.get('type'), target=text))
        by_type[rec.get('type') or '?'] += 1
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept), encoding='utf-8')
    print(f'wrote {len(kept)} -> {out}')
    print('kept', dict(by_type))
    print('skipped', dict(skipped))


if __name__ == '__main__':
    main()
