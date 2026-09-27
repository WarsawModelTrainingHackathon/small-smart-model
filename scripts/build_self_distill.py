# /// script
# requires-python = ">=3.11"
# ///
"""Keep full-score answers as SFT targets. Multiple --eval-dir: pick the higher score per id.

  python scripts/build_self_distill.py \
    --eval-dir /team/runs/eval/base-train \
    --eval-dir /team/angel/eval/angel-train-closed \
    --output /team/angel/sft/self_distill-bestof.jsonl

Closed: points == max_points. Essays: points >= 0.8 * max_points.
Ties keep the first eval-dir (usually the base run). Never reads CKE `answer`.
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


def ok(rec, essay_frac):
    pts, mx = rec.get('points'), rec.get('max_points')
    if pts is None or not mx:
        return False
    if rec.get('type') == 'essay':
        return pts >= essay_frac * mx
    return pts >= mx


def collect(eval_dir):
    ev = Path(eval_dir)
    gens = {g['id']: (g.get('output') or '') for g in load_jsonl(ev / 'generations.jsonl')}
    out = {}
    for rec in load_jsonl(ev / 'graded.jsonl'):
        rec = dict(rec)
        rec['_target'] = gens.get(rec['id'], '').strip()
        rec['_src'] = str(ev)
        out[rec['id']] = rec
    return out


def pick(cands, essay_frac):
    """cands: list of graded recs in eval-dir order. Best points; tie -> first that is keepable."""
    keepable = [r for r in cands if ok(r, essay_frac) and r.get('_target')]
    if not keepable:
        return None
    best = max(r['points'] for r in keepable)
    for r in keepable:
        if r['points'] == best:
            return r
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval-dir', action='append', required=True,
                   help='run dir with graded.jsonl + generations.jsonl (repeatable; first wins ties)')
    p.add_argument('--output', required=True)
    p.add_argument('--essay-frac', type=float, default=0.8)
    a = p.parse_args(argv)
    by_id = {}
    order = []
    for d in a.eval_dir:
        got = collect(d)
        for i, rec in got.items():
            if i not in by_id:
                order.append(i)
                by_id[i] = []
            by_id[i].append(rec)
    kept, skipped = [], Counter()
    by_type, from_src = Counter(), Counter()
    for i in order:
        rec = pick(by_id[i], a.essay_frac)
        if not rec:
            skipped[by_id[i][0].get('type') or '?'] += 1
            continue
        kept.append(dict(id=rec['id'], type=rec.get('type'), target=rec['_target'], src=rec['_src']))
        by_type[rec.get('type') or '?'] += 1
        from_src[rec['_src']] += 1
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept), encoding='utf-8')
    print(f'wrote {len(kept)} -> {out}')
    print('kept', dict(by_type))
    print('from', dict(from_src))
    print('skipped', dict(skipped))


if __name__ == '__main__':
    main()
