# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build self-distillation targets from a graded eval run on the train split.

Keeps the model's own answers that the judge / key scored at full points (essays: >= --essay-min share),
so fine-tuning reinforces good answers in the model's own style instead of the short CKE example answers.

  uv run scripts/build_self_distill.py runs/eval/base-train --output runs/sft/self_distill.jsonl
"""
import argparse
import json
import re
from pathlib import Path

THINK = re.compile(r'<\|channel\>.*?<channel\|>', re.S)


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', help='eval_matura.py output dir with generations.jsonl + graded.jsonl')
    p.add_argument('--output', required=True)
    p.add_argument('--essay-min', type=float, default=0.8, help='min share of max points to keep an essay')
    a = p.parse_args()
    run = Path(a.run_dir)
    gens = {g['id']: g['output'] for g in read(run / 'generations.jsonl')}
    kept, stats = [], {}
    for g in read(run / 'graded.jsonl'):
        pts, mx = g.get('points'), g.get('max_points') or 0
        ok = pts is not None and mx and (pts >= mx if g['type'] != 'essay' else pts >= a.essay_min * mx)
        s = stats.setdefault(g['type'], [0, 0]); s[1] += 1
        text = THINK.sub('', gens.get(g['id'], '')).strip()
        if ok and text:
            s[0] += 1
            kept.append(dict(id=g['id'], type=g['type'], target=text))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept), encoding='utf-8')
    print(f'kept {len(kept)} targets -> {a.output}')
    for t, (k, n) in sorted(stats.items()):
        print(f'  {t}: {k}/{n}')


if __name__ == '__main__':
    main()
