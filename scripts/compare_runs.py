# /// script
# requires-python = ">=3.11"
# ///
"""Compare matura eval runs side by side.

  python scripts/compare_runs.py runs/eval/base-4bit runs/eval/lora-4bit [--items] [--md out.md]

Each argument is a run dir (with report.json and graded.jsonl) or a report.json path.
With --items, also lists per-item point differences between the first two runs.
"""
import argparse
import json
from pathlib import Path


def load(path):
    p = Path(path)
    rep = p if p.suffix == '.json' else p / 'report.json'
    run_dir = rep.parent
    report = json.loads(rep.read_text(encoding='utf-8'))
    graded = {}
    if (run_dir / 'graded.jsonl').exists():
        for line in (run_dir / 'graded.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                g = json.loads(line)
                graded[g['id']] = g
    return report.get('meta', {}).get('label') or run_dir.name, report['summary'], graded


def cell(b):
    if not b:
        return '–'
    return f"{b['points']}/{b['max']} ({b['pct']}%)" + (f" +{b['ungraded']}?" if b.get('ungraded') else '')


def compare(paths, items=False):
    runs = [load(p) for p in paths]
    names = [r[0] for r in runs]
    head = '| | ' + ' | '.join(names) + ' |'
    sep = '|---|' + '---|' * len(runs)
    out = ['# Matura eval comparison', '', head, sep]
    out.append('| **total** | ' + ' | '.join(f"**{cell(s['total'])}**" for _, s, _ in runs) + ' |')
    out.append('| closed (auto) | ' + ' | '.join(cell(s['closed_items']) for _, s, _ in runs) + ' |')
    out.append('| unparseable closed | ' + ' | '.join(f"{s['closed_unparseable']['n']}/{s['closed_unparseable']['of']}" for _, s, _ in runs) + ' |')
    for section in ('by_type', 'by_session', 'by_visual'):
        keys = sorted({k for _, s, _ in runs for k in s[section]})
        out += ['', f'## {section}', '', head, sep]
        for k in keys:
            out.append(f'| {k} | ' + ' | '.join(cell(s[section].get(k)) for _, s, _ in runs) + ' |')
    if items and len(runs) >= 2:
        a, b = runs[0][2], runs[1][2]
        out += ['', f'## Per-item: {names[0]} → {names[1]}', '', '| id | type | max | A | B | Δ |', '|---|---|---|---|---|---|']
        for i in sorted(set(a) & set(b)):
            pa, pb = a[i].get('points'), b[i].get('points')
            if pa != pb:
                d = (pb - pa) if None not in (pa, pb) else '?'
                out.append(f"| {i} | {a[i]['type']} | {a[i]['max_points']} | {pa} | {pb} | {d} |")
    return '\n'.join(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--items', action='store_true')
    p.add_argument('--md', help='also write the table to this file')
    a = p.parse_args()
    text = compare(a.runs, a.items)
    print(text)
    if a.md:
        Path(a.md).write_text(text + '\n', encoding='utf-8')
