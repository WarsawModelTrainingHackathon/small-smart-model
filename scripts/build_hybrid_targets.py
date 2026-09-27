#!/usr/bin/env python3
"""Add only missing train-split matching targets to a self-distillation JSONL.

This isolates a small CKE-key supplement for a dev experiment while retaining
the model's own high-scoring targets for every other item. The output is
compatible with ``train_lora.py --targets-jsonl``.

Example:
  python scripts/build_hybrid_targets.py \
      --data /tmp/mh/data/matura-historia/data.json \
      --self-distill runs/sft/self_distill.jsonl \
      --output runs/sft/self_distill-plus-matching.jsonl

Only the train split is consulted. Distillation IDs not present in train are
rejected so dev/test targets cannot silently enter this training file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matura_format as mf


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='benchmark data.json; only train rows are used')
    parser.add_argument('--self-distill', required=True, help='existing train-only self-distillation JSONL')
    parser.add_argument('--output', required=True, help='new JSONL for train_lora.py --targets-jsonl')
    args = parser.parse_args()

    source_path = Path(args.self_distill)
    output_path = Path(args.output)
    if source_path.resolve() == output_path.resolve():
        parser.error('--output must be a new path; the source self-distillation file is never overwritten')

    train = mf.load_data(args.data, splits=['train']).get('train', [])
    train_by_id = {item['id']: item for item in train}
    if not train_by_id:
        parser.error('no train items found in --data')
    if len(train_by_id) != len(train):
        parser.error('train split contains duplicate IDs')

    distilled = read_jsonl(source_path)
    if not distilled:
        parser.error('self-distillation file is empty')

    seen: set[str] = set()
    output: list[dict] = []
    for row in distilled:
        item_id = row.get('id')
        target = (row.get('target') or '').strip()
        if not item_id or not target:
            parser.error('each self-distillation row must have a non-empty id and target')
        if item_id in seen:
            parser.error(f'duplicate self-distillation id: {item_id}')
        if item_id not in train_by_id:
            parser.error(f'self-distillation id is not in the train split: {item_id}')
        train_type = train_by_id[item_id].get('type')
        if row.get('type') and row['type'] != train_type:
            parser.error(f'self-distillation type does not match train item {item_id}')
        seen.add(item_id)
        output.append(dict(id=item_id, type=train_type,
                           target=target, source='self_distill'))

    supplements = 0
    for item in train:
        if item.get('type') != 'matching' or item['id'] in seen:
            continue
        target = mf.build_target(item)
        if not target:
            continue
        seen.add(item['id'])
        output.append(dict(id=item['id'], type='matching', target=target, source='cke_matching'))
        supplements += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in output),
                           encoding='utf-8')
    print(f'self-distill targets: {len(distilled)}')
    print(f'added train matching CKE targets: {supplements}')
    print(f'total targets: {len(output)} -> {output_path}')


if __name__ == '__main__':
    main()
