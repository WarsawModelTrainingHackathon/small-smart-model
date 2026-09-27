import json
from pathlib import Path

import build_self_distill as bsd


def _dump(dir, graded, gens):
    dir.mkdir(parents=True)
    (dir / 'graded.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in graded), encoding='utf-8')
    (dir / 'generations.jsonl').write_text(''.join(json.dumps(g) + '\n' for g in gens), encoding='utf-8')


def test_bestof_prefers_higher_score_and_keeps_first_on_tie(tmp_path):
    base = tmp_path / 'base'
    ada = tmp_path / 'ada'
    _dump(base, [
        {'id': 'tf', 'type': 'true_false', 'points': 1, 'max_points': 2},
        {'id': 'ab', 'type': 'closed_abcd', 'points': 1, 'max_points': 1},
        {'id': 'open', 'type': 'open_short', 'points': 1, 'max_points': 1},
    ], [
        {'id': 'tf', 'output': 'BASE-TF'},
        {'id': 'ab', 'output': 'BASE-AB'},
        {'id': 'open', 'output': 'BASE-OPEN'},
    ])
    _dump(ada, [
        {'id': 'tf', 'type': 'true_false', 'points': 2, 'max_points': 2},
        {'id': 'ab', 'type': 'closed_abcd', 'points': 1, 'max_points': 1},
        {'id': 'miss', 'type': 'matching', 'points': 0, 'max_points': 1},
    ], [
        {'id': 'tf', 'output': 'ADA-TF'},
        {'id': 'ab', 'output': 'ADA-AB'},
        {'id': 'miss', 'output': 'ADA-MISS'},
    ])
    out = tmp_path / 'out.jsonl'
    bsd.main(['--eval-dir', str(base), '--eval-dir', str(ada), '--output', str(out)])
    rows = {r['id']: r for r in map(json.loads, out.read_text().splitlines())}
    assert rows['tf']['target'] == 'ADA-TF'
    assert rows['ab']['target'] == 'BASE-AB'
    assert rows['open']['target'] == 'BASE-OPEN'
    assert 'miss' not in rows
    assert Path(rows['tf']['src']) == ada
