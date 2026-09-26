"""End-to-end CPU smoke test of scripts/eval_matura.py on a tiny random Gemma 4 model."""
import json

import pytest

import compare_runs
import eval_matura
from conftest import bnb_cpu_ok


def run(tiny_model, mini_data, out, *extra):
    return eval_matura.main(['--model', tiny_model, '--data', str(mini_data), '--judge', 'none',
                             '--max-new-tokens', '12', '--essay-max-new-tokens', '12', '--image-max-soft-tokens', '70',
                             '--batch-size', '2', '--output', str(out), *extra])


def test_eval_smoke_resume_regrade(tiny_model, mini_data, tmp_path):
    out = tmp_path / 'run'
    rep = run(tiny_model, mini_data, out, '--split', 'dev', '--limit', '5', '--label', 'tiny-dev')
    for f in ('run.json', 'generations.jsonl', 'graded.jsonl', 'report.json', 'report.md'):
        assert (out / f).exists(), f
    gens = [json.loads(x) for x in (out / 'generations.jsonl').read_text().splitlines()]
    assert len(gens) == 5 and all('output' in g for g in gens)
    assert rep['meta']['n_items'] == 5
    # resume: nothing left to generate, same report
    rep2 = run(tiny_model, mini_data, out, '--split', 'dev', '--limit', '5', '--label', 'tiny-dev')
    assert len((out / 'generations.jsonl').read_text().splitlines()) == 5
    assert rep2['summary']['n_generations'] == 5
    # re-grade from saved generations only
    rep3 = run(tiny_model, mini_data, out, '--split', 'dev', '--limit', '5', '--regrade')
    assert rep3['summary']['n_generations'] == 5


def test_eval_test_split_text_only_closed(tiny_model, mini_data, tmp_path):
    out = tmp_path / 'closed'
    rep = run(tiny_model, mini_data, out, '--split', 'test', '--text-only', '--label', 'tiny-test-text')
    s = rep['summary']
    assert s['closed_unparseable']['of'] >= 1  # random model -> unparseable, but counted
    assert s['n_essay_groups'] == 1 and s['by_type']['essay']['n'] == 1  # 3 topics -> one essay
    text = compare_runs.compare([out, out], items=True)
    assert 'total' in text


@pytest.mark.skipif(not bnb_cpu_ok(), reason='bitsandbytes 4-bit does not run on this CPU')
def test_eval_4bit(tiny_model, mini_data, tmp_path):
    rep = run(tiny_model, mini_data, tmp_path / 'q', '--split', 'dev', '--limit', '3', '--load-4bit')
    assert rep['meta']['load_4bit'] is True and rep['summary']['n_generations'] == 3
