import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
FIXTURE = ROOT / 'tests' / 'fixtures' / 'items.json'


@pytest.fixture(scope='session')
def items():
    """Real CKE items (text fields only; images are generated) keyed by id."""
    return {r['id']: r for r in json.loads(FIXTURE.read_text(encoding='utf-8'))}


@pytest.fixture(scope='session')
def mini_data(tmp_path_factory, items):
    """A data.json in the dataset layout, built from the fixture items + synthetic PNGs."""
    import random
    from PIL import Image
    root = tmp_path_factory.mktemp('data') / 'matura-historia'
    (root / 'images').mkdir(parents=True)
    rng = random.Random(0)
    splits = {'train': [], 'dev': [], 'test': []}
    for r in items.values():
        r = dict(r)
        for p in r['images']:
            Image.frombytes('RGB', (320, 200), bytes(rng.randrange(256) for _ in range(320 * 200 * 3))).save(root / p)
        splits[r['split']].append(r)
    (root / 'data.json').write_text(json.dumps(splits, ensure_ascii=False), encoding='utf-8')
    return root / 'data.json'


@pytest.fixture(scope='session')
def tiny_model(tmp_path_factory):
    import tiny_gemma
    return str(tiny_gemma.build(tmp_path_factory.mktemp('tiny') / 'gemma4-unified-tiny'))


def full_data_path():
    for p in (os.environ.get('MATURA_DATA'), ROOT / 'data/matura-historia/data.json', '/tmp/mh/data/matura-historia/data.json'):
        if p and Path(p).exists():
            return Path(p)
    return None


@pytest.fixture(scope='session')
def full_data():
    p = full_data_path()
    if p is None:
        pytest.skip('full dataset not found (set MATURA_DATA or `git worktree add /tmp/mh origin/data/matura-historia`)')
    return p


def bnb_cpu_ok():
    try:
        import bitsandbytes  # noqa: F401
        import torch
        from transformers import BitsAndBytesConfig  # noqa: F401
        lin = bitsandbytes.nn.Linear4bit(64, 64, compute_dtype=torch.float32, quant_type='nf4')
        lin = lin.to('cpu')
        lin.weight = bitsandbytes.nn.Params4bit(torch.randn(64, 64), requires_grad=False, quant_type='nf4').to('cpu')
        lin(torch.randn(1, 64))
        return True
    except Exception:  # noqa: BLE001
        return False
