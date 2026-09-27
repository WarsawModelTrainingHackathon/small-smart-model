"""CPU smoke tests for scripts/train_lora.py and scripts/export_submission.py on a tiny random Gemma 4."""
import json

import pytest

import eval_matura as em
import export_submission as ex
import matura_format as mf
import train_lora as tl
from conftest import bnb_cpu_ok

needs_bnb = pytest.mark.skipif(not bnb_cpu_ok(), reason='bitsandbytes 4-bit does not run on this CPU: 4-bit path skipped')


def train_args(tiny_model, mini_data, out, *extra):
    return tl.parse_args(['--model', tiny_model, '--data', str(mini_data), '--output', str(out), '--epochs', '1',
                          '--grad-accum', '4', '--lr', '1e-3', '--image-max-soft-tokens', '70',
                          '--dev-max-new-tokens', '8', '--log-every', '1', *extra])


def test_target_loss_is_assistant_only(tiny_model, items):
    """Our logits_to_keep loss == HF loss with prompt tokens masked to -100."""
    model, processor = em.load_model(tiny_model, device='cpu')
    a = tl.parse_args(['--output', 'x', '--text-only'])
    enc, n = tl.encode_example(processor, items['2015-maj-f2015-R-zad18'], 'Krótko.\nOdpowiedź: C', a, 'cpu')
    labels = enc['input_ids'].clone()
    labels[:, :-n] = -100
    ref = model(**enc, labels=labels).loss.item()
    assert abs(tl.target_loss(model, enc, n).item() - ref) < 1e-4
    tok = em.tokenizer_of(processor)
    assert tok.decode(enc['input_ids'][0, -n:]).startswith('Krótko.')
    assert '<turn|>' in tok.decode(enc['input_ids'][0, -n:])


def test_lora_targets_language_only(tiny_model):
    model, _ = em.load_model(tiny_model, device='cpu')
    names = tl.lora_target_names(model, tl.DEFAULT_TARGETS.split(','))
    assert names and all('language_model' in n for n in names)
    assert len(names) == 2 * 7


def test_train_smoke_text_only(tiny_model, mini_data, tmp_path, monkeypatch):
    seen = []
    orig = mf.load_data

    def spy(path, splits=None):
        seen.append(splits)
        return orig(path, splits)
    monkeypatch.setattr(mf, 'load_data', spy)
    rep = tl.run(train_args(tiny_model, mini_data, tmp_path / 't', '--no-4bit', '--text-only'))
    assert seen and all(s is not None and 'test' not in s for s in seen)  # the test split is never loaded
    assert (tmp_path / 't' / 'best_adapter' / 'adapter_config.json').exists()
    assert (tmp_path / 't' / 'final_adapter' / 'adapter_model.safetensors').exists()
    saved = json.loads((tmp_path / 't' / 'report.json').read_text())
    assert saved['steps'] == rep['steps'] > 0
    assert all(h['loss'] == h['loss'] for h in saved['history'])  # finite, not NaN
    assert [d['tag'] for d in saved['dev']] == ['epoch0-base', 'epoch1']
    assert 'dev_closed' in saved['dev'][1] and saved['dev'][1]['dev_loss'] is not None
    # the trained adapter loads in the eval harness
    model, _ = em.load_model(tiny_model, adapter=str(tmp_path / 't' / 'best_adapter'), device='cpu')
    assert type(model).__name__.startswith('PeftModel')


def test_train_max_seconds(tiny_model, mini_data, tmp_path):
    rep = tl.run(train_args(tiny_model, mini_data, tmp_path / 'ms', '--no-4bit', '--text-only', '--epochs', '5',
                            '--grad-accum', '1', '--max-seconds', '1', '--skip-base-eval'))
    assert rep['steps'] < 5 * rep['n_train']


@needs_bnb
def test_train_smoke_qlora_with_images(tiny_model, mini_data, tmp_path):
    rep = tl.run(train_args(tiny_model, mini_data, tmp_path / 'q', '--skip-base-eval'))
    assert rep['quantization'] == 'nf4-4bit' and rep['steps'] > 0
    assert (tmp_path / 'q' / 'best_adapter' / 'adapter_model.safetensors').exists()


def _logits(model, processor, text='Kto wygrał pod Grunwaldem? Odpowiedź: Jagiełło'):
    import torch
    enc = em.tokenizer_of(processor)([text], return_tensors='pt')
    with torch.no_grad():
        return model(**enc).logits.float()


@needs_bnb
@pytest.mark.parametrize('mode', ['dequant', 'bf16'])
def test_export_merged_4bit_matches_adapter(tiny_model, tmp_path, mode):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    base, processor = em.load_model(tiny_model, load_4bit=True, device='cpu')
    base_logits = _logits(base, processor)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=tl.lora_target_names(base, ['q_proj', 'v_proj', 'down_proj'])))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'lora_B' in n:
                p.normal_(0, 0.3)
    adapted_logits = _logits(model, processor)
    model.save_pretrained(tmp_path / 'adapter')
    del model, base
    out = tmp_path / f'sub-{mode}'
    info = ex.export(ex.parse_args(['--model', tiny_model, '--adapter', str(tmp_path / 'adapter'), '--output', str(out),
                                    '--merge-mode', mode, '--verify']))
    assert info['linear4bit_layers'] == 14 and info['size_bytes'] <= 8e9
    assert 'quantization_config' in json.loads((out / 'config.json').read_text())
    merged, processor2 = em.load_model(str(out), device='cpu')
    assert ex.count_4bit(merged) == 14
    merged_logits = _logits(merged, processor2)
    d_adapted = (merged_logits - adapted_logits).norm()
    d_base = (base_logits - adapted_logits).norm()
    assert d_adapted < 0.5 * d_base, (float(d_adapted), float(d_base))


@needs_bnb
def test_export_base_and_size_limit(tiny_model, tmp_path):
    info = ex.export(ex.parse_args(['--model', tiny_model, '--output', str(tmp_path / 'base4')]))
    assert info['kind'] == 'merged-4bit-nf4' and info['adapter'] is None
    with pytest.raises(SystemExit, match='limit'):
        ex.export(ex.parse_args(['--model', tiny_model, '--output', str(tmp_path / 'toobig'), '--max-gb', '1e-6']))


def test_export_adapter_only_and_estimate(tiny_model, tmp_path):
    from peft import LoraConfig, get_peft_model
    base, _ = em.load_model(tiny_model, device='cpu')
    get_peft_model(base, LoraConfig(r=4, target_modules=['q_proj'])).save_pretrained(tmp_path / 'ad')
    info = ex.export(ex.parse_args(['--model', tiny_model, '--adapter', str(tmp_path / 'ad'), '--output',
                                    str(tmp_path / 'ad-out'), '--adapter-only']))
    assert info['kind'] == 'adapter' and (tmp_path / 'ad-out' / 'adapter_config.json').exists()
    est = ex.export(ex.parse_args(['--model', tiny_model, '--estimate']))
    assert est['total_params'] > est['quantized_params'] > 0
