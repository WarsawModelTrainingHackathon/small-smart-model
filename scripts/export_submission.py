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
"""Export the submission: base + LoRA merged, saved as a 4-bit (bitsandbytes nf4) checkpoint <= 8 GB.

  uv run scripts/export_submission.py --adapter runs/train/qlora/best_adapter --output outputs/submission-4bit
  uv run scripts/export_submission.py --output outputs/base-4bit                  # no adapter: plain 4-bit base
  uv run scripts/export_submission.py --adapter ... --output outputs/adapter --adapter-only

merge modes:
  dequant (default) - load the base in nf4 exactly as during QLoRA training, merge the LoRA delta into the
                      dequantized weights and re-quantize (peft merge_and_unload on Linear4bit).
  bf16              - merge in bf16, save a temporary bf16 copy (~24 GB disk), reload it in nf4 and save.

The size of the output folder is printed and asserted <= --max-gb (decimal GB, 8 GB = 8e9 bytes).
The exported folder loads with plain `from_pretrained` (quantization_config is stored in config.json);
evaluate it with: uv run scripts/eval_matura.py --model <output> ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_matura as em  # noqa: E402


def dir_size(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def check_size(path, max_gb):
    size = dir_size(path)
    print(f'size on disk: {size / 1e9:.3f} GB ({size / 2**30:.3f} GiB) in {path}  [limit {max_gb} GB]', flush=True)
    for f in sorted(Path(path).rglob('*')):
        if f.is_file() and f.stat().st_size > 50e6:
            print(f'  {f.name}: {f.stat().st_size / 1e9:.3f} GB')
    if size > max_gb * 1e9:
        raise SystemExit(f'ERROR: {size / 1e9:.3f} GB > {max_gb} GB limit')
    return size


def count_4bit(model):
    return sum(type(m).__name__ == 'Linear4bit' for m in model.modules())


NF4_BITS = 4 + 8 / 64 + 32 / (64 * 256)  # nf4 + double-quantized absmax (blocksize 64)


def estimate(model_id):
    """Predicted nf4 checkpoint size from the config alone (meta device, no weights downloaded)."""
    import re
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id)
    with torch.device('meta'):
        try:
            model = AutoModelForImageTextToText.from_config(cfg)
        except ValueError:
            model = AutoModelForCausalLM.from_config(cfg)
    quant = other = 0
    seen = set()
    for name, module in model.named_modules():
        for pname, p in module.named_parameters(recurse=False):
            if id(p) in seen:
                continue
            seen.add(id(p))
            skipped = any(re.match(k, name) or name.endswith(k) for k in em.SKIP_QUANT)
            if isinstance(module, torch.nn.Linear) and pname == 'weight' and not skipped:
                quant += p.numel()
            else:
                other += p.numel()
    size = quant * NF4_BITS / 8 + other * 2
    res = dict(quantized_params=quant, bf16_params=other, total_params=quant + other, estimated_bytes=int(size),
               estimated_gb=round(size / 1e9, 3), estimated_gib=round(size / 2**30, 3))
    print(json.dumps(res, indent=2), flush=True)
    return res


def export(a):
    import torch
    t0 = time.time()
    if a.estimate:
        res = estimate(a.model)
        if res['estimated_bytes'] > a.max_gb * 1e9:
            raise SystemExit(f"WARNING: estimated {res['estimated_gb']} GB > {a.max_gb} GB")
        return res
    if not a.output:
        raise SystemExit('--output is required')
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty')
    out.mkdir(parents=True, exist_ok=True)
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if a.adapter_only:
        if not a.adapter:
            raise SystemExit('--adapter-only needs --adapter')
        shutil.copytree(a.adapter, out, dirs_exist_ok=True)
        size = check_size(out, a.max_gb)
        info = dict(kind='adapter', base_model=a.model, adapter=a.adapter, size_bytes=size)
        (out / 'submission_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        return info

    if a.merge_mode == 'dequant' or not a.adapter:
        model, processor = em.load_model(a.model, a.adapter, load_4bit=True, device=device, dtype=torch.bfloat16)
        if a.adapter:
            model = model.merge_and_unload()
        n4 = count_4bit(model)
        model.save_pretrained(out, max_shard_size=a.max_shard_size)
    else:
        with tempfile.TemporaryDirectory(dir=a.tmp_dir) as tmp:
            dtype = torch.bfloat16
            model, processor = em.load_model(a.model, a.adapter, load_4bit=False, device=device, dtype=dtype)
            model = model.merge_and_unload()
            model.save_pretrained(tmp, max_shard_size=a.max_shard_size)
            processor.save_pretrained(tmp)
            del model
            model, processor = em.load_model(tmp, load_4bit=True, device=device, dtype=dtype)
            n4 = count_4bit(model)
            model.save_pretrained(out, max_shard_size=a.max_shard_size)
    processor.save_pretrained(out)
    if n4 == 0:
        raise SystemExit('ERROR: no Linear4bit layers in the exported model')
    cfg = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    if 'quantization_config' not in cfg:
        raise SystemExit('ERROR: quantization_config missing from config.json')
    size = check_size(out, a.max_gb)
    info = dict(kind='merged-4bit-nf4', base_model=a.model, adapter=a.adapter, merge_mode=a.merge_mode if a.adapter else None,
                linear4bit_layers=n4, quantization_config=cfg['quantization_config'], size_bytes=size,
                size_gb=round(size / 1e9, 3), seconds=round(time.time() - t0, 1))
    (out / 'submission_info.json').write_text(json.dumps(info, indent=2, default=str), encoding='utf-8')
    del model
    if a.verify:
        verify(out, a, device)
    print(json.dumps({k: v for k, v in info.items() if k != 'quantization_config'}, indent=2), flush=True)
    return info


def verify(out, a, device):
    """Reload the exported folder like a grader would and answer one prompt."""
    import matura_format as mf
    model, processor = em.load_model(str(out), device=device)
    assert count_4bit(model) > 0, 'reloaded model is not 4-bit'
    if a.data:
        item = next(it for it in mf.load_data(a.data, splits=['dev'])['dev'] if mf.auto_gradable(it))
        g = em.generate(model, processor, [item], 64, text_only=True)[0]
        print(f"verify ({item['id']}): {g['output'][:200]!r}", flush=True)
    else:
        tok = em.tokenizer_of(processor)
        enc = tok(['Kto wygrał bitwę pod Grunwaldem?'], return_tensors='pt').to(next(model.parameters()).device)
        model.generate(**enc, max_new_tokens=4, do_sample=False)
    print('verify: reload OK', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='google/gemma-4-12B-it', help='base model (as used for training)')
    p.add_argument('--adapter', help='LoRA adapter dir (omit to export the plain 4-bit base)')
    p.add_argument('--output', help='output dir (required unless --estimate)')
    p.add_argument('--estimate', action='store_true', help='only predict the nf4 size from the config (no weights)')
    p.add_argument('--adapter-only', action='store_true', help='only save the adapter (no merge)')
    p.add_argument('--merge-mode', default='dequant', choices=['dequant', 'bf16'])
    p.add_argument('--max-gb', type=float, default=8.0)
    p.add_argument('--max-shard-size', default='2GB')
    p.add_argument('--tmp-dir', help='where the bf16 merge-mode temp copy goes (needs ~2x model size)')
    p.add_argument('--verify', action='store_true', help='reload the export and generate once')
    p.add_argument('--data', help='with --verify: answer one dev item')
    p.add_argument('--device')
    return p.parse_args(argv)


if __name__ == '__main__':
    export(parse_args())
