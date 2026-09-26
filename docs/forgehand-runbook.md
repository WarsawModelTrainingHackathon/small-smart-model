# Forgehand runbook: gemma-4-12B-it → matura historia QLoRA → ≤ 8 GB submission

> **GPU time costs money (gpu-l40s-small ≈ 1.86 USD/h) and needs the team's OK before you start a session.**
> Always finish with `fh session stop <id>` – an idle box keeps billing.

Plan: evaluate the untouched model in 4-bit (the form we'd submit), QLoRA-train on the train split,
pick the adapter on dev, evaluate on test, then merge + save a 4-bit nf4 checkpoint and check it's ≤ 8 GB.

| step | GPU time (estimate, L40S 48 GB) | cost @ 1.86 USD/h |
|---|---|---|
| start box, setup, download 12B into `/team/hf-cache` (once per team) | 10–20 min | 0.3–0.6 USD |
| size estimate + plain 4-bit base export (fallback submission) | 5–10 min | 0.2–0.3 USD |
| baseline eval, base 4-bit, test (81 items incl. 6 essays) | 20–30 min | 0.6–0.9 USD |
| QLoRA training, 2 epochs (~535 examples/epoch) | 40–60 min | 1.2–1.9 USD |
| eval adapter 4-bit on test | 20–30 min | 0.6–0.9 USD |
| export merged 4-bit + verify eval on dev | 10–15 min | 0.3–0.5 USD |
| **total** | **~2–3 h** | **~4–6 USD** |

Timing is a rough estimate (not measured on this box yet); watch the first logged training steps
(`elapsed`) and scale. Judge grading through an API costs extra (team key), `--judge hf` costs GPU time.

## 0. Start and connect

```bash
fh session start matura --class gpu-l40s-small --wait     # prints the session id
fh session ssh <id>
```

## 1. Setup on the box

```bash
nvidia-smi                                   # L40S, check free memory and driver version
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=$HOME/.local/bin:$PATH
git clone https://github.com/WarsawModelTrainingHackathon/small-smart-model.git && cd small-smart-model
git checkout training/gemma-lora
git fetch origin data/matura-historia && git worktree add /tmp/mh origin/data/matura-historia
export DATA=/tmp/mh/data/matura-historia/data.json
export HF_HOME=/team/hf-cache                # shared cache: the 24 GB download happens once
export HF_TOKEN=...                          # your own HF token with the Gemma licence accepted; never commit it
export TOKENIZERS_PARALLELISM=false
mkdir -p runs outputs
```

The scripts are PEP 723 (`uv run scripts/<x>.py` installs pinned torch/transformers/peft/bitsandbytes).
If the default CUDA wheel of torch doesn't match the driver, use a venv instead:
`uv venv -p 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements-test.txt --torch-backend=auto`,
then run `.venv/bin/python scripts/<x>.py` with the same arguments.

Run everything long inside tmux (`tmux new -s matura`; detach `Ctrl-b d`; re-attach `tmux a -t matura`)
or with `nohup ... > log 2>&1 &`.

## 2. Size check first + fallback submission (plain 4-bit base)

```bash
uv run scripts/export_submission.py --estimate            # config only: predicted nf4 size (seconds)
uv run scripts/export_submission.py --output outputs/base-4bit --verify --data $DATA
```

Rough expectation: ~11B params in nf4 (~0.52 bytes each) + the bf16 262k-vocab embedding (~2 GB) ≈ 7–7.8 GB.
Tight but under 8 GB. The script prints the real folder size and **fails if > 8 GB** (decimal GB).
`outputs/base-4bit` is a valid submission already, in case training doesn't help.

## 3. Baseline: untouched model, 4-bit, test split

```bash
uv run scripts/eval_matura.py --model google/gemma-4-12B-it --load-4bit --data $DATA \
    --split test --label base-4bit-test 2>&1 | tee runs/base-4bit-test.log
# quick dev-only look at closed items (~2 min):
uv run scripts/eval_matura.py --model google/gemma-4-12B-it --load-4bit --data $DATA \
    --split dev --types closed_abcd,true_false,matching --label base-4bit-dev-closed
```

Open items + essays are *ungraded* until you run a judge on the saved generations (no regeneration):

```bash
# API judge (team key, never commit):
MATURA_JUDGE_BASE_URL=... MATURA_JUDGE_API_KEY=... MATURA_JUDGE_MODEL=... \
  uv run scripts/eval_matura.py --output runs/eval/base-4bit-test --data $DATA --split test --regrade --judge openai
# or local judge on the GPU (base model, 4-bit):
uv run scripts/eval_matura.py --output runs/eval/base-4bit-test --data $DATA --split test --regrade \
    --judge hf --judge-model google/gemma-4-12B-it --judge-4bit
```

## 4. QLoRA training

```bash
mkdir -p runs/train
nohup uv run scripts/train_lora.py --data $DATA --output runs/train/qlora-r16 \
    --epochs 2 --lr 1e-4 --lora-r 16 --lora-alpha 32 --grad-accum 8 --max-len 4096 \
    --image-max-soft-tokens 560 --max-seconds 4200 > runs/train/qlora-r16.log 2>&1 &
tail -f runs/train/qlora-r16.log
```

* Base in nf4 4-bit + bf16 compute, LoRA on the language-model `q,k,v,o,gate,up,down` projections,
  assistant-only loss, gradient checkpointing, micro-batch 1 × grad-accum 8.
* Targets: closed items → short reasoning + `Odpowiedź: ...` line; open items → the CKE example answer; essays skipped.
* After each epoch: dev closed items generated + graded with the benchmark parser, and dev loss on all dev targets;
  `best_adapter/` = best dev (closed points, then dev loss). `report.json` has the full history.
  Dev has only 6 auto-gradable items (8 pts) → treat the dev closed score as noisy; the dev loss is the tie-break.
* The test split is never loaded by training.
* `--max-seconds` is a hard time budget (the epoch ends early and is still evaluated). OOM → `--image-max-soft-tokens 280`
  or `--max-len 3072`; `--text-only` trains without images.

## 5. Evaluate the adapter (4-bit, same as the submission)

```bash
uv run scripts/eval_matura.py --model google/gemma-4-12B-it --adapter runs/train/qlora-r16/best_adapter \
    --load-4bit --data $DATA --split test --label lora-r16-4bit-test
python scripts/compare_runs.py runs/eval/base-4bit-test runs/eval/lora-r16-4bit-test --items --md runs/compare.md
```

(run the same `--regrade --judge ...` on both runs before comparing total points.)

## 6. Export the submission (≤ 8 GB)

```bash
uv run scripts/export_submission.py --adapter runs/train/qlora-r16/best_adapter \
    --output outputs/submission-4bit --verify --data $DATA
# sanity: the exported folder should score like base+adapter on dev closed items
uv run scripts/eval_matura.py --model outputs/submission-4bit --data $DATA --split dev \
    --types closed_abcd,true_false,matching --label submission-dev-closed
```

`--merge-mode dequant` (default) merges into the same nf4 weights used during training and re-quantizes;
`--merge-mode bf16 --tmp-dir /team/tmp` merges in bf16 (~24 GB temp disk) and quantizes once.
`--adapter-only` saves only the adapter. `outputs/submission-4bit/submission_info.json` records size + config.

Keep the artifacts outside the ephemeral box (e.g. `cp -r outputs/submission-4bit runs /team/<you>/`) –
**never commit weights or tokens to git**.

## 7. Stop the box (always)

```bash
exit
fh session stop <id>
```
