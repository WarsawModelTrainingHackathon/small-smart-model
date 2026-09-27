---
name: train-llm-from-scratch
description: Navigate and evaluate the stared/train-llm-from-scratch workshop (ScratchGPT pretraining, Prawko SFT, RLVR, viewer). Use when running Modal/uv workshop commands, verifying runs, viewing reports, or scoring Qwen driving-exam / six-word / from-scratch models. Always follow the ordered eval command sequence in this skill.
---

# train-llm-from-scratch

Upstream: [github.com/stared/train-llm-from-scratch](https://github.com/stared/train-llm-from-scratch). Local clone: `train-llm-from-scratch/` (or this folder if the workspace *is* the clone).

**All commands below run from the repository root** (the directory that contains `README.md`). `pnpm dev` only starts the viewer; it never trains.

Official cloud path is **Modal**, not Forgehand. Do not mix volumes. Viewing results must not launch training.

Python **3.14** via uv. Node via **pnpm**. Modal CLI: `uv tool install modal==1.5.5` then `modal setup` — clone file `train-llm-from-scratch/modal-setup.md` (repo-root `modal-setup.md` if the workspace is the clone).

## Feature map

| Want | Open / run |
|---|---|
| Workshop path | `README.md` → numbered chapters |
| Tokens / BPE | `01-data-and-tokens.md` · viewer **Tokenization** |
| Pretrain ScratchGPT | `02-pretraining.md` · `scripts/scratch_recipe_modal.py` · viewer **Pretraining** |
| Driving exam SFT | `03-fine-tuning.md` · `scripts/prawko_modal.py` / `prawko.py` · viewer **SFT** |
| Six-word RLVR | `04-reinforcement-learning.md` · `scripts/rlvr_showcase_modal.py` · viewer **RLVR** |
| Model IDs | `scripts/models.json` |
| Included exam JSON | `datasets/prawko-v2/data.json` (100 train / 25 dev / 40 test) |
| Your metrics & samples | gitignored `runs/<name>/` |
| Weights (scratch) | Modal volume `model-training-workshop` → `/persist/runs/<name>/` (`best.pt`) |
| Adapter (SFT/RLVR) | same volume `adapter/` (+ original Qwen) |
| Saved experiment writeups | `results/` · viewer **Reports** (`http://localhost:5173/reports`) |
| Extra experiments | `additional/` · `LAB_NOTEBOOK.md` |
| UI conventions | `DESIGN.md` |

Scripts live in `scripts/`. `*_modal.py` = cloud launcher. Matching `prawko.py` / `train_scratch.py` / `rlvr_showcase.py` = actual loops.

## Ordered evaluation (do not skip)

Copy this checklist. **A before B before C.** Training jobs already evaluate before and after; the point of this order is: trust the code, then baseline, then train, then audit files, then look at the UI.

### A. CPU / no GPU — prove the repo, not the GPU model

```bash
uv run tests/run.py
pnpm exec node --test tests/test_prediction.mjs
uv run scripts/check_scratch_model.py
```

- `tests/run.py`: splits, Q/A formatting, tokenizers, deterministic RLVR checkers. No model download.
- `test_prediction.mjs`: browser probability / token rendering.
- `check_scratch_model.py`: pinned CPU checks of ScratchGPT internals.

CI is the same first two commands (`.github/workflows/cpu-tests.yml`). If A fails, do not spend GPU.

### B. Viewer (keep running in a second terminal)

```bash
pnpm install   # first clone only
pnpm dev
```

Open [http://localhost:5173](http://localhost:5173). Included **Example run** rows are historical, not your model. Select **your** job in **Run** after a training/eval job prints `Saved`.

### C. Choose one evaluation target

Pick **one** of C1–C3. Do not treat pretraining, SFT, and six-word RLVR as the same metric.

#### C1. Polish driving exam (Qwen + A/B/C) — primary “evaluate the model” path

Splits: train updates weights; **dev** selects checkpoint; **test** (40 questions) is the score. Report **correct/40**, not a homemade metric.

**1. Baseline only** (`screen` = no adapter training; still loads Qwen on GPU):

```bash
modal run scripts/prawko_modal.py --method screen --model qwen3.5-0.8b
```

Local Mac:

```bash
uv run scripts/prawko.py --device mps --precision bfloat16 --method screen --model qwen3.5-0.8b
```

Wait for `Saved runs/prawko-screen-<id>/`. Read `before_test` metrics in the terminal / `execution.json` / `result.json`.

**2. Train + evaluate** (script order is fixed: evaluate original → train LoRA → select by **dev** → evaluate selected on test):

```bash
modal run scripts/prawko_modal.py --method sft --epochs 10 --max-seconds 180
```

Local:

```bash
uv run scripts/prawko.py --device mps --precision bfloat16 --method sft --epochs 10 --max-seconds 180
```

Optional RLVR on the **same exam** (not the six-word task):

```bash
modal run scripts/prawko_modal.py --method rlvr --epochs 10 --max-seconds 180
```

**3. Audit saved records** (no retrain, no GPU):

```bash
uv run scripts/verify_prawko.py runs/prawko-sft-*
```

Must pass: recomputed accuracy, split isolation, checkpoint selection, reload of first 8 test predictions.

**4. Viewer:** **SFT** → your run → **Test answers** (before → after on 40 test items). Dev examples under the graph are **not** the test set.

**5. New question (inference, bills GPU, does not train).** Copy the folder name from `Saved` **without** `runs/`:

```bash
modal run scripts/try_adapter_modal.py --task exam --run YOUR_SFT_RUN_NAME \
  --question "…" --a "…" --b "…" --c "…"
```

#### C2. ScratchGPT pretraining (loss + samples, not exam accuracy)

**1. Data (once per Modal volume):**

```bash
modal run scripts/prepare_data_modal.py
```

Wait until the job says **Ready**.

**2. Train** (evaluates random init, trains, selects best **dev loss**, evaluates selected):

```bash
modal run scripts/scratch_recipe_modal.py --recipe wolne-lektury
```

**3. Audit reports** (no weight load unless you pass `--require-weights`):

```bash
uv run scripts/verify_pretraining.py runs/scratch-*
```

**4. Viewer:** **Pretraining** — loss curve and checkpoint continuations.

**5. Generate from weights** only if `best.pt`, `result.json`, and `tokenizer.json` are **together locally** (Modal does not download weights by default):

```bash
uv run scripts/sample_scratch.py runs/YOUR_SCRATCH_RUN --output scratch_samples.json --device cpu
```

Viewer **Next-token prediction** needs those three files in the run folder.

#### C3. Six-word RLVR (Qwen3.5-4B; independent of C1)

**1. Baseline screen:**

```bash
modal run scripts/rlvr_showcase_modal.py --task six_words --stage screen
```

**2. Train + evaluate:**

```bash
modal run scripts/rlvr_showcase_modal.py --task six_words
```

Local Mac:

```bash
uv run scripts/rlvr_showcase.py --device mps --precision bfloat16 --task six_words
```

**3. Viewer:** **RLVR** — sampled answers, rewards, held-out success. Metric is checker **success** (e.g. 1/32 → 24/32), not Prawko letters.

## What “correct evaluation” means here

- **Prawko:** constrained A/B/C from next-token logits on letters; option shuffle in **training** only; `test_rotated` exists in run JSON. Do not grade free-form text.
- **Scratch:** next-token **loss** (nats) on held-out windows + optional generation quality files. Literary continuation ≠ Q&A.
- **Six-word RLVR:** Python verifier on generated text (`scripts/rlvr_tasks.py`).
- Compare **before** vs **after** on the **same** split the script used (`test` for reported exam score). Dev is for selection.
- Example numbers in the markdown guides are historical, not pass/fail gates.

## Modal vs local

| | Modal | Local |
|---|---|---|
| Setup | `modal token info` after `modal setup` | `--device mps` (Mac) or `cuda` |
| Persistence | volume `model-training-workshop` | `runs/` on disk |
| Cost | GPU + storage | free compute; HF download once |

Keep the training/eval terminal connected until `Saved`.

## Agent defaults

- `cd` to the clone root before any command.
- Prefer `--method screen` before claiming a baseline.
- After every GPU job, run the matching `verify_*.py` when a run folder exists.
- Do not commit `runs/` weights or HF caches.
- Optional extras (Wikipedia recipes, poetry): `additional/` — not required for the four README exercises.

More flags and run-folder files: [reference.md](reference.md).
