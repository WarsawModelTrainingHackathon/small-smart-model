# Workshop scripts and run artifacts

## Modal setup (once)

From clone root, after creating a Modal account and setting spend limit as in `modal-setup.md`:

```bash
uv tool install modal==1.5.5
modal setup
modal token info
```

`pnpm dev` does not submit jobs.

## Prawko `runs/<name>/` files

| File | Role |
|---|---|
| `execution.json` / `result.json` | before/after metrics, hashes |
| `data.json` | splits used |
| `before_test.json` / `after_test.json` | per-question A/B/C + probabilities |
| `history.json` / `checkpoints.json` | training / selection |
| `reload.json` | first 8 test items after reload |
| `local_verification.json` | written by `verify_prawko.py` |

`verify_prawko.py` recomputes accuracy from per-row `correct` flags; it does not reload Qwen.

## Scratch `runs/<name>/`

Laptop: reports and samples for the viewer. Weights: Modal `/persist/runs/<name>/best.pt` plus `tokenizer.json` and config in `result.json`.

```bash
uv run scripts/scratch_quality.py runs/YOUR_RUN --device cpu --checkpoint best.pt
```

Needs local weights. Optional; not the README default.

## Recipe names (`scratch_recipe_modal.py`)

`wolne-lektury` (workshop default), `wiki`, `wiki-cheap`, `wiki-100m`, `sejm`, `stories`, `stories-cheap`, `popular-wiki`.

Workshop `max_seconds` is 60–600 (default 600 for that launcher).

## RLVR showcase tasks

`--task six_words` (README). Also `countdown`, `maze`, `all`. `--stage screen` | `train`.

## Extra (not the main path)

- `additional/scripts/prawko_report.py` — markdown from saved prawko runs
- `additional/scripts/scratch_report.py` — pretraining reports
- `scripts/rlvr_showcase_report.py` — RLVR markdown from run dirs
- `additional/scripts/scratch_posttrain.py` — SFT/RLVR on a **scratch** checkpoint (`exam` / `poetry` / `wiki-qa`)

## Viewer routes

- App: `http://localhost:5173`
- Reports: `http://localhost:5173/reports` (committed HTML in `results/`)
