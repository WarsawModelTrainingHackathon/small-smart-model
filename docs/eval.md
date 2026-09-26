# Matura historia – evaluation harness

Files:

| file | what |
|---|---|
| `scripts/matura_format.py` | **the** shared prompt / target / answer-parsing module (eval **and** training import it) |
| `scripts/matura_grading.py` | closed-item auto grading with CKE partial credit, LLM judge for open items + essays, report aggregation |
| `scripts/eval_matura.py` | CLI: generate → grade → `report.json` + `report.md` (PEP 723, `uv run`) |
| `scripts/compare_runs.py` | side-by-side table of several runs, per-item diffs |
| `scripts/wiki_bm25.py` + `fetch_wiki_minicorpus.py` | optional BM25 over a small Polish Wikipedia slice (harness, not training) |
| `tests/` | CPU tests: parsing/grading on real CKE items + end-to-end smoke on a tiny random Gemma 4 |

## Protocol (what the numbers mean)

* Prompt: Polish system prompt; user turn = images (or `adapted_660_text` with `--text-only`) + context + question
  (the `question` field already contains the options/statements, they are not appended again) + a type-specific instruction.
* Closed items (`closed_abcd`, `true_false`, `matching`) must end with a final line
  `Odpowiedź: C`, `Odpowiedź: 1-F; 2-P; 3-P` or `Odpowiedź: A-Karol IX; B-Henryk IV`.
  The parser reads **only the last** `Odpowiedź:` line (after stripping any thinking block); letters mid-reasoning never count.
  Matching values are compared case/whitespace/punctuation/diacritics-insensitively, with ` / ` alternatives and `[optional]` parts.
* Partial credit for multi-part closed items is read from the CKE scoring text (e.g. 2 pkt for 3/3, 1 pkt for 2/3).
* Closed items that also require a justification (`uzasadnij`): 0 if the choice is wrong, otherwise the judge scores it.
* Open items and essays: LLM judge with the CKE rubric (`scoring` / `scoring_raw`) + example answer → `{"points", "reason"}`.
  Without a judge (`--judge none`) they are reported as *ungraded* and the % is over graded items only.
* Essay topics of one session are alternatives: exactly **one essay per session** is counted
  (grouped by `choice_group` if present, else `session_key`+`group`). By default all topics are generated and the
  session's essay score is the **mean** over topics (`--essay-policy best|first` also available; `--essays one` generates only the first topic).
  Paper maximum: 60 per formula-2023 session → **test = 120, dev = 60**.
* Greedy decoding. `--load-4bit` = bitsandbytes nf4 + double quant + bf16 compute, i.e. exactly the submitted artifact
  (a checkpoint produced by `scripts/export_submission.py` is detected as pre-quantized automatically).

Report: total points/max and %, closed-only %, unparseable closed rate, per type, per session (with paper max), visual vs text.

## Data

The dataset is on branch `data/matura-historia` – do not commit it to other branches:

```bash
git fetch origin data/matura-historia
git worktree add /tmp/mh origin/data/matura-historia
DATA=/tmp/mh/data/matura-historia/data.json
```

(`--data` defaults to `data/matura-historia/data.json`; image paths are relative to the folder of `data.json`.
Branch `data/matura-historia-archiwum` with `choice_group` works the same way.)

## Commands on the Forgehand GPU box

See `docs/forgehand-runbook.md` for starting/stopping the box (training branch). Once on the box:

```bash
git clone https://github.com/WarsawModelTrainingHackathon/small-smart-model && cd small-smart-model
git checkout eval/benchmark
git worktree add /tmp/mh origin/data/matura-historia
export DATA=/tmp/mh/data/matura-historia/data.json
export HF_HOME=/team/hf-cache          # shared cache: the 12B download happens once per team
export HF_TOKEN=...                    # your own token with Gemma access; never commit it

# 1) quick sanity (closed items only, no judge, ~2 min)
uv run scripts/eval_matura.py --model google/gemma-4-12B-it --load-4bit --data $DATA \
    --split dev --types closed_abcd,true_false,matching --label base-4bit-dev-closed

# 2) baseline of the untouched model, 4-bit, full test split (~20-30 min on an L40S)
uv run scripts/eval_matura.py --model google/gemma-4-12B-it --load-4bit --data $DATA \
    --split test --label base-4bit-test

# 3) grade the open items + essays with a judge (re-uses saved generations, no GPU needed)
export MATURA_JUDGE_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible endpoint
export MATURA_JUDGE_API_KEY=...                          # from the team, never commit
export MATURA_JUDGE_MODEL=gpt-4.1-mini
uv run scripts/eval_matura.py --output runs/eval/base-4bit-test --data $DATA --split test --regrade --judge openai
#   or a local judge on the GPU: --judge hf --judge-model google/gemma-4-12B-it --judge-4bit

# 4) adapter / exported submission (same flags otherwise)
uv run scripts/eval_matura.py --model google/gemma-4-12B-it --adapter runs/train/<run>/best_adapter --load-4bit \
    --data $DATA --split test --label lora-4bit-test
uv run scripts/eval_matura.py --model outputs/submission-4bit --data $DATA --split test --label submission-test

# 5) compare
python scripts/compare_runs.py runs/eval/base-4bit-test runs/eval/lora-4bit-test --items --md runs/compare.md
```

Wiki BM25 (CPU corpus; still needs GPU for the model). Rebuild: `python3 scripts/fetch_wiki_minicorpus.py`. Then:

```bash
uv run scripts/eval_matura.py --model google/gemma-4-12B-it --load-4bit --data $DATA \
    --split dev --wiki --label base-4bit-dev-wiki --ids \
    2024-maj-R-zad3.1,2024-maj-R-zad7,2024-maj-R-zad12.1,2024-maj-R-zad19.2,2024-maj-R-zad5.1,2024-maj-R-zad24,2024-maj-R-zad25,2024-maj-R-zad1,2024-maj-R-zad11.2
```

Compare those ids to `base-dev` before a full-dev wiki run. `--wiki` with no path uses `harness/wiki/minicorpus.jsonl`.

Other flags: `--text-only` (no images, uses `adapted_660_text`), `--thinking on` (Gemma thinking mode, adds
`--thinking-budget` new tokens), `--image-max-soft-tokens 70|140|280|560|1120` (default 560), `--batch-size`,
`--limit`, `--types`, `--ids`. Runs are resumable: re-running the same command continues `generations.jsonl`.
Use the **dev** split for decisions; look at **test** only for the final numbers.

## Tests (CPU, no downloads)

```bash
uv venv -p 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements-test.txt
MATURA_DATA=/tmp/mh/data/matura-historia/data.json .venv/bin/python -m pytest
```

The smoke test builds a tiny random `Gemma4UnifiedForConditionalGeneration` + processor offline
(`tests/tiny_gemma.py`: real transformers classes, tiny sizes, a small BPE tokenizer and a Gemma-4-style chat template).
Full-dataset checks are skipped when the dataset is not available.
