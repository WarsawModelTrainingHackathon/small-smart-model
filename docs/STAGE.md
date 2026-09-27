# On stage: running the exam (team Żabka)

The organisers' exam script and answer format are announced on Telegram. **Follow their instructions for sending answers.** This page covers only how to start our model and get answers quickly.

## What we run

| Category | Model on disk | Harness | Our test score |
|---|---|---|---|
| Best score / best progress | `/team/runs/submission-sd2-4bit` (self-distilled Gemma 4 12B, nf4, 7.78 GB) | + Wikipedia RAG `/team/wiki/plwiki-20231101-v2`, k=5, 3000 chars | 84.2% |
| Base model (for "progress") | `google/gemma-4-12B-it` loaded 4-bit (nf4 export: `/team/runs/base-nf4`, 7.71 GB) | plain, no RAG | 76.1% |
| Mały, ale wariat | `/team/runs/submission-qwen3vl4B-4bit` (Qwen3-VL-4B, 2.89 GB) | + the same Wikipedia RAG | 42.8% |

Everything lives on the shared `/team` disk, so any new Forgehand box in workspace `matura` sees it.

## 1. Start a box (~5 min before your slot)

```bash
fh session start matura --class gpu-l40s-small          # do not use --wait; poll instead:
fh session ls matura                                     # wait until "running", note the IP
ssh root@<IP>                                            # if the host key changed: ssh -o StrictHostKeyChecking=accept-new ...
```

## 2. Environment on the box

```bash
cd /team/rag-wt          # code with RAG (branch harness/wiki-rag); /team/small-wt adds the Qwen fix
export HF_HOME=/team/hf-cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false   # offline: nothing is downloaded
```

## 3. Answer a paper

Convert the exam questions into our `data.json` format: `{"test": [items]}` with `id, type, context, question, max_points, images` (see `sources/DATASET.md`). Then generate without the judge; the organisers grade:

```bash
uv run scripts/eval_matura.py --model /team/runs/submission-sd2-4bit --load-4bit \
  --data exam.json --split test --judge none --batch-size 16 \
  --rag-index /team/wiki/plwiki-20231101-v2 --rag-k 5 --rag-max-chars 3000 \
  --label final-exam --output /team/runs/eval/final-exam
# answers: /team/runs/eval/final-exam/generations.jsonl (field "output"; closed items end with "Odpowiedź: ...")
```

Measured on one L40S: 81 items (two full papers) take about 7 minutes including model load and retrieval, so **one paper takes about 3–4 minutes**. Model load is about 30 s and retrieval about 50 ms per question.

- **Base model run:** use `--model google/gemma-4-12B-it --load-4bit` without the `--rag-*` flags.
- **Small model:** run from `/team/small-wt` with `--model /team/runs/submission-qwen3vl4B-4bit`.

## 4. Afterwards

`fh session stop <id>`. Always stop the box, because billing is per second.

## If something breaks

- **SSH resets on a box after hours of heavy work:** this happened twice. Stop it and start a new box; `/team` keeps everything.
- **Out of time:** drop `--rag-*` (the model without Wikipedia still gets 81.4%) or pass `--essays one`.
