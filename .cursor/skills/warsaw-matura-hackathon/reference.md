# Reference — z-abka matura hackathon

## Forgehand CLI (team defaults)

Workspace: `matura` (team `z-abka`). Class: `gpu-l40s-small`.

```bash
fh workspaces
fh classes
fh session start matura --class gpu-l40s-small --wait
fh session ls
fh session ssh <id>
fh session jupyter <id>
fh port expose <id> <port>        # add --public if needed
fh session stop <id>
```

SSH: `root@<ip>` — IP changes every session.

Python venv: `/scratch/.venv`. Persist installs via `/workspace/.forgehand/requirements.txt`. Startup: `/workspace/setup.sh`.

Bind services to `0.0.0.0` on ports 1024–65535, **not 8888**.

## Experiment log formats

### `runs.csv`

Header:

```text
run_id,timestamp,git_commit,description,model,top_k,n_votes,prompt_version,score,accuracy,avg_latency,total_runtime
```

Example:

```text
20260926-1430-bm25-k5,2026-09-26T14:30:00+02:00,abc1234,BM25 top_k=5 vs baseline,bielik-...,5,1,v2,41.2,0.38,1.8,620
```

- `score`: matura points (leaderboard metric).
- `accuracy`: fraction of questions graded correct, if available.
- `avg_latency` / `total_runtime`: seconds.
- Empty retrieval: `top_k=0` or blank; `n_votes=1` for single-pass.

### `runs/<run_id>.jsonl`

One JSON object per line:

```json
{
  "question_id": "q-001",
  "category": "matematyka",
  "retrieved_passage_ids": ["wiki-12#c3", "wiki-88#c0"],
  "prompt_version": "v2",
  "raw_output": "...",
  "final_answer": "42",
  "correct": true,
  "points": 2,
  "latency": 1.72
}
```

Baseline runs: `retrieved_passage_ids` is `[]`. Log them anyway so later runs are comparable.

## Fine-tuning snippets

SFT + LoRA (bf16), JSONL of chat messages. Continue:

```python
trainer.train(resume_from_checkpoint=checkpoint_dir)
```

or

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
```

Same base model as the adapter. Checkpoints: `/workspace` or `/team`, not only `/scratch`.

Workshop recipe: `https://github.com/stared/train-llm-from-scratch` (Modal-oriented; map volumes to Forgehand paths).

## Licensing reminder

Code Apache 2.0 or MIT. Model license from the HF card. Wikipedia CC BY-SA 4.0. No weights in git.
