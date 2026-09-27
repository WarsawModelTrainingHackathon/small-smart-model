---
name: warsaw-matura-hackathon
description: Warsaw Model Trainers Hackathon (Kolektyw3 × AI Tinkerers) for team z-abka. Use when working on the Polish matura LLM task, Bielik/PLLuM, harness vs fine-tuning tracks, experiment logging (runs.csv), Forgehand GPU (workspace matura), retrieval over Polish Wikipedia, calculator/voting/self-check, eval dashboard, or Sunday demo/submission.
---

# Warsaw Model Trainers Hackathon (z-abka)

Vlad is a TypeScript/Node.js full-stack engineer (React, NestJS, PostgreSQL, LLM tooling, CI/CD). Python is not his main language — write and debug Python for him; keep pipeline glue, dashboard, calculator tool, repo, and demo in the stack he owns unless the rest of the team already landed it in Python.

## Event

- **Where:** Kolektyw3, Koszykowa 54, Warsaw. Organisers Kolektyw3 × AI Tinkerers; partners JetBrains and Quesma.
- **When:** Fri 25 Sep 2026 18:00 → Sun 27 Sep 2026 16:00 (46 h). Demos and scoring Sunday 16:00.
- **Task:** Can a small open Polish model (Bielik, PLLuM, or another from the allowed list) pass the Polish matura? Questions are generated from Polish Wikipedia, graded automatically, scored in matura points. One leaderboard, prizes per track.

### Tracks

- **Harness** — no training. Build around the model: retrieval over Polish Wikipedia, calculator tool, multiple attempts + voting, second-pass check. Scored on matura points.
- **Fine-tuning** — train a small model (a few GB) on our own data. Two size classes, GPU credits included. Scored on matura points + gain over the untouched base model.

Ask organizers before assuming a team may combine tracks or submit to both.

## Team (z-abka)

| Who | Area |
|---|---|
| Marcin | Polish Wikipedia: load, chunk, index, retrieval |
| Daniel | Model serving, prompts, voting, self-check; fine-tuning |
| Angel | Evaluation + error analysis (failure categories) |
| Vlad | Pipeline integration, calculator tool, experiment log + dashboard, repo, submission, demo |

Stay in lane unless asked. Integrate others' pieces; do not rewrite retrieval or training without cause.

## Working rules

- Baseline first: score the bare model (no retrieval) before adding tools.
- One change → one eval run → log it. Keep only what raises the score.
- Main branch always works.
- Watch runtime (leaderboard / per-question limits may apply — confirm with organizers).
- Sleep.

### Schedule (intent)

- Friday evening: baseline, then sleep.
- Saturday 10–22: BM25 retrieval → prompts → calculator → voting → self-check → embeddings/hybrid if needed.
- Sunday: feature freeze **12:00**, final run, README, demo (pipeline diagram + score gain per step).

## Experiment logging (no DB)

Always record git commit + config. Log retrieved passages so retrieval failures can be split from model failures.

**`runs.csv` columns:** `run_id`, `timestamp`, `git_commit`, `description`, config (`model`, `top_k`, `n_votes`, `prompt_version`), `score`, `accuracy`, `avg_latency`, `total_runtime`.

**`runs/<run_id>.jsonl` per question:** `question_id`, `category`, `retrieved_passage_ids`, `prompt_version`, `raw_output`, `final_answer`, `correct`, `points`, `latency`.

Do not introduce a database for this. Dashboard reads these files.

Schemas and example rows: [reference.md](reference.md).

## Forgehand (GPU; not Modal)

Team **z-abka**, workspace **`matura`**, class used: **`gpu-l40s-small`** (L40S, 46 GB). SSH is `root@<ip>` (IP changes each session).

| Path | Persistence |
|---|---|
| `/workspace` | Shared by all sessions of this workspace (code, checkpoints). `~` = `/workspace/.home`. HF cache here. |
| `/team` | Shared across the whole team (datasets, shared checkpoints). |
| `/scratch` | Fast NVMe, **deleted on stop**. Training data, temp outputs. venv: `/scratch/.venv`. |

Packages: `uv pip install` is preinstalled. Installed Python packages go to `/workspace/.forgehand/requirements.txt` and reinstall on next start. Apt packages are recorded too. `/workspace/setup.sh` runs at every start.

Secrets (e.g. `HF_TOKEN`) live in Settings and inject as env vars.

Expose ports: bind `0.0.0.0`, port **1024–65535 except 8888**, then `fh port expose` (`team` or `public`). Use for vLLM and the demo.

Billing is per second while a session exists. Idle stop after 4 h with no processes/SSH/Jupyter. Long jobs in **tmux**. Always `fh session stop <id>` when done.

Several teammates can run sessions on the same workspace — **agree on separate output dirs**.

Forgehand also has a team LLM API (GPT-6 Sol/Luna, OpenAI-compatible). **Do not use it as the answering model** unless organizers allow it. Likely OK only for data generation / local dev help — ask.

CLI cheat sheet and paths: [reference.md](reference.md). For general Forgehand mechanics, use the existing `forgehand-sessions`, `forgehand-workspaces`, `forgehand-connect`, `forgehand-account`, `forgehand-llm`, `forgehand-secrets` skills.

## Fine-tuning (Daniel's lane; support only)

Minimal path: Hugging Face transformers + TRL `SFTTrainer` + LoRA (peft), bf16, JSONL chat messages (question → answer).

Continue training: `trainer.train(resume_from_checkpoint=...)`, or `PeftModel.from_pretrained(base, path, is_trainable=True)` on the **same** base model.

Workshop template (written for Modal; adapt to Forgehand): [stared/train-llm-from-scratch](https://github.com/stared/train-llm-from-scratch) — step 3 fine-tunes Qwen3.5-0.8B on the Polish driving exam. Local clone may exist as `train-llm-from-scratch/` in this workspace.

Do not commit model weights.

## Licensing

- Our code: Apache 2.0 or MIT.
- Model: keep Bielik/PLLuM original license (HF model card).
- Wikipedia data: CC BY-SA 4.0.
- Do not commit weights to git.

## Organizer questions (do not invent answers)

1. Model access: provided endpoint or we run it? Allowed models including embeddings/rerankers?
2. Is Polish Wikipedia provided as a dataset? Sample matura questions for dev?
3. Submission format, time limits per question, leaderboard runs?
4. Can a team combine fine-tuning + harness / submit to both tracks?
5. Is the Forgehand LLM API (GPT-6) allowed anywhere in the solution?

## Side notes

- **quesma-shipper** is installed and **paused**. Do not resume until old NDA projects are excluded from collection. Afterward: `brew uninstall --cask --zap quesmaorg/tap/quesma-shipper`.
- CV target for Vlad: a concrete bullet such as "built the end-to-end pipeline, tool calling and evaluation dashboard; improved matura score from X to Y". Preserve run logs so X and Y are real numbers.

## Agent defaults

- Prefer measuring over theorizing. If a change is not logged with a score, it did not happen.
- Do not resume paused Quesma collection.
- Stop Forgehand sessions when work is done; do not leave GPUs idle.
- Demo artifacts: pipeline diagram + per-step score gain from `runs.csv`.
