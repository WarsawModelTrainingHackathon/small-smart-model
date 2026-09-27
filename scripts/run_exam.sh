#!/bin/bash
# Winning Sunday run: Gemma 4 12B-it 4-bit nf4 + plwiki RAG, no wiki on essays.
# No LoRA. Uploads answers.json only (not weights). TEAM_KEY stays out of the JSON.
set -euo pipefail
EXAM_DIR="${1:?unpack the organizer zip and pass that folder (contains exam.json + images/)}"
OUT="${2:-answers.json}"

export HOME="${HOME:-/team/angel/home}"
export HF_HOME="${HF_HOME:-/team/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/team/angel/uv-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/team/angel/uv-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HOME" "$HF_HOME" "$UV_CACHE_DIR"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="/team/rag-wt/scripts:${PYTHONPATH:-}"

uv run --with tantivy==0.26.2 scripts/submit_exam.py \
  --exam-dir "$EXAM_DIR" \
  --model google/gemma-4-12B-it \
  --load-4bit \
  --rag-index /team/wiki/plwiki-20231101-v2 \
  --rag-k 5 \
  --rag-max-chars 3000 \
  --rag-skip-types essay \
  --rag-scripts /team/rag-wt/scripts \
  --output "$OUT"

echo "Upload $OUT as solution. For base/progress, rerun without --rag-index."
