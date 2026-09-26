# Mini Wikipedia (CC BY-SA 4.0)

Polish Wikipedia extracts for BM25 RAG on the matura harness. Not CKE exam text.

```bash
python3 scripts/fetch_wiki_minicorpus.py
uv run scripts/eval_matura.py --model google/gemma-4-12B-it --load-4bit --data $DATA \
  --split dev --wiki --label base-dev-wiki
```

`--wiki` injects top-3 chunks into the prompt (essays skipped unless `--wiki-essays`).
Rebuild the jsonl from the network; do not vendor a full Wikipedia dump.
