# small-smart-model: Gemma 4 12B passes the Polish history matura offline

**Team Żabka (z-abka)**, Warsaw Model Trainers hackathon, Kolektyw3, 25–27.09.2026 (see [SOURCE.md](SOURCE.md)).

**PL, w skrócie:** `google/gemma-4-12B-it` skwantyzowana do 4 bitów (7,7 GB na dysku, limit 8 GB), dostrojona przez **samodestylację** (QLoRA na własnych odpowiedziach modelu, które dostały pełne punkty według kluczy CKE) i uzupełniona o **lokalną polską Wikipedię** (RAG, offline). Na odłożonych maturach CKE z maja 2025 i 2026 wynik rośnie z **76,1%** (model bez zmian) do **81,4%** (sam trening) i do **84,2%** (trening + Wikipedia, +8,1 pkt proc.). Dev (maj 2024): 83,9%.

## Results

Internal held-out test: the full CKE *historia, poziom rozszerzony* papers from May 2025 and May 2026 (81 scored items, 2 × 60 points; one essay per paper is counted, as in the real exam). No item from these sessions, and no near-duplicate of their questions or sources, is in the training data.

| Model (all 4-bit nf4, images on, greedy) | Test score | + local Wikipedia RAG |
|---|---|---|
| `gemma-4-12B-it`, untouched (**base**) | 91.3 / 120 = **76.1%** | 98.3 / 120 = **81.9%** |
| First QLoRA on CKE example answers (rejected) | 69.0 / 120 = 57.5% | – |
| Self-distillation, round 1 (426 targets) | 94.3 / 120 = 78.6% | 99.3 / 120 = 82.8% |
| **Self-distillation, round 2 (1183 targets)**, submitted | 97.7 / 120 = **81.4%** | 101.0 / 120 = **84.2%** |

Submitted configuration (fine-tuned 4-bit model + Wikipedia RAG) on dev (May 2024): 50.3 / 60 = 83.9%. Speed on one L40S, batch 16: both test papers (81 items) generate in about 7 minutes including model load and retrieval, so about 3–4 minutes per paper.

Model sizes on disk: base 4-bit 7.71 GB; submitted merged 4-bit checkpoint 7.78 GB (limit 8 GB). The Wikipedia index does not count toward the limit.

Caveats. Open questions and essays are graded by an LLM judge (the same local Gemma, rubric = CKE *zasady oceniania* + example answer), not by human examiners, so absolute numbers are approximate. Every row uses the same judge and the same items, so the comparisons between rows are fair. For reference, the organisers' benchmark gives the untouched Gemma 4 12B 76.7% on the 2023 paper with images; we measured 76.1%.

## What we did and why

1. **Data** (`scripts/prepare_matura_historia*.py`). The scripts download official CKE papers and scoring rules (URLs and SHA-256 in `sources/cke-manifest.json` and `sources/cke-archive-manifest.json`) and parse them into ~1,800 scored tasks with keys, rubrics and figure crops. The sources are formula 2023 (2023–2026), formula 2015 (2015–2023) and the old formula (2005–2020, podstawowy and rozszerzony), plus informatory and próbne. Whole sessions are held out: dev = May 2024, test = May 2025 + May 2026. `scripts/check_matura_historia_leaks.py` removes near-duplicates of dev/test from train. See `sources/DATASET.md` and `sources/QC.md`: all 123 dev/test items were checked by hand against the PDFs. **No CKE PDFs or images are in this branch.** Run the scripts to fetch them.
2. **Evaluation** (`scripts/eval_matura.py`, `matura_grading.py`, `matura_format.py`). Closed items are graded automatically from the last `Odpowiedź:` line, with CKE partial credit. Open items and essays go to an LLM judge.
3. **The first fine-tune made things worse.** QLoRA on the short CKE example answers dropped the score from 76.1% to 57.5%. The essay score fell from 88% to 23%, because the model learned to answer too briefly.
4. **Self-distillation fixed it** (`scripts/build_self_distill.py`, `train_lora.py --targets-jsonl`). The base model answers every training task, the judge grades the answers against the CKE key, and only full-score answers (essays at ≥ 80%) become targets. QLoRA (r=16, lr 5e-5, 2 epochs, nf4 base) on those targets gave 78.6% in round 1 and 81.4% in round 2, after adding the archival exams. The model keeps its own style and gets better at the exam format.
5. **Local Wikipedia RAG** (`scripts/wiki_build_index.py`, `wiki_rag.py`, `eval_matura.py --rag-index`). The full Polish Wikipedia (`wikimedia/wikipedia 20231101.pl`, CC BY-SA) is split into 2.6 M passages and indexed with Tantivy BM25 plus crude Polish stemming. The top 5 passages (≤ 3,000 chars) go into the prompt. Median query time is ~4 ms, fully offline. See `docs/wiki-rag-retrieval.md`.
6. **Export** (`scripts/export_submission.py`). The script merges the LoRA into the 4-bit base, saves an nf4 checkpoint and asserts that it is ≤ 8 GB.

## Reproduce

Everything runs with `uv run scripts/<x>.py` (PEP 723 pinned deps). We used one NVIDIA L40S 48 GB (Forgehand).

```bash
# data (downloads the official CKE PDFs; nothing copyrighted is stored in git)
uv run scripts/prepare_matura_historia.py
uv run scripts/prepare_matura_historia_archiwum.py
DATA=data/matura-historia/data.json

# baseline, untouched model, 4-bit
uv run scripts/eval_matura.py --model google/gemma-4-12B-it --load-4bit --data $DATA --split test \
  --judge hf --judge-model google/gemma-4-12B-it --judge-4bit --label base-4bit-test

# self-distillation: answer + grade the train split, keep full-score answers, train, export
uv run scripts/eval_matura.py --model google/gemma-4-12B-it --load-4bit --data $DATA --split train \
  --judge hf --judge-model google/gemma-4-12B-it --judge-4bit --label base-train --output runs/eval/base-train
uv run scripts/build_self_distill.py runs/eval/base-train --output runs/sft/sd.jsonl
uv run scripts/train_lora.py --data $DATA --targets-jsonl runs/sft/sd.jsonl --output runs/train/sd \
  --epochs 2 --lr 5e-5 --lora-r 16 --lora-alpha 32 --grad-accum 8 --max-len 6144
uv run scripts/export_submission.py --adapter runs/train/sd/best_adapter --output outputs/submission-4bit --verify --data $DATA

# Wikipedia index (once, CPU) and a RAG evaluation
uv run scripts/wiki_build_index.py --help
uv run scripts/eval_matura.py --model outputs/submission-4bit --load-4bit --data $DATA --split test \
  --rag-index <wiki-index-dir> --rag-k 5 --rag-max-chars 3000 --judge hf --judge-model google/gemma-4-12B-it --judge-4bit
```

More detail: `docs/eval.md`, `docs/forgehand-runbook.md`, `docs/wiki-rag-retrieval.md`.

## Rules compliance

- **Model limit:** one model, ≤ 8 GB on disk (7.78 GB nf4). The LoRA is merged; the RAG index is excluded from the limit by the rules.
- **Offline exam:** no external API and no web search at exam time. The Wikipedia index is a local directory.
- **Closed-model APIs** were used only as coding assistants while building. They were not used to generate training answers: every training target comes from our own Gemma.
- **Data and licences:** CKE materials are fetched by script and not committed. Wikipedia is CC BY-SA. Model weights follow the Gemma licence.

## Team

Daniel, Marcin, Angel, Vlad.
