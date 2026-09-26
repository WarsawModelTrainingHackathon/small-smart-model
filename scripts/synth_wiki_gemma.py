# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.0",
#   "torchvision==0.29.0",
#   "transformers==5.17.0",
#   "accelerate==1.15.0",
#   "bitsandbytes==0.50.2; sys_platform == 'linux'",
#   "pillow==12.3.0",
# ]
# ///
"""Generate matura-style train pairs from Wikipedia chunks using local Gemma (offline).

Writes JSONL {item, target} for scripts/train_lora.py --extra-jsonl.
Does not call ChatGPT. Exam-time answering model stays Gemma ≤ 8 GB nf4.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_matura as em  # noqa: E402
import wiki_bm25  # noqa: E402

SYS = (
    'Jesteś nauczycielem historii. Na podstawie TYLKO podanego fragmentu Wikipedii '
    'napisz jedno krótkie zadanie maturalne po polsku (open_short) i wzorcową odpowiedź. '
    'Odpowiedź: 3–8 zdań z konkretnymi faktami, datami i nazwami. '
    'Jeśli to pytanie zamknięte, ostatnia linia musi być dokładnie: Odpowiedź: A  (lub B/C/D). '
    'Nie wymyślaj faktów spoza fragmentu. Zwróć wyłącznie JSON: '
    '{"question":"...","context":"...","target":"...","type":"open_short"}'
)


def parse_json(text):
    text = text.strip()
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default=str(wiki_bm25.DEFAULT_CORPUS))
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='google/gemma-4-12B-it')
    p.add_argument('--n', type=int, default=180)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--load-4bit', action='store_true')
    a = p.parse_args()
    docs = []
    for line in Path(a.corpus).read_text(encoding='utf-8').splitlines():
        if line.strip():
            d = json.loads(line)
            if not wiki_bm25.is_junk_chunk(d.get('text') or '') and len(d.get('text') or '') > 200:
                docs.append(d)
    rng = random.Random(a.seed)
    rng.shuffle(docs)
    docs = docs[: max(a.n * 3, a.n)]
    model, processor = em.load_model(a.model, load_4bit=a.load_4bit)
    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, d in enumerate(docs):
            if n_ok >= a.n:
                break
            user = f"Tytuł: {d['title']}\n\nFragment:\n{d['text'][:1200]}"
            messages = [
                {'role': 'system', 'content': [{'type': 'text', 'text': SYS}]},
                {'role': 'user', 'content': [{'type': 'text', 'text': user}]},
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
            tok = em.tokenizer_of(processor)
            enc = tok(prompt, return_tensors='pt')
            enc = {k: v.to(model.device) for k, v in enc.items()}
            import torch
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=400, do_sample=False)
            text = tok.decode(gen[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
            obj = parse_json(text)
            if not obj or not obj.get('question') or not obj.get('target'):
                print(f'  skip {i} {d["title"]}', flush=True)
                continue
            typ = obj.get('type') if obj.get('type') in ('open_short', 'closed_abcd', 'true_false') else 'open_short'
            item = {
                'id': f'wiki-synth-{n_ok:04d}',
                'type': typ,
                'session_key': 'wiki-synth',
                'split': 'train',
                'task': f'synth {n_ok}',
                'question': obj['question'],
                'context': (obj.get('context') or f"Źródło: Wikipedia — {d['title']} (CC BY-SA)\n{d['text'][:900]}"),
                'max_points': 1,
                'images': [],
                'needs_visual': False,
                'auto_gradable': False,
                'wiki_title': d['title'],
                'wiki_id': d['id'],
            }
            rec = {'item': item, 'target': obj['target'].strip()}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            n_ok += 1
            print(f'  {n_ok}/{a.n} {d["title"]}', flush=True)
    print(f'wrote {n_ok} -> {out_path}')


if __name__ == '__main__':
    main()
