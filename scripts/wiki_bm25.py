"""Tiny Polish-Wikipedia BM25 index for the matura harness (no torch, no GPU).

Corpus: harness/wiki/minicorpus.jsonl  (rebuild: python3 scripts/fetch_wiki_minicorpus.py)
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / 'harness' / 'wiki' / 'minicorpus.jsonl'
_TOKEN = re.compile(r'[a-ząćęłńóśźż0-9]+', re.I)
K1, B = 1.5, 0.75
# Exam boilerplate — searching these only pulls random wiki.
_QUERY_STOP = {
    'zadanie', 'zadania', 'pkt', 'punkt', 'punkty', 'podaj', 'wyjaśnij', 'uzasadnij',
    'źródło', 'źródła', 'fragment', 'fragmentu', 'fragmenty', 'na', 'podstawie',
    'rycina', 'mapa', 'tabela', 'ilustracja', 'określ', 'wskaż', 'rozstrzygnij',
    'prawda', 'fałsz', 'odpowiedź', 'polecenie', 'materiał', 'materiały',
}
_JUNK = re.compile(
    r'(?i)^\s*(przypisy|bibliografia|zobacz też|linki zewnętrzne|twórczość)\b'
    r'|isbn\s*\d|issn\s*\d',
)


def tokenize(text):
    text = unicodedata.normalize('NFKC', text or '').lower()
    return _TOKEN.findall(text)


def is_junk_chunk(text):
    head = (text or '')[:400]
    if _JUNK.search(head):
        return True
    # bibliography-only: many years+ISBN, almost no verbs
    if head.count('ISBN') >= 1 or head.lower().count('bibliografia') >= 1:
        return True
    return False


def query_from_item(item):
    """Question + a few proper-name-ish tokens from CKE context. Not the whole source.

    Searching the full source dump matches rare bibliography names and hurts items
    the base model already gets right. This is paper-agnostic: Sunday's exam
    still has a question string.
    """
    q = item.get('question') or ''
    names = re.findall(r'\b[A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźż-]{3,}\b', item.get('context') or '')
    # keep unique, drop all-caps OCR noise longer than 12
    seen, keep = set(), []
    for n in names:
        if n.lower() in _QUERY_STOP or n in seen:
            continue
        seen.add(n)
        keep.append(n)
        if len(keep) >= 8:
            break
    return f'{q}\n{" ".join(keep)}'


def format_hits(hits, max_chars=1400):
    chunks, n = [], 0
    for h in hits:
        block = f"[{h['title']}] {h['text'].strip()}"
        if n + len(block) > max_chars and chunks:
            break
        chunks.append(block)
        n += len(block) + 2
    return '\n\n'.join(chunks)


class WikiIndex:
    def __init__(self, docs):
        """docs: [{id, title, text, url}]"""
        self.docs = docs
        self.N = max(1, len(docs))
        self.tf = []
        self.dl = []
        df = Counter()
        for d in docs:
            toks = tokenize(d['text'])
            bag = Counter(toks)
            self.tf.append(bag)
            self.dl.append(len(toks) or 1)
            df.update(bag.keys())
        self.avgdl = sum(self.dl) / self.N
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}

    def search(self, query, k=2):
        q = [t for t in tokenize(query) if t not in _QUERY_STOP and len(t) > 2]
        if not q:
            return []
        scores = []
        for i, bag in enumerate(self.tf):
            s = 0.0
            dl = self.dl[i]
            overlap = 0
            for t in q:
                idf = self.idf.get(t)
                if not idf:
                    continue
                f = bag.get(t, 0)
                if f:
                    overlap += 1
                    s += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / self.avgdl))
            if s > 0:
                scores.append((s, overlap, i))
        scores.sort(reverse=True)
        if not scores:
            return []
        # Weak match: do not inject. Better to stay silent than distract the base model.
        top_s, top_ov, _ = scores[0]
        if top_ov < 2 or top_s < 4.0:
            return []
        best_title = self.docs[scores[0][2]]['title']
        out = []
        for s, ov, i in scores:
            d = self.docs[i]
            if d['title'] != best_title:
                continue
            if s < 0.5 * top_s:
                break
            out.append(dict(id=d['id'], title=d['title'], url=d.get('url', ''), text=d['text'],
                            score=round(s, 4), overlap=ov))
            if len(out) >= k:
                break
        return out

    @classmethod
    def load(cls, path=DEFAULT_CORPUS):
        path = Path(path)
        docs = []
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if is_junk_chunk(d.get('text') or ''):
                continue
            docs.append(d)
        if not docs:
            raise SystemExit(f'empty wiki corpus after junk filter: {path}')
        return cls(docs)


def attach_rag(items, corpus=DEFAULT_CORPUS, k=2, max_chars=1100, skip_essays=True):
    """Mutates items: sets rag_context + rag_passage_ids. Empty rag_context = no inject."""
    index = WikiIndex.load(corpus)
    for it in items:
        if skip_essays and it.get('type') == 'essay':
            continue
        hits = index.search(query_from_item(it), k=k)
        it['rag_passages'] = hits
        it['rag_context'] = format_hits(hits, max_chars=max_chars)
        it['rag_passage_ids'] = [h['id'] for h in hits]
    return index
