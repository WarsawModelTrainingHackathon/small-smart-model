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
# BM25
K1, B = 1.5, 0.75


def tokenize(text):
    text = unicodedata.normalize('NFKC', text or '').lower()
    return _TOKEN.findall(text)


def query_from_item(item):
    """Search string: question + a bit of CKE context (not the whole source dump)."""
    ctx = (item.get('context') or '')[:800]
    q = (item.get('question') or '')
    return f"{item.get('task', '')} {q}\n{ctx}"


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

    @classmethod
    def load(cls, path=DEFAULT_CORPUS):
        path = Path(path)
        docs = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
        if not docs:
            raise SystemExit(f'empty wiki corpus: {path}')
        return cls(docs)

    def search(self, query, k=3):
        q = tokenize(query)
        if not q:
            return []
        scores = []
        for i, bag in enumerate(self.tf):
            s = 0.0
            dl = self.dl[i]
            for t in q:
                idf = self.idf.get(t)
                if not idf:
                    continue
                f = bag.get(t, 0)
                s += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / self.avgdl))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        out = []
        for s, i in scores[:k]:
            d = self.docs[i]
            out.append(dict(id=d['id'], title=d['title'], url=d.get('url', ''), text=d['text'], score=round(s, 4)))
        return out


def attach_rag(items, corpus=DEFAULT_CORPUS, k=3, max_chars=1400, skip_essays=True):
    """Mutates items: sets rag_context + rag_passage_ids. Returns the index."""
    index = WikiIndex.load(corpus)
    for it in items:
        if skip_essays and it.get('type') == 'essay':
            continue
        hits = index.search(query_from_item(it), k=k)
        it['rag_passages'] = hits
        it['rag_context'] = format_hits(hits, max_chars=max_chars)
        it['rag_passage_ids'] = [h['id'] for h in hits]
    return index
