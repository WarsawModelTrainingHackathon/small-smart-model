"""CPU tests for BM25 wiki harness (no downloads, no GPU)."""
import json

import matura_format as mf
import wiki_bm25


def test_bm25_ranks_relevant_chunk(tmp_path):
    docs = [
        {'id': 'a#0', 'title': 'Juliusz Cezar', 'url': 'u', 'text': 'Gajusz Juliusz Cezar został zamordowany w 44 p.n.e. Panował krótko.'},
        {'id': 'b#0', 'title': 'Oktawian August', 'url': 'u',
         'text': 'Oktawian August sprawował władzę przez prawie pół wieku i ustanowił pryncypat, monarchię o pozorach republiki.'},
        {'id': 'c#0', 'title': 'Wikingowie', 'url': 'u', 'text': 'Wyprawy wikingów na Islandię i Ruś w VIII–XI wieku.'},
    ]
    p = tmp_path / 'c.jsonl'
    p.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in docs), encoding='utf-8')
    idx = wiki_bm25.WikiIndex.load(p)
    hits = idx.search('prawie pół wieku monarchia pozory republiki', k=2)
    assert hits[0]['title'] == 'Oktawian August'
    hits2 = idx.search('mapa wyprawy wikingów Islandię', k=1)
    assert hits2[0]['title'] == 'Wikingowie'


def test_user_text_injects_rag(items):
    it = dict(items['2015-maj-f2015-R-zad18'])
    it['rag_context'] = '[Oktawian August] princpat test chunk'
    text = mf.user_text(it)
    assert 'Wikipedii' in text and 'princpat test chunk' in text
    assert 'NIE jest źródło z arkusza' in text


def test_attach_skips_essays(tmp_path, items):
    docs = [{'id': 'x#0', 'title': 'T', 'url': 'u', 'text': 'Jan II Kazimierz Waza abdykował po śmierci Ludwiki Marii.'}]
    p = tmp_path / 'c.jsonl'
    p.write_text(json.dumps(docs[0], ensure_ascii=False) + '\n', encoding='utf-8')
    essay = dict(next(v for v in items.values() if v['type'] == 'essay'))
    other = dict(next(v for v in items.values() if v['type'] != 'essay'))
    wiki_bm25.attach_rag([essay, other], corpus=p, skip_essays=True)
    assert not essay.get('rag_context')
    assert other.get('rag_context')
