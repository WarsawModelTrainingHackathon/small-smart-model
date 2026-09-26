"""Optional Wikipedia RAG: prompt block (matura_format) and eval wiring (eval_matura --rag-index).

scripts/wiki_rag.py (WikiIndex(index_dir).search(query, k) -> [{title, text, score}], optional build_query(item))
is replaced by a fake module here, so these tests need neither the index nor a network.
"""
import hashlib
import json
import sys
import types

import pytest

import eval_matura as em
import matura_format as mf

# sha256 of build_messages() for every fixture item (images + text-only) computed with the prompt code BEFORE
# RAG was added. The default prompt must stay byte-identical: adapters are trained on it. Only update this
# digest for a deliberate prompt change (and then retrain / re-evaluate).
DEFAULT_PROMPT_SHA256 = '9c93ce530059a7d933b7c1e09205a4fe204fe2a1263727852f9761edef2e366f'

PASSAGES = [
    {'title': 'Bitwa pod Grunwaldem', 'text': 'Starcie zbrojne 15 lipca 1410 roku.\n\nWojska polsko-litewskie '
                                             'pokonały zakon krzyżacki.', 'score': 12.5},
    {'title': 'Władysław II Jagiełło', 'text': 'Król Polski w latach 1386–1434, wielki książę litewski.', 'score': 9.1},
    {'title': '', 'text': 'Fragment bez tytułu.', 'score': 1.0},
]


def long_passages(n=5, words=200):
    return [{'title': f'Hasło {i}', 'text': ' '.join(f'słowo{i}_{j}' for j in range(words)), 'score': float(n - i)}
            for i in range(n)]


# ----------------------------------------------------------------- default prompt unchanged

def test_default_prompt_is_byte_identical(items):
    rows = [[i, t, mf.build_messages(items[i], text_only=t)] for i in sorted(items) for t in (False, True)]
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    assert digest == DEFAULT_PROMPT_SHA256
    empty_hits = [{'title': 'Pusty', 'text': '  \n ', 'score': 1.0}]
    for it in items.values():
        for t in (False, True):
            base = mf.user_text(it, t)
            assert mf.user_text(it, t, passages=None) == base
            assert mf.user_text(it, t, passages=[]) == base
            assert mf.user_text(it, t, passages=empty_hits) == base  # nothing to insert -> no header either
            assert mf.build_messages(it, t, passages=[]) == mf.build_messages(it, t)
            assert mf.RAG_HEADER not in base


# ----------------------------------------------------------------- prompt insertion

def test_block_goes_first_before_exam_materials(items):
    it = items['2015-maj-f2015-R-zad18']  # has an image and a context
    for text_only in (False, True):
        base = mf.user_text(it, text_only)
        text = mf.user_text(it, text_only, passages=PASSAGES)
        block = mf.passages_block(PASSAGES)
        assert text == block + '\n\n' + base  # the exam part itself is untouched
        assert text.startswith(mf.RAG_HEADER + '\n[1] Bitwa pod Grunwaldem: Starcie zbrojne 15 lipca 1410 roku. Wojska')
        assert '[2] Władysław II Jagiełło: Król Polski' in text and '[3] Fragment bez tytułu.' in text
        assert text.index('[3] Fragment') < text.index('Materiały do zadania:') < text.index('Zadanie ')
        assert text.rstrip().endswith('Odpowiedź: X')
    msgs = mf.build_messages(it, passages=PASSAGES)
    parts = msgs[1]['content']
    assert [c['type'] for c in parts] == ['image'] * len(it['images']) + ['text']  # images stay first
    assert parts[-1]['text'].startswith(mf.RAG_HEADER)
    assert msgs[0] == mf.build_messages(it)[0]  # system prompt unchanged


def test_render_prompt_passes_passages(items):
    class Proc:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
            return json.dumps([msgs, enable_thinking], ensure_ascii=False)
    it = items['2024-maj-R-zad11.1']
    with_rag = mf.render_prompt(Proc(), it, text_only=True, passages=PASSAGES)
    assert mf.RAG_HEADER in with_rag and 'Bitwa pod Grunwaldem' in with_rag
    assert mf.render_prompt(Proc(), it, text_only=True) == json.dumps([mf.build_messages(it, True), False], ensure_ascii=False)


# ----------------------------------------------------------------- truncation

@pytest.mark.parametrize('max_chars', [250, 400, 1000, 3000, 8000])
def test_block_respects_budget(max_chars):
    ps = long_passages()
    block = mf.passages_block(ps, max_chars)
    assert len(block) <= max_chars
    lines = block.split('\n')
    assert lines[0] == mf.RAG_HEADER
    fitted = mf.fit_passages(ps, max_chars)
    assert len(lines) == 1 + len(fitted) >= 2
    for i, line in enumerate(lines[1:]):  # rank order, contiguous numbering, one line per passage
        assert line.startswith(f'[{i + 1}] Hasło {i}: słowo{i}_0 ')
    if len(fitted) < len(ps) or fitted[-1][1].endswith('…'):
        assert max_chars - len(block) < 2 * mf.MIN_RAG_TRUNCATED_CHARS  # the budget is actually used


def test_truncated_passage_is_cut_at_a_word():
    ps = long_passages(n=1, words=1000)
    title, text = mf.fit_passages(ps, 1000)[0]
    assert text.endswith('…') and len(text) < len(ps[0]['text'])
    assert ps[0]['text'].startswith(text[:-1]) and text[-2] != ' '
    assert ps[0]['text'][len(text) - 1] == ' '  # the cut lands on a word boundary


def test_no_room_means_no_block(items):
    assert mf.passages_block(PASSAGES, len(mf.RAG_HEADER) + 20) == ''
    assert mf.passages_block(PASSAGES, 0) == ''
    it = items['2015-maj-f2015-R-zad18']
    assert mf.user_text(it, passages=PASSAGES, rag_max_chars=10) == mf.user_text(it)


def test_small_passage_after_budget_is_kept_whole_and_later_ones_dropped():
    ps = [{'title': 'A', 'text': 'krótki tekst'}, {'title': 'B', 'text': 'x ' * 5000}, {'title': 'C', 'text': 'też krótki'}]
    fitted = mf.fit_passages(ps, 600)
    assert fitted[0] == ('A', 'krótki tekst')
    assert fitted[1][0] == 'B' and fitted[1][1].endswith('…')
    assert len(fitted) == 2  # C comes after the passage that crossed the budget
    assert mf.fit_passages(['tylko tekst'], 500) == [('', 'tylko tekst')]


# ----------------------------------------------------------------- retrieval glue

def fake_wiki_rag(monkeypatch, with_build_query=True, stats=None):
    """Install a fake scripts/wiki_rag.py; returns the stats dict (opened index dirs, queries)."""
    import numpy as np
    stats = stats if stats is not None else {'opened': [], 'queries': []}

    class WikiIndex:
        def __init__(self, index_dir):
            stats['opened'].append(index_dir)

        def search(self, query, k=5):
            stats['queries'].append(query)
            return [{'title': f'Hasło {i}', 'text': f'FAKEPASSAGE{i} ' + 'tekst hasła ' * 40, 'score': np.float32(10 - i)}
                    for i in range(k + 2)]  # more than k: the harness keeps k

    mod = types.ModuleType('wiki_rag')
    mod.WikiIndex = WikiIndex
    if with_build_query:
        mod.build_query = lambda item: 'Q:' + item['id']
    monkeypatch.setitem(sys.modules, 'wiki_rag', mod)
    return stats


def test_query_source_selection(monkeypatch, items, tmp_path):
    fake_wiki_rag(monkeypatch, with_build_query=True)
    _, build_query, src = em.open_rag_index(tmp_path)
    assert src == 'wiki_rag.build_query' and build_query({'id': 'x'}) == 'Q:x'
    fake_wiki_rag(monkeypatch, with_build_query=False)
    index, build_query, src = em.open_rag_index(tmp_path)
    assert src == 'fallback' and build_query is em.fallback_query
    index.build_query = lambda item: 'method'
    sys.modules['wiki_rag'].WikiIndex = lambda d: index
    assert em.open_rag_index(tmp_path)[2] == 'WikiIndex.build_query'
    q = em.fallback_query(items['2015-maj-f2015-R-zad18'])
    assert q.startswith(' '.join(items['2015-maj-f2015-R-zad18']['question'].split()[:5])) and len(q) <= 1000
    assert '\n' not in q


def test_missing_wiki_rag_module_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'wiki_rag', None)  # import fails
    with pytest.raises(SystemExit, match='wiki_rag'):
        em.open_rag_index(tmp_path)


# ----------------------------------------------------------------- end-to-end on the tiny model

def run(tiny_model, data, out, *extra):
    return em.main(['--model', tiny_model, '--data', str(data), '--judge', 'none',
                    '--max-new-tokens', '8', '--essay-max-new-tokens', '8', '--image-max-soft-tokens', '70',
                    '--batch-size', '2', '--output', str(out), *extra])


def spy_prompts(monkeypatch):
    seen = []
    orig = mf.render_prompt

    def spy(processor, item, *args, **kw):
        text = orig(processor, item, *args, **kw)
        seen.append((item['id'], text))
        return text
    monkeypatch.setattr(mf, 'render_prompt', spy)
    return seen


def check_rag_run(out, rep, ids, prompts, k=5, max_chars=mf.DEFAULT_RAG_MAX_CHARS):
    gens = [json.loads(x) for x in (out / 'generations.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sorted(g['id'] for g in gens) == sorted(ids)
    for g in gens:
        assert [h['title'] for h in g['rag']['hits']] == [f'Hasło {i}' for i in range(k)]
        assert [h['score'] for h in g['rag']['hits']] == [float(10 - i) for i in range(k)]
        assert 1 <= g['rag']['used'] <= k and 0 < g['rag']['chars'] <= max_chars
    ret = [json.loads(x) for x in (out / 'retrieval.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sorted(r['id'] for r in ret) == sorted(ids)
    assert all(len(r['passages']) == k and r['passages'][0]['text'].startswith('FAKEPASSAGE0') for r in ret)
    assert {i for i, _ in prompts} == set(ids)
    for _, text in prompts:
        assert mf.RAG_HEADER in text and '[1] Hasło 0: FAKEPASSAGE0' in text
        assert text.index(mf.RAG_HEADER) < text.index('Zadanie ')
    meta = rep['meta']
    assert meta['rag']['k'] == k and meta['rag']['max_chars'] == max_chars
    assert meta['rag_stats']['n_items'] == len(ids) and meta['rag_stats']['mean_top_score'] == 10.0
    assert '- **rag**: Wikipedia k=' in (out / 'report.md').read_text(encoding='utf-8')
    return gens


def test_rag_eval_smoke_resume_regrade(tiny_model, mini_data, tmp_path, monkeypatch):
    stats = fake_wiki_rag(monkeypatch)
    prompts = spy_prompts(monkeypatch)
    index_dir = tmp_path / 'wiki-index'
    index_dir.mkdir()
    out = tmp_path / 'rag'
    args = ('--split', 'dev', '--limit', '4', '--rag-index', str(index_dir))
    rep = run(tiny_model, mini_data, out, *args, '--label', 'tiny-dev')  # multimodal (fixture items have images)
    ids = json.loads((out / 'run.json').read_text())['item_ids']
    assert rep['meta']['label'] == 'tiny-dev-rag5' and rep['meta']['text_only'] is False
    check_rag_run(out, rep, ids, prompts)
    assert sorted(stats['queries']) == sorted('Q:' + i for i in ids)  # once per item, via wiki_rag.build_query
    assert stats['opened'] == [str(index_dir)]  # the index is opened once

    # resume: nothing to generate -> no retrieval, same report
    rep2 = run(tiny_model, mini_data, out, *args, '--label', 'tiny-dev')
    assert len(stats['queries']) == len(ids) and len(stats['opened']) == 1
    assert rep2['summary']['n_generations'] == len(ids)

    # regrade: saved generations only, no index needed, RAG metadata kept from run.json
    monkeypatch.setitem(sys.modules, 'wiki_rag', None)
    rep3 = run(tiny_model, mini_data, out, '--split', 'dev', '--limit', '4', '--regrade')
    assert rep3['meta']['rag']['k'] == 5 and rep3['meta']['label'] == 'tiny-dev-rag5'
    rep4 = run(tiny_model, mini_data, out, *args, '--regrade')
    assert rep4['summary'] == rep3['summary']

    # a RAG run dir cannot be resumed without RAG (or with other RAG settings) and vice versa
    with pytest.raises(SystemExit, match='rag'):
        run(tiny_model, mini_data, out, '--split', 'dev', '--limit', '4')
    with pytest.raises(SystemExit, match='rag'):
        run(tiny_model, mini_data, out, *args, '--rag-k', '3')


def test_plain_run_never_imports_wiki_rag(tiny_model, mini_data, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'wiki_rag', None)  # any import would raise
    prompts = spy_prompts(monkeypatch)
    rep = run(tiny_model, mini_data, tmp_path / 'plain', '--split', 'dev', '--limit', '2', '--text-only')
    assert rep['meta']['rag'] is None and 'rag' not in rep['meta']['label']
    assert not (tmp_path / 'plain' / 'retrieval.jsonl').exists()
    gens = [json.loads(x) for x in (tmp_path / 'plain' / 'generations.jsonl').read_text().splitlines()]
    assert all('rag' not in g for g in gens)
    assert prompts and all(mf.RAG_HEADER not in t for _, t in prompts)
    assert '**rag**' not in (tmp_path / 'plain' / 'report.md').read_text(encoding='utf-8')
    with pytest.raises(SystemExit, match='rag'):  # plain run dir, resumed with RAG
        run(tiny_model, mini_data, tmp_path / 'plain', '--split', 'dev', '--limit', '2', '--text-only',
            '--rag-index', str(tmp_path))


@pytest.fixture(scope='module')
def tiny_adapter(tiny_model, tmp_path_factory):
    """A (random) LoRA adapter on the tiny model, saved like train_lora.py does."""
    from peft import LoraConfig, get_peft_model
    import train_lora as tl
    model, _ = em.load_model(tiny_model, device='cpu')
    targets = tl.lora_target_names(model, tl.DEFAULT_TARGETS.split(','))
    out = tmp_path_factory.mktemp('adapter')
    get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=targets)).save_pretrained(out)
    return str(out)


def real_dev_ids(data_path):
    """3 real dev items: a closed item with an image, a closed item without one, an open item without one."""
    dev = mf.load_data(data_path, splits=['dev'])['dev']
    pick = [next(it for it in dev if mf.auto_gradable(it) and it.get('images')),
            next(it for it in dev if mf.auto_gradable(it) and not it.get('images')),
            next(it for it in dev if it['type'] in mf.OPEN_TYPES and not it.get('images'))]
    return [it['id'] for it in pick]


@pytest.mark.parametrize('mode', ['multimodal', 'text-only'])
def test_rag_real_dev_items_with_adapter(tiny_model, tiny_adapter, full_data, tmp_path, monkeypatch, mode):
    stats = fake_wiki_rag(monkeypatch, with_build_query=(mode == 'multimodal'))
    prompts = spy_prompts(monkeypatch)
    ids = real_dev_ids(full_data)
    out = tmp_path / mode
    extra = ['--text-only'] if mode == 'text-only' else []
    rep = run(tiny_model, full_data, out, '--split', 'dev', '--ids', ','.join(ids), '--adapter', tiny_adapter,
              '--rag-index', str(tmp_path), '--rag-k', '3', '--rag-max-chars', '1200', *extra)
    assert rep['meta']['adapter'] == tiny_adapter and rep['meta']['text_only'] is (mode == 'text-only')
    assert rep['meta']['label'].endswith('-rag3c1200')
    check_rag_run(out, rep, ids, prompts, k=3, max_chars=1200)
    if mode == 'multimodal':
        assert stats['queries'] and all(q.startswith('Q:') for q in stats['queries'])
        assert any('<|image|>' in t for _, t in prompts)  # the image item really went in with its image
    else:
        assert rep['meta']['rag_stats']['query_source'] == 'fallback'
        assert all(not q.startswith('Q:') and len(q) <= 1000 for q in stats['queries'])
    rep2 = run(tiny_model, full_data, out, '--split', 'dev', '--ids', ','.join(ids), '--regrade')
    assert rep2['summary']['n_generations'] == 3 and rep2['meta']['rag']['k'] == 3
