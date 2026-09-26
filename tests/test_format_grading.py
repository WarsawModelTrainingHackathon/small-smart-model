"""Parsing / prompt / grading on REAL CKE items (tests/fixtures/items.json)."""
import json

import pytest

import matura_format as mf
import matura_grading as mg


def closed(items, i):
    return items[i]


# ----------------------------------------------------------------- final-answer parsing

def test_only_last_answer_line_counts(items):
    it = items['2015-maj-f2015-R-zad18']  # key C
    text = 'Na początku myślałem, że A.\nOdpowiedź: B\nAle jednak nie.\nOdpowiedź: C'
    assert mf.parse_prediction(it, text) == 'C'
    assert mg.grade_closed(it, text)['points'] == 1


def test_letters_mid_reasoning_are_ignored(items):
    it = items['2015-maj-f2015-R-zad18']
    text = 'Odpowiedź C jest poprawna, bo rewolucja kulturalna... Wybieram C.'
    assert mf.final_answer(text) is None
    r = mg.grade_closed(it, text)
    assert r['points'] == 0 and not r['parse_ok']


def test_thinking_block_is_stripped(items):
    it = items['2015-maj-f2015-R-zad18']
    text = '<|channel>thought\nMoże Odpowiedź: A\nOdpowiedź: A<channel|>Krótko: to rewolucja kulturalna.\nOdpowiedź: C'
    assert mf.parse_prediction(it, text) == 'C'
    assert mf.parse_prediction(it, '<think>Odpowiedź: C</think>\nOdpowiedź: A') == 'A'


@pytest.mark.parametrize('text', ['Odpowiedź: C', 'odpowiedz: c.', '**Odpowiedź:** C. Rewolucją Kulturalną.',
                                  'Odpowiedź końcowa: C', 'Odpowiedź:\nC'])
def test_answer_line_variants(items, text):
    assert mf.parse_prediction(items['2015-maj-f2015-R-zad18'], text) == 'C'


def test_placeholder_is_not_an_answer(items):
    assert mf.parse_prediction(items['2015-maj-f2015-R-zad18'], 'Odpowiedź: X') is None


# ----------------------------------------------------------------- closed grading

def test_true_false_all_or_nothing(items):
    it = items['2015-maj-f2015-R-zad8.3']  # 1-F 2-P 3-P, 1 pkt for all three
    assert mg.grade_closed(it, 'Odpowiedź: 1-F; 2-P; 3-P')['points'] == 1
    assert mg.grade_closed(it, 'Odpowiedź: 1-F; 2-P; 3-F')['points'] == 0
    assert mg.grade_closed(it, 'Odpowiedź: 1 – fałsz, 2 – prawda, 3 – prawda')['points'] == 1
    assert mg.grade_closed(it, 'Odpowiedź: F P P')['points'] == 1


def test_true_false_partial_credit(items):
    it = items['2023-maj-R-zad3']  # 1-F 2-P 3-P; 2 pkt / 1 pkt for two correct
    assert mg.credit_table(it) == [(3, 2), (2, 1)]
    assert mg.grade_closed(it, 'Odpowiedź: 1-F; 2-P; 3-P')['points'] == 2
    assert mg.grade_closed(it, 'Odpowiedź: 1-F; 2-P; 3-F')['points'] == 1
    assert mg.grade_closed(it, 'Odpowiedź: 1-P; 2-F; 3-P')['points'] == 0


def test_multi_part_abcd_partial(items):
    it = items['2016-maj-f2015-R-zad25.1']  # {'1': 'B', '2': 'A'}, 2/1 pkt
    assert mf.answer_format(it) == 'Odpowiedź: 1-…; 2-…'
    assert mg.grade_closed(it, 'Odpowiedź: 1-B; 2-A')['points'] == 2
    assert mg.grade_closed(it, 'Odpowiedź: 1-B; 2-C')['points'] == 1
    assert mg.grade_closed(it, 'Odpowiedź: 1. B, 2. A')['points'] == 2


def test_matching_five_parts_partial(items):
    it = items['2022-marzec-pokazowy-R-zad7']  # 2 pkt for 5, 1 pkt for 3-4
    gold = it['answer_key']
    full = 'Odpowiedź: ' + '; '.join(f'{k}-{v}' for k, v in gold.items())
    assert mg.grade_closed(it, full)['points'] == 2
    four = full.replace(gold['E'], 'Amerigo Vespucci')
    assert mg.grade_closed(it, four)['points'] == 1
    two = four.replace(gold['D'], 'Kolumb').replace(gold['C'], 'Magellan')
    assert mg.grade_closed(it, two)['points'] == 0


def test_matching_normalization(items):
    it = items['2024-maj-R-zad11.1']  # A: Karol IX, B: Henryk IV
    for text in ['Odpowiedź: A-Karol IX; B-Henryk IV', 'Odpowiedź: A – karol ix, B – HENRYK IV.',
                 'Odpowiedź: A. Karol IX; B. Henryk IV', 'Odpowiedź: Karol IX; Henryk IV']:
        assert mg.grade_closed(it, text)['points'] == 1, text
    assert mg.grade_closed(it, 'Odpowiedź: A-Karol X; B-Henryk IV')['points'] == 0


def test_matching_alternatives_and_optional_parts(items):
    it = items['2018-maj-f2015-R-zad10.2']  # 1 Kłuszyn, 2 Kircholm / Chocim, 3 Wiedeń / Chocim
    assert mg.grade_closed(it, 'Odpowiedź: 1-Kluszyn; 2-Chocim; 3-Wieden')['points'] == 1
    assert mg.grade_closed(it, 'Odpowiedź: 1-Kłuszyn; 2-Kircholm; 3-Wiedeń')['points'] == 1
    pepin = items['2018-maj-f2015-R-zad5.1']  # 'Pepin Mały / Krótki', 'Karol Młot / Martel'
    assert mg.grade_closed(pepin, 'Odpowiedź: 1-Pepin Krótki; 2-Karol Martel; 3-Karol Wielki')['points'] == 1
    assert mg.grade_closed(pepin, 'Odpowiedź: 1-Pepin Mały; 2-Karol Młot; 3-Karol Wielki')['points'] == 1
    boles = items['2022-marzec-pokazowy-R-zad6.1']  # 'Bolesław Śmiały [Szczodry]'
    base = 'Odpowiedź: A-Kazimierz Odnowiciel; B-Bolesław Chrobry; C-'
    assert mg.grade_closed(boles, base + 'Bolesław Śmiały')['points'] == 2
    assert mg.grade_closed(boles, base + 'Bolesław Śmiały Szczodry')['points'] == 2
    usa = items['2025-maj-R-zad11.1']  # 'Stany Zjednoczone [Ameryki]', 'Polska / Rzeczpospolita'
    assert mg.grade_closed(usa, 'Odpowiedź: A-Stany Zjednoczone Ameryki; B-Francja; C-Rzeczpospolita')['points'] == 2


def test_matching_long_labels(items):
    it = items['2022-grudzien-diagnostyczny-R-zad10.1']
    text = 'Odpowiedź: Król Czech i Węgier: Władysław; Król Polski i Wielki Książę Litewski: Zygmunt'
    assert mg.grade_closed(it, text)['points'] == 1
    assert mg.grade_closed(it, 'Odpowiedź: Król Czech i Węgier – Zygmunt; Król Polski i Wielki Książę Litewski – Władysław')['points'] == 0


def test_gold_targets_round_trip(items):
    for it in items.values():
        if mf.is_closed(it):
            r = mg.grade_closed(it, mf.build_target(it))
            assert r['n_correct'] == r['n_parts'], it['id']


# ----------------------------------------------------------------- prompts / targets

def test_prompt_does_not_repeat_options(items):
    it = items['2015-maj-f2015-R-zad18']
    text = mf.user_text(it)
    assert text.count('Długim Marszem') == 1
    assert text.rstrip().endswith('Odpowiedź: X')
    tf = mf.user_text(items['2015-maj-f2015-R-zad8.3'])
    assert tf.count('Ojciec i dziadek króla') == 1
    assert 'Odpowiedź: 1-…; 2-…; 3-…' in tf


def test_messages_images_and_text_only(items):
    it = items['2015-maj-f2015-R-zad18']
    msgs = mf.build_messages(it)
    assert msgs[0]['role'] == 'system' and msgs[1]['role'] == 'user'
    assert [c['type'] for c in msgs[1]['content']] == ['image'] * len(it['images']) + ['text']
    msgs = mf.build_messages(it, text_only=True)
    assert [c['type'] for c in msgs[1]['content']] == ['text']
    assert mf.NO_IMAGE_NOTE in msgs[1]['content'][0]['text'] or 'Opis materiału' in msgs[1]['content'][0]['text']


def test_targets(items):
    assert mf.build_target(items['2015-maj-f2015-R-zad18']).endswith('Odpowiedź: C')
    assert mf.build_target(items['2015-maj-f2015-R-zad8.3']).endswith('Odpowiedź: 1-F; 2-P; 3-P')
    t = mf.build_target(items['2018-maj-f2015-R-zad10.2'])
    assert t.endswith('Odpowiedź: 1-Kłuszyn; 2-Kircholm; 3-Wiedeń')
    just = mf.build_target(items['2016-maj-f2015-R-zad23'])
    assert 'ZSRR' in just and just.endswith('Odpowiedź: C')
    open_long = mf.build_target(items['2015-maj-f2015-R-zad4'])
    assert not open_long.startswith('Przykładowa') and 'Anioł nakłada' in open_long
    assert mf.build_target(items['2025-maj-R-zad25-t1']) is None  # essay


# ----------------------------------------------------------------- judge + report

def test_parse_judge():
    assert mg.parse_judge('bla {"points": 2, "reason": "ok"}', 1) == {'points': 1, 'reason': 'ok'}
    assert mg.parse_judge('```json\n{"points": 0, "reason": "zle"}\n```', 2)['points'] == 0
    assert mg.parse_judge('nic', 2) is None


def test_justification_item(items):
    it = items['2016-maj-f2015-R-zad23']  # closed + uzasadnij -> not auto-gradable
    by_id = {it['id']: it}
    wrong = mg.grade_records(by_id, [{'id': it['id'], 'output': 'Bo tak.\nOdpowiedź: A'}], mg.NoJudge())[0]
    assert wrong['points'] == 0 and wrong['method'] == 'auto-wrong-choice'
    right = mg.grade_records(by_id, [{'id': it['id'], 'output': 'ZSRR i USA w wojnie od 1941.\nOdpowiedź: C'}], mg.NoJudge())[0]
    assert right['points'] is None and right['method'] == 'ungraded'

    class Fake:
        name = 'fake'

        def __call__(self, prompts):
            assert 'Zasady oceniania CKE' in prompts[0] and 'ZSRR' in prompts[0]
            return ['{"points": 1, "reason": "dobre uzasadnienie"}']
    judged = mg.grade_records(by_id, [{'id': it['id'], 'output': 'ZSRR i USA.\nOdpowiedź: C'}], Fake())[0]
    assert judged['points'] == 1


def test_one_essay_per_session(items):
    ids = ['2025-maj-R-zad25-t1', '2025-maj-R-zad25-t2', '2025-maj-R-zad25-t3', '2025-maj-R-zad11.1']
    by_id = {i: items[i] for i in ids}

    class Fake:
        name = 'fake'
        pts = iter([15, 9, 0])

        def __call__(self, prompts):
            return [json.dumps({'points': next(self.pts), 'reason': ''}) for _ in prompts]
    gens = [{'id': i, 'output': 'Wypracowanie...'} for i in ids[:3]] + [{'id': ids[3], 'output': 'Odpowiedź: A-Stany Zjednoczone; B-Francja; C-Polska'}]
    graded = mg.grade_records(by_id, gens, Fake())
    s = mg.summarize(graded, by_id)
    assert s['exam_max'] == 15 + 2
    assert s['by_type']['essay']['max'] == 15 and s['by_type']['essay']['points'] == 8.0
    assert s['total']['points'] == 10.0
    assert mg.summarize(graded, by_id, essay_policy='best')['by_type']['essay']['points'] == 15
    md = mg.report_markdown(s, {'label': 'x'})
    assert 'By session' in md and '2025-maj' in md


def test_choice_group_field_supported(items):
    a = dict(items['2025-maj-R-zad25-t1'], choice_group='zad25')
    b = dict(items['2025-maj-R-zad25-t2'], choice_group='zad25', group=99)
    assert mf.essay_group(a) == mf.essay_group(b)


# ----------------------------------------------------------------- full dataset (skipped if absent)

def test_full_dataset_invariants(full_data):
    d = mf.load_data(full_data)
    allr = [r for rows in d.values() for r in rows]
    for it in allr:
        if mf.is_closed(it):
            r = mg.grade_closed(it, mf.build_target(it))
            assert r['n_correct'] == r['n_parts'], it['id']
            assert mf.answer_format(it) in mf.user_text(it)
    by_id = {r['id']: r for r in allr}
    gens = [{'id': r['id'], 'output': mf.build_target(r) or ''} for r in d['test']]
    s = mg.summarize(mg.grade_records(by_id, gens, mg.NoJudge()), by_id)
    assert s['exam_max'] == 120
    assert all(v['exam_max'] == 60 for v in s['by_session'].values())
    assert s['closed_items']['pct'] == 100.0
    train_sessions = {r['session_key'] for r in d['train']}
    assert not train_sessions & {r['session_key'] for r in d['dev'] + d['test']}
