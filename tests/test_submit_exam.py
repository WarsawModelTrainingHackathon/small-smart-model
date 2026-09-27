import submit_exam as se
import matura_format as mf


def test_classify_closed_and_essay():
    assert se.classify('A', 'Dokończ zdanie')[0] == 'closed_abcd'
    typ, key = se.classify('1: P\n2: F\n3: P')
    assert typ == 'true_false' and list(key) == ['1', '2', '3']
    typ, key = se.classify('A: 1\nB: 1')
    assert typ == 'matching' and key['A'] == '1'
    assert se.classify('Jeden tekst: numer wybranego tematu i całe wypracowanie.')[0] == mf.ESSAY
    assert se.classify('Tekst po polsku. Podaj wszystkie wymagane elementy odpowiedzi.')[0] == 'open_short'


def test_remap_and_validate(tmp_path):
    assert se.remap_closed('1-P; 2-F; 3-P', '1: P\n2: F\n3: P') == '1: P\n2: F\n3: P'
    assert se.remap_closed('wybieram B bo...', 'A') == 'B'
    exam = {'exam_id': 'history-2023-mock-v1', 'items': [{'id': '1', 'answer_format': 'A', 'question': 'x'}]}
    gens = [{'id': '1', 'output': 'bo mapa.\nOdpowiedź: C'}]
    se.write_answers(exam, gens, tmp_path / 'answers.json')
    payload = (tmp_path / 'answers.json').read_text(encoding='utf-8')
    assert '"exam_id": "history-2023-mock-v1"' in payload
    assert '"answer": "C"' in payload
