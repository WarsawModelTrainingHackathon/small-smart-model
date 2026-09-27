"""Shared prompt / target / answer-parsing module for the matura z historii tasks.

THE single source of truth used by BOTH scripts/eval_matura.py and scripts/train_lora.py,
so the model is trained on exactly the prompt it is evaluated with and the dev metric
during training uses exactly the parser of the final benchmark.

Dataset: data/matura-historia/data.json ({train, dev, test}); image paths are relative
to the folder of data.json. Pure python (no torch), importable from tests.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

DEFAULT_DATA = 'data/matura-historia/data.json'
CLOSED_TYPES = ('closed_abcd', 'true_false', 'matching')
OPEN_TYPES = ('open_short', 'open_long')
ESSAY = 'essay'
ANSWER_PREFIX = 'Odpowiedź:'

SYSTEM_PROMPT = (
    'Jesteś bardzo dobrym uczniem, który zdaje maturę z historii na poziomie rozszerzonym. '
    'Rozwiązujesz zadanie z arkusza CKE. Czytaj uważnie materiały źródłowe (teksty, mapy, ilustracje, tabele) '
    'i łącz je z własną wiedzą historyczną. Odpowiadaj po polsku, rzeczowo i zgodnie z poleceniem, '
    'tak jak w arkuszu egzaminacyjnym: podawaj konkretne fakty, nazwy, daty i postaci.'
)

NO_IMAGE_NOTE = '[Materiał graficzny do tego zadania nie jest dostępny – korzystaj z opisu i własnej wiedzy.]'


# --------------------------------------------------------------------------- data

def load_data(path=DEFAULT_DATA, splits=None):
    """Load data.json -> {split: [item, ...]}; each item gets `_root` (folder for images)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    root = str(path.resolve().parent)
    out = {}
    for split, rows in data.items():
        if splits and split not in splits:
            continue
        if not isinstance(rows, list):
            continue
        out[split] = [dict(r, _root=root) for r in rows]
    return out


def essay_group(item):
    """Key of the set of alternative essay topics a candidate chooses ONE from."""
    cg = item.get('choice_group')
    if cg not in (None, ''):
        return f"{item['session_key']}::{cg}"
    return f"{item['session_key']}::{item.get('group')}"


def is_closed(item):
    return item.get('type') in CLOSED_TYPES and item.get('answer_key') not in (None, '', {})


def auto_gradable(item):
    """Fully gradable by the parser: closed, with a key and no required justification."""
    return bool(item.get('auto_gradable')) and is_closed(item)


def key_labels(item):
    key = item.get('answer_key')
    return list(key.keys()) if isinstance(key, dict) else []


# --------------------------------------------------------------------------- prompt

def answer_format(item):
    """Example of the required final line for a closed item (None for open items)."""
    if not is_closed(item):
        return None
    key = item['answer_key']
    if not isinstance(key, dict):
        return f'{ANSWER_PREFIX} X'
    return f'{ANSWER_PREFIX} ' + '; '.join(f'{k}-…' for k in key)


def instruction(item):
    t = item.get('type')
    fmt = answer_format(item)
    if fmt:
        if t == 'true_false':
            what = 'Dla każdego zdania rozstrzygnij: P (prawda) albo F (fałsz).'
            where = 'w miejsce … wpisz P albo F'
        elif t == 'closed_abcd' and isinstance(item['answer_key'], dict):
            what = 'Dla każdej części zadania wybierz jedną poprawną odpowiedź.'
            where = 'w miejsce … wpisz literę A, B, C albo D'
        elif t == 'closed_abcd':
            what = 'Wybierz jedną poprawną odpowiedź.'
            where = 'X to litera A, B, C albo D'
        else:
            what = 'Dla każdego elementu podaj właściwe przyporządkowanie.'
            where = 'w miejsce … wpisz odpowiedź (nazwę, imię, literę) dla danego elementu'
        just = (' Zadanie wymaga uzasadnienia – napisz je przed ostatnim wierszem.'
                if item.get('requires_justification') or not item.get('auto_gradable') else
                ' Możesz najpierw krótko uzasadnić wybór (1–3 zdania).')
        return (f'{what}{just} Ostatni wiersz odpowiedzi musi mieć dokładnie postać ({where}):\n{fmt}')
    if t == ESSAY:
        return ('Napisz wypracowanie na podany temat: z wyraźną tezą, uporządkowaną argumentacją opartą na '
                'faktach historycznych, właściwą chronologią i terminologią oraz wnioskami.')
    return 'Odpowiedz zwięźle i konkretnie, tak jak w arkuszu egzaminacyjnym (bez powtarzania polecenia).'


def image_paths(item):
    root = Path(item.get('_root') or '.')
    return [str(root / p) for p in (item.get('images') or [])]


def user_text(item, text_only=False):
    parts = []
    has_images = bool(item.get('images'))
    if has_images and text_only:
        adapted = (item.get('adapted_660_text') or '').strip()
        parts.append(('Opis materiału graficznego (wersja dla osób niewidomych):\n' + adapted) if adapted else NO_IMAGE_NOTE)
    context = (item.get('context') or '').strip()
    if context:
        parts.append('Materiały do zadania:\n' + context)
    # `question` already contains the options / statements text: never append them again.
    parts.append(f"Zadanie {item.get('task', '')} ({item.get('max_points', 1)} pkt).\n" + (item.get('question') or '').strip())
    parts.append(instruction(item))
    return '\n\n'.join(parts)


def build_messages(item, text_only=False, system_prompt=SYSTEM_PROMPT):
    """Chat messages; image parts are placeholders `{'type': 'image'}` in the order of image_paths()."""
    content = []
    if not text_only:
        content += [{'type': 'image'} for _ in image_paths(item)]
    content.append({'type': 'text', 'text': user_text(item, text_only)})
    msgs = []
    if system_prompt:
        msgs.append({'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]})
    msgs.append({'role': 'user', 'content': content})
    return msgs


def render_prompt(processor, item, text_only=False, thinking=False):
    """Prompt string through the model's own chat template (generation prompt included)."""
    return processor.apply_chat_template(build_messages(item, text_only), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=bool(thinking))


# --------------------------------------------------------------------------- targets (training)

_EXAMPLE_HEADER = re.compile(r'^\s*(Przykładow[aey]\s+(odpowied[zź]i?|rozwiązani[ae]|uzasadnieni[ae])|Rozwiązanie|Odpowiedź)\s*:?\s*\n',
                             re.I)


def clean_example_answer(text):
    text = (text or '').strip()
    text = _EXAMPLE_HEADER.sub('', text, count=1)
    # PDF line wraps inside a paragraph -> spaces; keep list/paragraph breaks.
    lines = [ln.rstrip() for ln in text.split('\n')]
    out = []
    for ln in lines:
        if out and out[-1] and ln and not re.match(r'^\s*([•\-–]|\d+[.)]|[A-Z][.)]|Przykładow|Uzasadnienie|Rozstrzygnięcie)', ln) \
                and not out[-1].endswith(('.', ':', '?', '!')):
            out[-1] += ' ' + ln.strip()
        else:
            out.append(ln.strip())
    return '\n'.join(out).strip()


def final_line(item):
    """Gold final answer line, e.g. 'Odpowiedź: 1-F; 2-P; 3-P'."""
    key = item['answer_key']
    if isinstance(key, dict):
        return f'{ANSWER_PREFIX} ' + '; '.join(f'{k}-{v.split(" / ")[0]}' for k, v in key.items())
    return f'{ANSWER_PREFIX} {key}'


def build_target(item):
    """Assistant target for SFT, or None when the item should be skipped."""
    t = item.get('type')
    if is_closed(item):
        key = item['answer_key']
        reasoning = ''
        if not item.get('auto_gradable'):
            # Justification required: the CKE example (it contains the choice + an example justification).
            reasoning = clean_example_answer(item.get('answer'))
        elif t == 'closed_abcd' and isinstance(key, str) and (item.get('options') or {}).get(key):
            reasoning = f"Poprawne dokończenie to {key}. {item['options'][key]}"
        elif t == 'true_false' and isinstance(key, dict):
            st = item.get('statements') or {}
            reasoning = '\n'.join(f"{k}. {'Prawda' if v == 'P' else 'Fałsz'}" + (f' – {st[k]}' if st.get(k) else '')
                                  for k, v in key.items())
        elif isinstance(key, dict):
            reasoning = '\n'.join(f'{k} – {v.split(" / ")[0]}' for k, v in key.items())
        return (reasoning.strip() + '\n' if reasoning.strip() else '') + final_line(item)
    if t in OPEN_TYPES:
        ans = clean_example_answer(item.get('answer'))
        return ans or None
    return None  # essays: no example answer in the dataset


# --------------------------------------------------------------------------- parsing

_THINK_END = ('<channel|>', '</think>')
_ANSWER_LINE = re.compile(r'^[\s>*_#]*odpowied[zź][\s*_]*(?:ko[nń]cowa)?[\s*_]*:[\s*_]*(.*?)[\s*_]*$', re.I)


def strip_thinking(text):
    text = text or ''
    for marker in _THINK_END:
        if marker in text:
            text = text.rsplit(marker, 1)[1]
    return text.replace('<turn|>', '').replace('<eos>', '').strip()


def final_answer(text):
    """Content of the LAST 'Odpowiedź:' line (after any thinking block), or None."""
    lines = strip_thinking(text).split('\n')
    for i in range(len(lines) - 1, -1, -1):
        m = _ANSWER_LINE.match(lines[i])
        if m:
            value = m.group(1).strip()
            if not value:  # 'Odpowiedź:' followed by the answer on the next line(s)
                value = ' '.join(ln.strip() for ln in lines[i + 1:] if ln.strip())
            return value or None
    return None


def normalize(s):
    """Case/whitespace/punctuation/diacritics-insensitive form."""
    s = (s or '').lower().replace('ł', 'l')
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def gold_variants(gold):
    """Accepted normalized strings for a gold value with ' / ' alternatives and [optional] parts."""
    out = set()
    alts = [a.strip() for a in str(gold).split(' / ') if a.strip()]
    first_words = alts[0].split() if alts else []
    expanded = list(alts)
    for a in alts[1:]:
        # 'Pepin Mały / Krótki' -> also 'Pepin Krótki'
        if len(a.split()) == 1 and len(first_words) > 1:
            expanded.append(' '.join(first_words[:-1] + [a]))
    for a in expanded:
        out.add(normalize(re.sub(r'\[[^\]]*\]', ' ', a)))
        out.add(normalize(a.replace('[', ' ').replace(']', ' ')))
    return {v for v in out if v}


def value_matches(pred, gold):
    return bool(pred) and normalize(pred) in gold_variants(gold)


_TF = {'p': 'P', 'prawda': 'P', 'prawdziwe': 'P', 'prawdziwy': 'P', 't': 'P', 'tak': 'P', 'true': 'P',
       'f': 'F', 'falsz': 'F', 'falszywe': 'F', 'falszywy': 'F', 'n': 'F', 'nie': 'F', 'false': 'F'}


def _letter(value):
    m = re.match(r'^\W*([ABCD])(?![a-ząćęłńóśźż])', value.strip(), re.I)
    return m.group(1).upper() if m else None


def _tf(value):
    n = normalize(value).split()
    return _TF.get(n[0]) if n else None


_SEP = r'\s*(?:[-–—:=)]|\.\s*[-–—:]?)\s*'


def parse_pairs(answer, labels):
    """'1-F; 2-P' / 'A – Karol IX; B – Henryk IV' -> {label: value} for the given labels."""
    labs = '|'.join(re.escape(k) for k in sorted(labels, key=len, reverse=True))
    comma = r',\s*(?=(?:fragment\s+|zdanie\s+)?(?:' + labs + r')\s*[-–—:.)=])' if labs else r'(?!)'
    segments = [s.strip() for s in re.split(r'[;\n]|' + comma, answer, flags=re.I) if s.strip()]
    out = {}
    unmatched = []
    norm_labels = sorted(labels, key=lambda k: -len(k))
    for seg in segments:
        hit = None
        for lab in norm_labels:
            if lab in out:
                continue
            m = re.match(r'^\W*(?:fragment|zdanie|opis|element|tekst|punkt)?\s*' + re.escape(lab) + _SEP + r'(.+)$', seg, re.I)
            if not m and len(lab) > 3:
                m = re.match(r'^\W*' + re.escape(lab) + r'\s*[-–—:.]?\s*(.+)$', seg, re.I)
            if m:
                hit = (lab, m.group(1).strip().rstrip('.').strip())
                break
        if hit:
            out[hit[0]] = hit[1]
        else:
            unmatched.append(seg)
    if not out and len(unmatched) == len(labels):
        out = {lab: seg.strip().rstrip('.') for lab, seg in zip(labels, unmatched)}  # positional fallback
    return out


def parse_prediction(item, text):
    """Structured prediction for a closed item: 'C' or {label: value}; None if unparseable."""
    ans = final_answer(text)
    if ans is None:
        return None
    key = item['answer_key']
    t = item['type']
    if not isinstance(key, dict):
        return _letter(ans)
    pairs = parse_pairs(ans, list(key))
    if t == 'true_false':
        if not pairs:  # 'Odpowiedź: F P P'
            toks = [x for x in re.split(r'[\s,;]+', ans) if x]
            if len(toks) == len(key) and all(_tf(x) for x in toks):
                pairs = dict(zip(key, toks))
        pairs = {k: _tf(v) for k, v in pairs.items()}
    elif t == 'closed_abcd':
        if not pairs:
            toks = re.findall(r'\b[ABCD]\b', ans)
            if len(toks) == len(key):
                pairs = dict(zip(key, toks))
        pairs = {k: _letter(v) for k, v in pairs.items()}
    pairs = {k: v for k, v in pairs.items() if v}
    return pairs or None


def closed_correct(item, prediction):
    """(n_correct, n_parts) for a closed item prediction (None = unparseable -> 0 correct)."""
    key = item['answer_key']
    if not isinstance(key, dict):
        return int(prediction is not None and prediction == str(key).strip().upper()[:1]), 1
    prediction = prediction if isinstance(prediction, dict) else {}
    n = 0
    for k, gold in key.items():
        pred = prediction.get(k)
        if pred is None:
            continue
        if item['type'] == 'matching':
            n += value_matches(pred, gold)
        else:
            n += pred.strip().upper() == str(gold).strip().upper()
    return n, len(key)
