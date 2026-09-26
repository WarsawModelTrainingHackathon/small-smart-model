# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf>=1.24", "certifi"]
# ///
"""Build a matura z historii (poziom rozszerzony) dataset from official CKE PDFs.

Downloads (or reuses cached) question papers ("arkusz") and scoring rules
("zasady oceniania") from cke.gov.pl, parses them into one item per scored task,
renders figures for visual tasks and holds out WHOLE exam sessions for dev/test.

    uv run scripts/prepare_matura_historia.py            # download missing + build
    uv run scripts/prepare_matura_historia.py --offline  # build from cached raw/ only

Outputs data/matura-historia/{data.json,manifest.json,images/}.
Parsing is heuristic (PDF layout); see manifest["known_issues"].
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import ssl
import time
import urllib.request

import certifi

import pymupdf

OUT = Path('data/matura-historia')
DPI = 150
UA = 'Mozilla/5.0 (compatible; matura-historia-dataset/1.0; educational research)'
C = 'https://cke.gov.pl/images'
F15 = f'{C}/_EGZAMIN_MATURALNY_OD_2015'
F23 = f'{C}/_EGZAMIN_MATURALNY_OD_2023'
PAGES = {
    '2023': 'https://cke.gov.pl/egzamin-maturalny/egzamin-maturalny-w-formule-2023/arkusze/',
    '2015': 'https://cke.gov.pl/egzamin-maturalny/egzamin-maturalny-w-formule-2015/arkusze/',
    'md23': 'https://cke.gov.pl/egzamin-maturalny/egzamin-maturalny-w-formule-2023/materialy-dodatkowe/',
    'md15': 'https://cke.gov.pl/egzamin-maturalny/egzamin-maturalny-w-formule-2015/materialy-dodatkowe/',
}
# key: (year, session, formula, date, arkusz url, zasady url, adapted-660 docx url or None)
SESSIONS = {
    '2015-maj-f2015': (2015, 'maj', '2015', '2015-05', f'{F15}/Arkusze_egzaminacyjne/2015/formula_od_2015/MHI-R1_1P-152.pdf', f'{F15}/Arkusze_egzaminacyjne/2015/formula_od_2015/odpowiedzi/MHI-R1-N.pdf', None),
    '2016-maj-f2015': (2016, 'maj', '2015', '2016-05', f'{F15}/Arkusze_egzaminacyjne/2016/formula_od_2015/MHI-R1_1P-162.pdf', f'{F15}/Arkusze_egzaminacyjne/2016/formula_od_2015/zasady_oceniania/MHI-R1-N.pdf', None),
    '2017-maj-f2015': (2017, 'maj', '2015', '2017-05', f'{F15}/Arkusze_egzaminacyjne/2017/formula_od_2015/historia/MHI-R1_1P-172.pdf', f'{F15}/Arkusze_egzaminacyjne/2017/formula_od_2015/zasady_oceniania/MHI-R1-N.pdf', None),
    '2018-maj-f2015': (2018, 'maj', '2015', '2018-05', f'{F15}/Arkusze_egzaminacyjne/2018/formula_od_2015/historia/MHI-R1_1P-182.pdf', f'{F15}/Arkusze_egzaminacyjne/2018/formula_od_2015/Zasady_oceniania/MHI-R1_1P-182_zasady_oceniania.pdf', None),
    '2019-maj-f2015': (2019, 'maj', '2015', '2019-05', f'{F15}/Arkusze_egzaminacyjne/2019/formula_od_2015/historia/MHI-R1_1P-192.pdf', f'{F15}/Arkusze_egzaminacyjne/2019/formula_od_2015/Zasady_oceniania/MHI-R1_1P-192_model.pdf', None),
    '2020-kwiecien-probny-f2015': (2020, 'kwiecien-probny', '2015', '2020-04', f'{F15}/Probny/2020/MHI-R1_1P.pdf', f'{F15}/Probny/2020/MHI-R1-zasady.pdf', None),
    '2020-maj-f2015': (2020, 'maj', '2015', '2020-06', f'{F15}/Arkusze_egzaminacyjne/2020/formula_od_2015/historia/MHI-R1_1P-202.pdf', f'{F15}/Arkusze_egzaminacyjne/2020/formula_od_2015/Zasady_oceniania/MHI-PR-202_zasady.pdf', None),
    '2021-marzec-diagnostyczny-f2015': (2021, 'marzec-diagnostyczny', '2015', '2021-03', f'{F15}/Probny/2021/EHIP-R0-100-2103.pdf', f'{F15}/Probny/2021/EHIP-R0-100-2103-zasady.pdf', f'{F15}/Probny/2021/EHIP-R0-660-2103.docx'),
    '2021-maj-f2015': (2021, 'maj', '2015', '2021-05', f'{F15}/Arkusze_egzaminacyjne/2021/Historia/poziom_rozszerzony/EHIP-R0-100-2105.pdf', f'{F15}/Arkusze_egzaminacyjne/2021/Zasady_Oceniania/EHIP-R0-100-2105-zasady.pdf', None),
    '2022-maj-f2015': (2022, 'maj', '2015', '2022-05', f'{F15}/Arkusze_egzaminacyjne/2022/Historia/poziom_rozszerzony/EHIP-R0-100-2205.pdf', f'{F15}/Arkusze_egzaminacyjne/2022/Zasady_oceniania/EHIP-R0-100-2205-zasady.pdf', None),
    '2023-maj-f2015': (2023, 'maj', '2015', '2023-05', f'{F15}/Arkusze_egzaminacyjne/2023/Historia/EHIP-R0-100-2305.pdf', f'{F15}/Arkusze_egzaminacyjne/2023/Historia/EHIP-R0-100-2305-zasady.pdf', None),
    '2022-marzec-pokazowy': (2022, 'marzec-pokazowy', '2023', '2022-03', f'{F23}/materialy_dodatkowe/pokazowe/Historia/MHIP-R0-100-2305.pdf', f'{F23}/materialy_dodatkowe/pokazowe/Historia/MHIP-R0-100-200-300-400-660-700-Q00-2203-zasady.pdf', f'{F23}/materialy_dodatkowe/pokazowe/Historia/MHIP-R0-660-2305.docx'),
    '2022-grudzien-diagnostyczny': (2022, 'grudzien-diagnostyczny', '2023', '2022-12', f'{F23}/materialy_dodatkowe/diagnostyczne_12/historia/MHIP-R0-100-2212.pdf', f'{F23}/materialy_dodatkowe/diagnostyczne_12/historia/MHIP-R0-100-200-300-400-660-700-Q00-Z00-2212-zasady.pdf', f'{F23}/materialy_dodatkowe/diagnostyczne_12/historia/MHIP-R0-660-2212.docx'),
    '2023-maj': (2023, 'maj', '2023', '2023-05', f'{F23}/Arkusze_egzaminacyjne/2023/Historia/MHIP-R0-100-2305.pdf', f'{F23}/Arkusze_egzaminacyjne/2023/Historia/MHIP-R0-100-2305-zasady.pdf', f'{F23}/Arkusze_egzaminacyjne/2023/Historia/MHIP-R0-660-2305.docx'),
    '2024-maj': (2024, 'maj', '2023', '2024-05', f'{F23}/Arkusze_egzaminacyjne/2024/Historia/MHIP-R0-100-A-2405-arkusz.pdf', f'{F23}/Arkusze_egzaminacyjne/2024/Historia/MHIP-R0-100-2405-zasady.pdf', None),
    '2024-grudzien-diagnostyczny': (2024, 'grudzien-diagnostyczny', '2023', '2024-12', f'{F15}/Probny/2024/Historia/MHIP-R0-100-A-2412-arkusz.pdf', f'{F15}/Probny/2024/Historia/MHIP-R0-100-200-300-400-660-Q00-2412-zasady.pdf', f'{F15}/Probny/2024/Historia/MHIP-R0-660-A-2412-arkusz.docx'),
    '2025-maj': (2025, 'maj', '2023', '2025-05', f'{F23}/Arkusze_egzaminacyjne/2025/Historia/MHIP-R0-100-A-2505-arkusz.pdf', f'{F23}/Arkusze_egzaminacyjne/2025/zasady_oceniania/MHIP-R0-100-2505-zasady.pdf', None),
    '2026-styczen-probny': (2026, 'styczen-probny', '2023', '2026-01', f'{F23}/materialy_dodatkowe/probny_egzamin/2026_styczen/Historia/MHIP-R0-100-A-2601-arkusz.pdf', f'{F23}/materialy_dodatkowe/probny_egzamin/2026_styczen/Historia/MHIP-R0-100-2601-zasady.pdf', None),
    '2026-maj': (2026, 'maj', '2023', '2026-05', f'{F23}/Arkusze_egzaminacyjne/2026/Historia/MHIP-R0-100-A-2605-arkusz.pdf', f'{F23}/Arkusze_egzaminacyjne/2026/Historia/MHIP-R0-100-2605-zasady.pdf', None),
}
# Whole sessions are held out. Test = two most recent main (May) sessions of formula 2023;
# dev = the preceding May session; everything else (incl. formula 2015, mock/diagnostic papers) = train.
SPLIT = {'2026-maj': 'test', '2025-maj': 'test', '2024-maj': 'dev'}

HEADER_RE = re.compile(r'^Zadanie\.?\s+(\d+)(?:\.(\d+))?\.?\s*(?:\(\s*(0\s*[–−-][–−\d\s-]*\d)\s*\))?\s*(?:pkt\.?)?\s*$')
PAGE_JUNK = re.compile(r'^(Strona \d+ z \d+|M[A-Z]{2,4}[-_]R\d?[-_ ].*|[ME]HI[A-Z]?[-_].*|MHI_1R|Zasady oceniania rozwiązań zadań|'
                       r'Egzamin maturalny z historii.*|Historia\. Poziom rozszerzony.*|Egzamin maturalny\. Historia.*|'
                       r'Próbny egzamin maturalny z historii.*|.*Test diagnostyczny.*grud.*|Więcej arkuszy.*|'
                       r'Uzyskana liczba pkt|Nr zadania|Maks\. liczba pkt|Wypełnia|egzaminator|WYPEŁNIA EGZAMINATOR|BRUDNOPIS.*|Brudnopis.*)\s*$')
DOTS = re.compile(r'[.…_]{4,}|(?:\. ){4,}')
INSTR = re.compile(r'^(Rozstrzygnij|Podaj|Wyjaśnij|Oceń|Dokończ|Zaznacz|Przyporządkuj|Wymień|Sformułuj|Porównaj|Uzupełnij|Napisz|'
                   r'Uporządkuj|Wskaż|Określ|Scharakteryzuj|Przedstaw|Uzasadnij|Rozpoznaj|Wybierz|Zapisz|Wpisz|Ustal|Nazwij|'
                   r'Na podstawie(?!:)|Korzystając|Odwołując|Zadanie zawiera|Wypisz|Udowodnij|Zidentyfikuj|Wykaż|Przeanalizuj|'
                   r'Dopasuj|Zinterpretuj|Uszereguj|Opisz|Odczytaj|Podkreśl|Oblicz|Wykorzystując|W oparciu|Spośród|'
                   r'Przyporządkuj|Zestaw|Ułóż|Rozpoznaj|Zakreśl|Oznacz|Sprecyzuj)\b')
VISUAL_WORDS = re.compile(r'\b(map[aęyie]|mapk|wykres|diagram|ilustracj|fotografi|zdjęci|plakat|karykatur|rysun|schemat|'
                          r'znacz(ek|ki|ka)|monet|banknot|obraz|rycin|infografik|kadr|drzeworyt|relief|grafik|herb|pieczę|medal|'
                          r'fresk|mozaik|miniatur|rzeźb|portret|widokówk|pocztówk|ulotk|okładk|plan\b|szkic|tablic)', re.I)
SCORING_M = re.compile(r'^(Zasady oceniania|Schemat punktowania|Schemat oceniania|Kryteria oceniania( wypowiedzi( argumentacyjnej)?)?|'
                       r'Zasady przyznawania punktów)\s*:?\s*$', re.I)  # re.I: 'KRYTERIA OCENIANIA WYPOWIEDZI ARGUMENTACYJNEJ'
SOLUTION_M = re.compile(r'^(Rozwiązanie|Rozwiązania|Poprawna odpowiedź|Poprawne odpowiedzi|Prawidłowa odpowiedź|'
                        r'Prawidłowe odpowiedzi|Odpowiedź poprawna|Odpowiedzi poprawne)\b\s*:?\s*(.*)$')  # \b: not 'Rozwiązaniem greckim'
EXAMPLE_M = re.compile(r'^Przykładow\w+(\s+\w+){0,2}\s*:?\s*$|^Przykładow\w+(\s+\w+){0,2}\s*:')
NOTE_M = re.compile(r'^(Uwaga|Uwagi)\b\s*[:.!]?\s*(.*)$')
REQ_M = re.compile(r'^Wymagani[ea] (ogólne|szczegółowe)\s*$')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def download(url, dest):
    if dest.exists():
        return False
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    data = urllib.request.urlopen(req, timeout=120, context=ssl.create_default_context(cafile=certifi.where())).read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    time.sleep(1.0)  # polite, sequential
    return True


def clean(s):
    s = s.replace(' ', ' ').replace('­', '').replace('', '•')
    return re.sub(r'[ \t]+', ' ', s).strip()


def page_lines(doc, start_page=0):
    """Yield content lines (and significant image markers) in reading order with positions."""
    out = []
    for pno in range(start_page, len(doc)):
        pg = doc[pno]
        W, H = pg.rect.width, pg.rect.height
        d = pg.get_text('dict', sort=True)
        items = []
        imgs = [b for b in d['blocks'] if b['type'] == 1]
        big = []
        for b in imgs:
            x0, y0, x1, y1 = b['bbox']
            cx = (x0 + x1) / 2
            # flat wide blocks too: genealogy boxes drawn as images (2025-maj zad5 top box is 136x29 pt)
            if (x1 - x0 >= 40 and y1 - y0 >= 30 or x1 - x0 >= 80 and y1 - y0 >= 18) and 60 < cx < W - 60 and y0 > 45 and y1 < H - 45:
                big.append(pymupdf.Rect(b['bbox']))
        # vector charts/maps: several filled shapes in >=2 saturated colours -> one pseudo-image (union bbox)
        vec = []
        for dr in pg.get_drawings():
            r, f = dr['rect'], dr.get('fill')
            if f is None or r.x0 < 55 or r.x1 > W - 50 or r.y0 < 45 or r.y1 > H - 45 or r.width * r.height < 30:
                continue
            if max(f) - min(f) > 0.25:
                vec.append((tuple(round(c, 2) for c in f), r))
        if len(vec) >= 4 and len({c for c, _ in vec}) >= 2:
            u = pymupdf.Rect(vec[0][1])
            for _, r in vec[1:]:
                u |= r
            if not any(b.intersects(u) for b in big):
                big.append(u)
        # the "Wypełnia egzaminator" score table (formula 2015) and cut-off at BRUDNOPIS
        cut = H
        for b in d['blocks']:
            if b['type'] != 0:
                continue
            for l in b['lines']:
                t = clean(''.join(s['text'] for s in l['spans']))
                if t in ('Wypełnia', 'WYPEŁNIA', 'Nr zadania') and l['bbox'][1] > H * 0.6:
                    cut = min(cut, l['bbox'][1] - 12)
        for b in d['blocks']:
            if b['type'] != 0:
                continue
            for l in b['lines']:
                x0, y0, x1, y1 = l['bbox']
                cx = (x0 + x1) / 2
                if not (60 < cx < W - 58) or y0 < 40 or y0 > H - 55 or y0 >= cut:
                    continue
                t = clean(''.join(s['text'] for s in l['spans']))
                if not t or PAGE_JUNK.match(t):
                    continue
                items.append(dict(page=pno, y0=y0, y1=y1, x0=x0, x1=x1, text=t, img=False))
        for r in big:
            items.append(dict(page=pno, y0=r.y0, y1=r.y1, x0=r.x0, x1=r.x1, text='[OBRAZ]', img=True, rect=[r.x0, r.y0, r.x1, r.y1]))
        # stable reading order: text order from pymupdf sort, images inserted by y
        text_items = [i for i in items if not i['img']]
        for im in (i for i in items if i['img']):
            k = next((n for n, t in enumerate(text_items) if t['y0'] > im['y0'] + 1), len(text_items))
            text_items.insert(k, im)
        out.extend(text_items)
    return out


def strip_answer_lines(lines):
    res = []
    for l in lines:
        if l['img']:
            res.append(l)
            continue
        t = DOTS.sub(' ', l['text'])
        t = clean(t)
        if not t or t in ('•', '–', '-'):
            continue
        if re.fullmatch(r'0(\s*[–−-]\s*\d+)+|[•\s]+', t):  # score-box remnants
            continue
        res.append(dict(l, text=t))
    return res


def join_text(lines):
    """Join physical lines into paragraphs-ish text; keep list-like lines on their own."""
    out = []
    for l in lines:
        t = l['text']
        if l['img']:
            out.append('\n[OBRAZ]\n')
            continue
        if out and not out[-1].endswith('\n'):
            prev = out[-1]
            if prev.endswith('-') and not prev.endswith(' -') and re.match(r'[a-ząćęłńóśźż]', t):
                out[-1] = prev[:-1] + t
                continue
            if re.match(r'^([A-F]\.|\d+\.|•|–|Źródło|Fragment|Temat|Rozstrzygnięcie|Uzasadnienie|P$|F$|Tak$|Nie$)', t) or \
               re.search(r'[.:;!?”"]$', prev) and (t[:1].isupper() or t[:1].isdigit()):
                out.append('\n' + t)
            else:
                out.append(' ' + t)
        else:
            out.append(t)
    s = ''.join(out)
    s = re.sub(r'\n\s*\n+', '\n', s)
    return s.strip()


def split_tasks(lines):
    """Return list of (num, sub, pts, [lines]) segments; sub None for group headers or single tasks."""
    segs, cur = [], None
    for l in lines:
        m = HEADER_RE.match(l['text']) if not l['img'] else None
        if m:
            cur = [int(m.group(1)), int(m.group(2)) if m.group(2) else None, m.group(3), []]
            segs.append(cur)
        elif cur is not None:
            cur[3].append(l)
    return segs


def max_points(p):
    return int(re.findall(r'\d+', p)[-1]) if p else None


def footnote_y(blocks, H):
    """Top of the page-bottom footnote area (legal references to the Rozporządzenie / Dz.U. and the note
    'Zwracamy uwagę, że w liceum ...'), recognised by a line starting with a superscript number in the lower
    half of the page; H if there is none. Everything from there down belongs to the footnotes."""
    ys = []
    for b in blocks:
        if b['type'] != 0:
            continue
        for l in b['lines']:
            sp = [s for s in l['spans'] if s['text'].strip()]
            if len(sp) >= 2 and re.fullmatch(r'\d{1,2}', sp[0]['text'].strip()) and sp[0]['size'] <= 8.5 and \
               max(s['size'] for s in sp[1:]) > sp[0]['size'] + 0.4 and l['bbox'][1] > H * 0.45 and l['bbox'][0] < 80:
                ys.append(l['bbox'][1])
    return min(ys) if ys else H


def parse_zasady(path):
    doc = pymupdf.open(path)
    lines = []
    for pno, pg in enumerate(doc):
        H = pg.rect.height
        blocks = pg.get_text('dict')['blocks']
        cut = footnote_y(blocks, H)
        for b in blocks:  # native order: two-column requirement tables stay intact
            if b['type'] != 0:
                continue
            for l in b['lines']:
                t = clean(''.join(s['text'] for s in l['spans']))
                if not t or PAGE_JUNK.match(t) or l['bbox'][1] > H - 40 or l['bbox'][1] >= cut - 1:
                    continue
                lines.append(t)
    tasks = {}
    cur = None
    for t in lines:
        m = HEADER_RE.match(t)
        if m and m.group(3):
            key = m.group(1) + (('.' + m.group(2)) if m.group(2) else '')
            cur = dict(points=max_points(m.group(3)), lines=[])
            if key in tasks and tasks[key]['points'] == cur['points']:  # repeated header (essay over pages): append
                cur = tasks[key]
            elif key in tasks:  # same number reused for a different task (CKE typo)
                key = f'{key}@{cur["points"]}'
            tasks[key] = cur
        elif cur is not None:
            cur['lines'].append(t)
    for key, t in tasks.items():
        state, buckets = 'req', defaultdict(list)
        for s in t['lines']:
            if SCORING_M.match(s):
                state = 'scoring'
                continue
            m = SOLUTION_M.match(s)
            if m:
                state = 'solution'
                if m.group(2):
                    buckets[state].append(m.group(2))
                continue
            if EXAMPLE_M.match(s):  # keep the label ("Przykładowe uzasadnienie:") inside the answer text
                state = 'solution'
            m = NOTE_M.match(s)
            if m:
                state = 'notes'
                if m.group(2):
                    buckets[state].append(m.group(2))
                continue
            if REQ_M.match(s):
                state = 'req'
                continue
            buckets[state].append(s)
        t['scoring'] = '\n'.join(buckets['scoring']).strip()
        t['answer'] = '\n'.join(buckets['solution']).strip()
        t['notes'] = '\n'.join(buckets['notes']).strip()
        t['raw'] = '\n'.join(t['lines']).strip()
        # essays: scoring/criteria text is usually after requirements; keep whole block too
    return tasks


def classify(q, pts, has_options):
    ql = q.lower()
    if (pts or 0) >= 10 or 'wypracowanie' in ql or 'wybierz jeden z nich' in ql or 'wybierz jeden temat' in ql:
        return 'essay'
    if 'oceń prawdziwość' in ql or re.search(r'zaznacz\s+p[,\s]', ql) or 'jeśli stwierdzenie jest prawdziwe' in ql:
        return 'true_false'
    if re.search(r'przyporządkuj|dopasuj|uzupełnij.*spośród|wybierz (je|ją|go|odpowiedni).*spośród', ql, re.S):
        return 'matching'
    if re.search(r'uporządkuj|ułóż .*chronolog|od najwcześniejszego|od najdawniejszego', ql):
        return 'ordering'
    if has_options and re.search(r'zaznacz|podkreśl|zakreśl|wybierz', ql):
        return 'closed_abcd'
    return 'open_long' if (pts or 0) >= 3 else 'open_short'


def parse_options(q):
    """{'A': text, ...} for one stem, or {'1': {'stem': .., 'A': ..}, '2': {...}} for 'dokończ zdania 1. i 2.'."""
    parts, cur_part, cur = {}, None, None
    single = {}
    for line in q.split('\n'):
        line = line.strip()
        mo = re.match(r'^([A-F])\.\s+(.*)$', line)
        mn = re.match(r'^(\d)\.\s*(\S.*)$', line)
        if mo:
            tgt = parts[cur_part] if cur_part else single
            cur = [mo.group(1)]
            tgt[mo.group(1)] = mo.group(2)
        elif mn and not mo:
            cur_part = mn.group(1)
            parts[cur_part] = {'stem': mn.group(2)}
            cur = None
        elif cur is not None and line and not re.match(r'^(Uzasadnienie|Rozstrzygnięcie)', line):
            tgt = parts[cur_part] if cur_part else single
            tgt[cur[0]] += ' ' + line
        elif cur is None and cur_part and line and len(parts[cur_part]) == 1:
            parts[cur_part]['stem'] += ' ' + line
    def ok(d):
        ks = [k for k in d if k != 'stem']
        return len(ks) >= 2 and ks == [chr(65 + i) for i in range(len(ks))]
    multi = {k: v for k, v in parts.items() if ok(v)}
    if len(multi) >= 2:
        return multi
    if ok(single):
        return single
    return None


def parse_statements(q):
    body = re.split(r'fałszyw\w*\.?', q, maxsplit=1)[-1]
    clean_body = re.sub(r'(?m)^\s*(P|F|P F|Prawda|Fałsz)\s*$\n?', '', body)
    if re.search(r'(?m)^\d\.', clean_body):
        bits = re.split(r'(?m)^(\d)\.\s*', clean_body)
        return {bits[i]: re.sub(r'\s+', ' ', bits[i + 1]).strip() for i in range(1, len(bits) - 1, 2)}
    parts = [p.strip() for p in re.split(r'\s*\bP\s*\n\s*F\b\s*', body) if p.strip()]
    return {str(n): re.sub(r'\s+', ' ', p).strip() for n, p in enumerate(parts, 1)}


def normalize_key(typ, answer, options):
    a = answer.split('Przykładowe uzasadnienie')[0].split('Uzasadnienie')[0]
    if typ == 'closed_abcd':
        head = ' '.join(a.split('\n')[:4])
        head = re.sub(r'\[.*?\]', '', head)
        multi = re.findall(r'(\d)\s*\.?\s*[–-]?\s*\b([A-F])\b', head)
        if isinstance(options, dict) and all(isinstance(v, dict) for v in options.values()):
            multi = [(k, v) for k, v in multi if k in options]
        if len(multi) >= 2:
            return {k: v for k, v in multi}
        letters = re.findall(r'(?<![\w/])([A-F])(?![\w/])', head)
        letters = [x for x in letters if options and x in options]
        if len(set(letters)) == 1:
            return letters[0]
        if 1 < len(letters) <= 4 and len(set(letters)) == len(letters):
            return letters
        return None
    if typ == 'true_false':
        pairs = re.findall(r'(\d)\s*\.?\s*[–-]\s*(P|F|Prawda|Fałsz)\b', a)
        if pairs:
            return {k: v[0] for k, v in pairs}
        m = re.match(r'^\s*([PF](?:\s*[,;]?\s*[PF])+)\s*$', a.split('\n')[0])
        if m:
            return {str(i + 1): v for i, v in enumerate(re.findall('[PF]', m.group(1)))}
        seq = re.findall(r'(?m)^\s*(P|F)\s*$', a)
        return {str(i + 1): v for i, v in enumerate(seq)} or None
    if typ == 'matching':
        pairs = re.findall(r'(?m)^\s*([A-F1-9])\s*\.?\s*[–-]\s*(.+?)\s*$', a)
        if not pairs:
            pairs = re.findall(r'([A-F1-9])\s*\.?\s*[–-]\s*([^,;\n]+)', a)
        if not pairs:
            pairs = re.findall(r'(?m)^\s*(?:(\d)\.\s*)?[^:\n]*:\s*([A-F])\s*$', answer)
            pairs = [(k, v) for k, v in pairs if k]
        if not pairs:
            pairs = re.findall(r'(?m)^\s*([^:\n]{3,80}):\s*(\S.*?)\s*$', a)
        return {k: v.strip().rstrip(',;') for k, v in pairs} or None
    if typ == 'ordering':
        seq = re.findall(r'\b([A-F1-9])\b', a.split('\n')[0])
        return seq or None
    return None


def render_regions(doc, regs, stem, img_dir):
    paths = []
    for n, (pno, r) in enumerate(regs, 1):
        pg = doc[pno]
        W, H = pg.rect.width, pg.rect.height
        # at least the text column (x 45..W-45), wider when the figure or its captions stick out into the margin
        clip = pymupdf.Rect(max(min(45, r[0] - 4), 5), max(r[1], 30), min(max(W - 45, r[2] + 4), W - 5), min(r[3], H - 30))
        data = pg.get_pixmap(dpi=DPI, clip=clip).tobytes('png')
        p = img_dir / f'{stem}-{n}.png'
        if not p.exists() or p.read_bytes() != data:  # rewrite only on change (friendlier to synced folders)
            p.write_bytes(data)
        paths.append(p.relative_to(OUT).as_posix())
    return paths


# a line that starts something new below a figure: next source, next task, the question itself
STOP_BELOW = re.compile(r'^(Źródło\s*\d|Zadanie\b|Tabela\b|Mapa\b|Wykres|Tekst\b|Fragment\b|Materiał|Temat\b)')


def regions(seg_lines, only_with_images):
    """Page crops for an item: around its figures, or the whole source text region when the item is visual
    only by heading keyword. Sibling subtasks sharing the same figures get byte-identical crops (deduplicated by git).

    Figure crops: vertically the union of the figure boxes, plus the title lines directly above (no gap, <= 50 pt) and the
    caption/legend lines that follow the figure without a gap, up to the next source, task header or instruction
    (no half-cut lines, no next source); horizontally the text column, widened to any figure/caption in the margin."""
    by_page = defaultdict(list)
    for l in seg_lines:
        by_page[l['page']].append(l)
    regs = []
    for pno, ls in sorted(by_page.items()):
        imgs = [l for l in ls if l['img']]
        if only_with_images:
            if not imgs:
                continue
            top, bot = min(l['y0'] for l in imgs), max(l['y1'] for l in imgs)
            txt = [l for l in ls if not l['img']]
            above, first = [], top
            for l in sorted((l for l in txt if l['y1'] <= top + 2), key=lambda l: -l['y1']):
                if first - l['y1'] > 20 or top - l['y0'] > 55 or CITATION.search(l['text']):
                    break  # title lines sit right above the figure; stop at a gap or the previous source's citation
                above.append(l)
                first = min(first, l['y0'])
            top = first
            inside = [l for l in txt if l['y0'] >= top - 1 and l['y0'] <= bot]  # incl. captions overlapping the box
            last = max([bot] + [l['y1'] for l in inside])
            below = sorted((l for l in txt if l['y0'] > bot), key=lambda l: l['y0'])
            nxt = None
            for l in below:
                if l['y0'] - last > 16 or l['y1'] - bot > 90 or STOP_BELOW.match(l['text']) or INSTR.match(l['text']) \
                   or QWORD.match(l['text']):
                    nxt = l
                    break
                inside.append(l)
                last = max(last, l['y1'])
            prev = max((l['y1'] for l in txt if l['y1'] <= top), default=0)
            # 4 pt padding, but never into the neighbouring line (no half-cut next heading / previous line)
            bot = min(last + 4, nxt['y0'] - 1) if nxt else last + 4
            xs = imgs + inside + above
            regs.append((pno, [min(l['x0'] for l in xs), max(top - 4, prev + 1), max(l['x1'] for l in xs), bot]))
        else:
            regs.append((pno, [min(l['x0'] for l in ls), min(l['y0'] for l in ls) - 4, max(l['x1'] for l in ls),
                               max(l['y1'] for l in ls) + 4]))
    return regs


CITATION = re.compile(r'\bs\.\s*\d|https?://|www\.|^Na podstawie:|\[dostęp|\.(pl|com|org|net|de|uk)\b|'
                      r'\b(Warszawa|Kraków|Wrocław|Poznań|Łódź|Lublin|Gdańsk|Katowice|Toruń|Londyn|Paryż|Olsztyn|Białystok)\s+\d{4}')
QWORD = re.compile(r'^(Który|Która|Które|Którego|Których|Jaki|Jaka|Jakie|Jakiego|Czy|Kto|Kiedy|Gdzie|Dlaczego|Ile|W jakim|'
                   r'W którym|Do każdego|Każdemu|Każdej|Interpretując|Analizując|Porównując|Dopisz|Objaśnij|Zaproponuj)\b')


def split_context_question(lines):
    """Context = sources before the instruction; question = instruction onward (heuristic).

    Prefer the first instruction-like line after the last bibliographic citation (sources end with one);
    otherwise the first line starting with an instruction verb."""
    txt = [(i, l['text']) for i, l in enumerate(lines) if not l['img']]
    first_instr = next((i for i, t in txt if INSTR.match(t)), None)
    last_cit = max((i for i, t in txt if CITATION.search(t)), default=None)
    if last_cit is not None and (first_instr is None or first_instr < last_cit):
        cand = next((i for i, t in txt if i > last_cit and (INSTR.match(t) or QWORD.match(t))), None)
        if cand is not None:
            return lines[:cand], lines[cand:]
    if first_instr is not None:
        return lines[:first_instr], lines[first_instr:]
    q = next((i for i, t in txt if QWORD.match(t)), None)
    if q is not None:
        return lines[:q], lines[q:]
    return [], lines


def scoring_points(scoring):
    lv = [int(x) for x in re.findall(r'(?m)^\s*(\d+)\s*(?:pkt|p\.|punkt)', scoring)]
    return max(lv) if lv else None


def split_essay(qlines):
    """Split an essay task into topics; attach 'Materiały źródłowe do tematu N' blocks to their topic."""
    mats, cur, head_lines = defaultdict(list), None, []
    for l in qlines:
        m = None if l['img'] else re.match(r'^Materiały źródłowe do tematu\s*(?:nr\s*)?(\d)', l['text'])
        if m:
            cur = int(m.group(1))
        elif cur is None:
            head_lines.append(l)
        else:
            mats[cur].append(l)
    text = join_text(head_lines)
    text = re.split(r'\n(?:WYPRACOWANIE|wypracowanie|Wypracowanie)\b', text)[0]
    parts = re.split(r'\n(?=(?:Temat\s*(?:nr\s*)?)?\d\.\s+\S)', '\n' + text)
    intro = parts[0].strip()
    topics = [t.strip() for t in parts[1:] if len(t.strip()) > 30]
    if len(topics) < 2:  # unnumbered topics: split on instruction-verb lines
        body = text.split('\n', 1)[1] if text.startswith('Zadanie zawiera') else text
        intro = text.split('\n', 1)[0] if text.startswith('Zadanie zawiera') else ''
        topics, curt = [], []
        for line in body.split('\n'):
            if INSTR.match(line) and not line.startswith('W pracy') and curt:
                topics.append(' '.join(curt)); curt = []
            curt.append(line)
        if curt:
            topics.append(' '.join(curt))
        topics = [re.sub(r'^\d\.\s*', '', t).strip() for t in topics]
    else:
        topics = [re.sub(r'^(?:Temat\s*(?:nr\s*)?)?\d\.\s*', '', t).strip() for t in topics]
    return intro, topics, mats


ESSAY_CRIT = re.compile(r'^(Kryteria oceniania|Zasady oceniania|KRYTERIA OCENIANIA|Poziom IV\b)')


def essay_scoring(z, n_topics, formula):
    """Per-topic (scoring, scoring_raw) for an essay from its zasady block.

    Formula 2015: each topic has its own block (requirements + 'Kryteria oceniania' levels + notes), starting at a
    'Temat N' line (2021-03: at 'Wymagania egzaminacyjne'); scoring = the topic's criteria, scoring_raw = its block.
    Formula 2023: per-topic requirements, then ONE criteria table shared by all topics; scoring = shared criteria
    (from 'Zasady oceniania' / 'KRYTERIA OCENIANIA ...' to the end), scoring_raw = whole block."""
    L = z['lines']
    if formula == '2015':
        starts = [i for i, l in enumerate(L) if re.match(r'^Temat\b', l)]
        if len(starts) != n_topics:
            starts = [i for i, l in enumerate(L) if re.match(r'^Wymagania egzaminacyjne', l)]
        if len(starts) != n_topics:
            return None
        out = []
        for k, a in enumerate(starts):
            blk = L[a:starts[k + 1] if k + 1 < n_topics else len(L)]
            c = next((i for i, l in enumerate(blk) if ESSAY_CRIT.match(l)), None)
            if c is None:
                return None
            out.append(('\n'.join(blk[c:]).strip(), '\n'.join(blk).strip()))
        return out
    c = next((i for i, l in enumerate(L) if SCORING_M.match(l) or re.match(r'^A\. NARRACJA', l)), None)
    if c is None:
        return None
    return [('\n'.join(L[c:]).strip(), z['raw'])] * n_topics


def parse_660(path):
    """Adapted paper for blind students (arkusz 660, .docx): figures are replaced by text descriptions ("Opis ...").
    Returns ({(task, max_points)}, {group: text})."""
    import zipfile
    x = zipfile.ZipFile(path).read('word/document.xml').decode('utf-8')
    paras = [clean(re.sub(r'<[^>]+>', '', p)) for p in re.findall(r'<w:p[ >].*?</w:p>', x, re.S)]
    heads, groups, cur = set(), defaultdict(list), None
    for p in paras:
        m = HEADER_RE.match(p)
        if m:
            cur = int(m.group(1))
            if m.group(3):
                heads.add((m.group(1) + ('.' + m.group(2) if m.group(2) else ''), max_points(m.group(3))))
        if cur is not None and p and not re.fullmatch(r'[.…\s]*', p):
            groups[cur].append(re.sub(r'\s*(\.{3,}|…)\s*$', '', p))
    return heads, {k: '\n'.join(v) for k, v in groups.items()}


def build_session(key, spec, raw, img_dir, report):
    year, session, formula, date, a_url, z_url, _ = spec
    doc = pymupdf.open(raw / f'{key}-arkusz.pdf')
    lines = page_lines(doc)
    first = next(i for i, l in enumerate(lines) if HEADER_RE.match(l['text']) and HEADER_RE.match(l['text']).group(1) == '1')
    lines = strip_answer_lines(lines[first:])
    segs = split_tasks(lines)
    zas = parse_zasady(raw / f'{key}-zasady.pdf')
    items, group_ctx = [], {}
    adapted = None
    if spec[6] and (raw / f'{key}-660.docx').exists():
        heads660, groups660 = parse_660(raw / f'{key}-660.docx')
        std = {(f'{n}' + (f'.{sb}' if sb else ''), max_points(p)) for n, sb, p, _ in segs if p}
        if heads660 == std:
            adapted = groups660
        report['adapted_660'][key] = 'used (task numbering/points identical)' if adapted else \
            f'not used: numbering differs ({len(std ^ heads660)} task/point mismatches)'
    prefix = f'{year}-{session}' + ('-f2015' if formula == '2015' else '') + '-R'
    seen, used_z = set(), set()
    W = report['warnings']

    def visuals(item, vis_lines, heading_text, crop_lines=None):
        has_img = any(l['img'] for l in vis_lines)
        kw = bool(re.search(r'\b(Mapa|Wykres|Wykresy|Diagram|Schemat|Infografika|Plan)\b', heading_text))
        item['needs_visual'] = has_img or kw
        item['visual_reason'] = 'image_in_task' if has_img else ('heading_keyword' if kw else None)
        item['images'] = []
        if item['needs_visual']:
            # keyword-only visuals (vector chart/table without a raster image): crop the sources, not the question
            src = vis_lines if has_img or not crop_lines else crop_lines
            item['images'] = render_regions(doc, regions(src, only_with_images=has_img), item['id'], img_dir)

    for num, sub, pts, ls in segs:
        if pts is None and sub is None:  # group header: shared sources
            group_ctx[num] = ls
            continue
        tid = f'{num}' + (f'.{sub}' if sub else '')
        if tid in seen:
            W.append(f'{key}: duplicate header zad {tid}')
            continue
        seen.add(tid)
        mp_ark = max_points(pts)
        z = zas.get(tid)
        if (mp_ark or 0) >= 10 and (z is None or (z['points'] or 0) < 10):  # essay numbered differently in zasady
            cand = [k for k, v in zas.items() if (v['points'] or 0) >= 10]
            if len(cand) == 1:
                W.append(f'{key}: essay is zad {tid} in arkusz but zad {cand[0]} in zasady (matched)')
                z = zas[cand[0]]
                used_z.add(cand[0])
        if z is None:
            W.append(f'{key}: no zasady for zad {tid}')
        used_z.add(tid)
        mp = mp_ark
        if z:
            sp = scoring_points(z['scoring']) if mp_ark and mp_ark < 10 else None
            # arkusz points are authoritative: with them every session sums to 50 (f2015) / 60 (f2023)
            if (sp and sp != mp_ark) or z['points'] != mp_ark:
                W.append(f'{key}: zad {tid} points: arkusz {mp_ark}, zasady header {z["points"]}, '
                         f'zasady scoring levels max {sp}; kept arkusz')
        gl = group_ctx.get(num, []) if sub else []
        local_ctx, qlines = split_context_question(ls)
        ctx_lines = gl + local_ctx
        context, question = join_text(ctx_lines), join_text(qlines)
        options = parse_options(question)
        typ = classify(question, mp, bool(options))
        answer = z['answer'] if z else ''
        base = dict(id=f'{prefix}-zad{tid}', session_key=key, year=year, session=session, formula=formula,
                    exam_date=date, level='rozszerzony', task=tid, group=num, max_points=mp, type=typ)
        src = dict(arkusz_url=a_url, zasady_url=z_url)
        if typ == 'essay':
            intro, topics, mats = split_essay(ctx_lines + qlines)
            if len(topics) < 2:
                W.append(f'{key}: essay {tid} topics not split')
                topics = [question]
            n_decl = re.search(r'Zadanie zawiera (\w+) temat', intro)
            n_decl = {'dwa': 2, 'trzy': 3, 'cztery': 4, 'pięć': 5}.get(n_decl.group(1)) if n_decl else None
            if n_decl != len(topics):
                raise SystemExit(f'{key}: essay {tid} declares {n_decl} topics, parsed {len(topics)}')
            esc = essay_scoring(z, len(topics), formula) if z else None
            if not esc or not all(sc for sc, _ in esc):
                raise SystemExit(f'{key}: essay {tid}: scoring criteria not found per topic')
            for k, t in enumerate(topics, 1):
                item = dict(base, id=f'{base["id"]}-t{k}', topic=k, choice_group=f'{key}-essay',
                            context=join_text(mats.get(k, [])),
                            question=(intro + '\n' + f'Temat {k}. ' + t).strip() if intro else t,
                            answer=answer, answer_key=None, scoring=esc[k - 1][0],
                            scoring_notes=z['notes'], scoring_raw=esc[k - 1][1],
                            auto_gradable=False)
                ml = mats.get(k, [])
                visuals(item, ml, join_text(ml))
                item['source'] = dict(src, pages=sorted({l['page'] + 1 for l in qlines + ml}))
                item['split'] = SPLIT.get(key, 'train')
                items.append(item)
            continue
        item = dict(base, choice_group=None, context=context, question=question)
        if typ == 'closed_abcd':
            item['options'] = options
        if typ in ('closed_abcd', 'true_false', 'matching', 'ordering'):
            item['requires_justification'] = bool(re.search(r'uzasadnij|uzasadnienie', question, re.I))
        if typ == 'true_false':
            item['statements'] = parse_statements(question)
        key_norm = normalize_key(typ, answer, options) if z else None
        item.update(answer=answer, answer_key=key_norm, scoring=z['scoring'] if z else '',
                    scoring_notes=z['notes'] if z else '')
        item['auto_gradable'] = bool(key_norm) and typ in ('closed_abcd', 'true_false', 'matching', 'ordering') and \
            not item.get('requires_justification')
        heads = '\n'.join(l['text'] for l in ctx_lines + qlines if not l['img'] and re.match(r'^(Źródło|Fragment|Mapa|Wykres|Schemat|Tabela|Diagram|Infografika|Plan)', l['text']))
        visuals(item, ctx_lines + qlines, heads, crop_lines=ctx_lines or None)
        if adapted and item['needs_visual'] and 'Opis' in adapted.get(num, ''):
            item['adapted_660_text'] = adapted[num]
        item['source'] = dict(src, pages=sorted({l['page'] + 1 for l in ctx_lines + qlines}))
        item['split'] = SPLIT.get(key, 'train')
        items.append(item)
    report['sessions'][key] = dict(items=len(items), zasady_tasks=len(zas), max_points=session_score_max(items),
                                   arkusz_sha256=sha(raw / f'{key}-arkusz.pdf'), zasady_sha256=sha(raw / f'{key}-zasady.pdf'))
    missing = set(zas) - used_z
    if missing:
        W.append(f'{key}: zasady tasks not found in arkusz: {sorted(missing)}')
    return items


OFFICIAL_MAX = {'2015': 50, '2023': 60}


def session_score_max(items):
    """Max score of one session: items with the same choice_group are alternatives (the examinee writes ONE
    essay topic), so each choice_group counts once (its max over members); other items count individually."""
    groups = defaultdict(int)
    total = 0
    for d in items:
        if d['choice_group']:
            groups[d['choice_group']] = max(groups[d['choice_group']], d['max_points'])
        else:
            total += d['max_points']
    return total + sum(groups.values())


def check_totals(data):
    by = defaultdict(list)
    for d in data:
        by[d['session_key']].append(d)
    bad = []
    for key, its in by.items():
        got, want = session_score_max(its), OFFICIAL_MAX[its[0]['formula']]
        if got != want:
            bad.append(f'{key}: {got} pts (one item per choice_group), official max {want}')
    if bad:
        raise SystemExit('session point totals differ from the official maximum:\n  ' + '\n  '.join(bad))


def main():
    global OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', default='data/matura-historia')
    p.add_argument('--offline', action='store_true', help='do not download; use cached raw/ files only')
    p.add_argument('--only', nargs='*', help='restrict to these session keys (debug)')
    p.add_argument('--dpi', type=int, default=150, help='render resolution for figure crops')
    a = p.parse_args()
    global DPI
    DPI = a.dpi
    OUT = Path(a.output)
    raw, img_dir = OUT / 'raw', OUT / 'images'
    raw.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(exist_ok=True)
    keys = a.only or list(SESSIONS)
    if not a.offline:
        for key in keys:
            spec = SESSIONS[key]
            for kind, url in (('arkusz.pdf', spec[4]), ('zasady.pdf', spec[5]), ('660.docx', spec[6])):
                if url and download(url, raw / f'{key}-{kind}'):
                    print('downloaded', key, kind)
    report = dict(sessions={}, warnings=[], adapted_660={})
    data = []
    for key in keys:
        data.extend(build_session(key, SESSIONS[key], raw, img_dir, report))
    check_totals(data)
    if not a.only:  # drop stale renders
        keep = {Path(p).name for d in data for p in d['images']}
        for f in img_dir.glob('*.png'):
            if f.name not in keep:
                f.unlink()
    splits = {s: [d for d in data if d['split'] == s] for s in ('train', 'dev', 'test')}
    manifest = dict(
        description='Matura z historii (poziom rozszerzony), official CKE papers + zasady oceniania; one item per scored (sub)task, essays split per topic.',
        source_pages=PAGES, user_agent=UA, formulas={'2015': 'formuła 2015 (2015–2023)', '2023': 'formuła 2023 (2023–)'},
        split_policy='Whole sessions held out: test = 2026-maj + 2025-maj (two most recent May sessions, formula 2023); '
                     'dev = 2024-maj; train = all other sessions (formula 2015 May papers, mock/diagnostic/demo papers, 2023-maj).',
        split_sessions={s: sorted({d['session_key'] for d in v}) for s, v in splits.items()},
        counts={s: len(v) for s, v in splits.items()},
        type_counts={s: dict(Counter(d['type'] for d in v)) for s, v in splits.items()},
        needs_visual={s: sum(d['needs_visual'] for d in v) for s, v in splits.items()},
        auto_gradable={s: sum(d['auto_gradable'] for d in v) for s, v in splits.items()},
        points={s: sum(d['max_points'] or 0 for d in v) for s, v in splits.items()},
        session_max_points={k: v['max_points'] for k, v in report['sessions'].items()},
        sessions=report['sessions'], parse_warnings=report['warnings'],
        adapted_660_versions={k: dict(url=v[6], status=report['adapted_660'].get(k)) for k, v in SESSIONS.items() if v[6]},
        adapted_660_items=sum('adapted_660_text' in d for d in data),
        not_available='CKE publishes only main-term (May) papers plus mock/diagnostic/demo papers; June (dodatkowy) and '
                      'August (poprawkowy) historia papers are not on cke.gov.pl. Formula 2015 and 2023 historia exist only at poziom rozszerzony.',
        fields={'choice_group': 'null for normal items; the same string (e.g. "2024-maj-essay") for alternative items of which '
                                'the examinee answers ONE (the essay topics). A session score counts one item per choice_group.',
                'context': 'shared sources for the task group + sources preceding the instruction (text; [OBRAZ] marks a figure)',
                'question': 'instruction and answer scaffold (dotted answer lines removed)',
                'answer': 'raw solution / accepted answers text from zasady oceniania',
                'answer_key': 'normalized key for closed types (letter, list, or {statement/item: value}); null if not parsed',
                'scoring': 'raw scoring rule text (points per criterion) from zasady oceniania',
                'images': 'PNG crops (150 dpi) of the task region(s) containing figures, relative to dataset dir',
                'adapted_660_text': 'only some visual items: text of the same task group from the official adapted paper for blind '
                                    'students (660), where figures are described in words ("Opis ..."); wording may differ slightly'},
        known_issues=[
            'Text order follows PDF layout; tables are linearized cell by cell and can be hard to read (figure crops include them).',
            'context/question split is heuristic (first instruction line after the last source citation); a few items keep a lead sentence in context or question.',
            'needs_visual = raster figure (or multi-colour vector chart, or Mapa/Wykres/Diagram/Schemat heading) inside the task group; flagged for every subtask of a group with a figure, even when a given subtask can be answered from the text alone.',
            'type is heuristic: open_short vs open_long is by points (>=3 -> open_long); closed tasks that also demand a justification have requires_justification=true and auto_gradable=false.',
            'matching keys can be free text with alternatives ("A / B") and optional parts in [brackets]; grade them with normalisation, not exact match.',
            'Essays (wypracowanie) are split one item per topic; answer is empty and the full scoring block (requirements + criteria, raw text) is in scoring_raw.',
            'max_points comes from the arkusz; CKE zasady contain a few inconsistent point headers/levels (listed in parse_warnings). Counting one item per choice_group, every session sums to exactly 50 (formula 2015) or 60 (formula 2023); the build fails otherwise.',
        ],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'data.json').write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps({k: manifest[k] for k in ('counts', 'type_counts', 'needs_visual', 'auto_gradable')}, ensure_ascii=False, indent=1))
    print(len(report['warnings']), 'warnings')
    for w in report['warnings']:
        print(' ', w)


if __name__ == '__main__':
    main()
