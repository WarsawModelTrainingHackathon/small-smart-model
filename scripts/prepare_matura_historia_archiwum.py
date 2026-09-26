# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf>=1.24", "certifi"]
# ///
"""Add archival CKE matura z historii papers to the TRAIN split of data/matura-historia.

Run AFTER scripts/prepare_matura_historia.py (it rewrites data.json/manifest.json with the core sessions):

    uv run scripts/prepare_matura_historia.py --offline
    uv run scripts/prepare_matura_historia_archiwum.py [--archive-dir DIR] [--offline]

Archival PDFs are NOT committed: they are cached in data/matura-historia/raw-archiwum/ (git-ignored), copied from
--archive-dir if present there (files named as in the archive manifest), otherwise downloaded from the CKE source
URLs below; every file is verified against its SHA-256.

Sessions (all -> split 'train'):
  * formula 2015 extras: grudzień 2013 przykładowy, grudzień 2014 próbny, maj 2024 (EHIP, formula 2015 retake).
  * old formula ('stara', do 2014), maj 2015-2020 retake sessions, poziom podstawowy (P, 100 pts) and rozszerzony (R,
    50 pts). Arkusz headers are 'Zadanie N. (k pkt)' with inline sub-tasks 'N.m.'; the zasady use the formula-2015
    'Zadanie N.m. (0–k)' layout. R part II sources ('Źródło A', 'Źródło B', ...) precede the tasks and are shared:
    each task gets the sources named in its 'Na podstawie źródeł X i Y ...' line as context.
A session is kept only if its points (one item per choice_group) equal the official maximum and every arkusz task
matches the zasady (numbering and points); otherwise it is skipped as a whole.

Leak check: every new item whose question or context is a near-duplicate of a dev/test item (difflib ratio >= 0.8 on
normalised text of >= 80 chars, or a shared passage >= 200 chars) is dropped.
"""
import argparse
from collections import Counter, defaultdict
import difflib
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_matura_historia as P  # noqa: E402

CKE = 'https://cke.gov.pl/images/_EGZAMIN_MATURALNY_OD_2015'
# stem in the archive: (year, session, formula, level, arkusz url, arkusz sha256, zasady url, zasady sha256)
ARCHIVE = {
    '2013-grudzien-przykladowy-f2015-R': (2013, 'grudzien-przykladowy', 'f2015', 'R', f'{CKE}/Przykladowe_arkusze/2015/historia_PR/historia_PR_A1.pdf', '76440a5cddbbe78cfebf0262afbbbdf7f867d7bdf1f20d0af86171749965c596', f'{CKE}/Przykladowe_arkusze/2015/historia_PR/historia_model_PR_A1_A2_A3_A4_A7.pdf', 'b3be058ff1c363062f770f2dcc624e187c267fd5311fcdc387fa929ce4f681d8'),
    '2014-grudzien-probny-f2015-R': (2014, 'grudzien-probny', 'f2015', 'R', f'{CKE}/egzamin_probny_2015/historia_pr/A1Historia_PR_arkusz.pdf', 'a1fcfca265c19e78b9f7cc44ab92d28d8410055e9c1d384a3f271b0b8c616614', f'{CKE}/egzamin_probny_2015/historia_pr/A1A2A3A4A7Historia_PR_model_odpowiedzi.pdf', 'def8a547476f0ef7db9641b6661a6c301bc58ae583d2c16a254773dec4ed49c6'),
    '2024-maj-f2015-R': (2024, 'maj', 'f2015', 'R', f'{CKE}/Arkusze_egzaminacyjne/2024/Historia/EHIP-R0-100-A-2405-arkusz.pdf', 'b8a03c1a10b1300febfd51c4146fdfe9c13003f040c0f827f1416ed6ffbce7e4', f'{CKE}/Arkusze_egzaminacyjne/2024/Historia/EHIP-R0-100-2405-zasady.pdf', '607151cbf8862128cde63834f64d3d043a25c83ffc69388f043bd6e9baaaa046'),
    '2010-maj-stara-P': (2010, 'maj', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2010/Historia/historia_pp.pdf', '87919b436d766290f03545982ff7e79f42ab6a0e2d4724fc579f53703f12d71f', f'{CKE}/Arkusze_egzaminacyjne/2010/Historia/historia_klucz_pp.pdf', '610aa187e6d60dd35bdc2a1e4756dacf7ba759af437a143006f16a7e13674eec'),
    '2010-maj-stara-R': (2010, 'maj', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2010/Historia/historia_pr.pdf', '5c90d0d343897ba069b8c16d9c84e57055210f5d3cf21cdaf521c50d3edd8b9d', f'{CKE}/Arkusze_egzaminacyjne/2010/Historia/historia_klucz_pr.pdf', '60249af7e727a134296476f0b85dd7ade2b4f09b51fdcb28d795af9ceec2daa9'),
    '2011-maj-stara-P': (2011, 'maj', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2011/P/historia_pp.pdf', '5896844d4e21b7fc5c3f44d9e08c467f5a4d1117890283675848579e87afb081', f'{CKE}/Arkusze_egzaminacyjne/2011/kryteria/historia_model_pp.pdf', 'e8c60d6d09fe6aaf605fa4d7fe4164c1bdbc0b35bcde1c4c1e691ba52e687c8d'),
    '2011-maj-stara-R': (2011, 'maj', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2011/R/historia_pr.pdf', '6f9990d3e739592a2315584c6d3d300f7e7ddf0a84d56db3a72209b9860e1cb7', f'{CKE}/Arkusze_egzaminacyjne/2011/kryteria/historia_model_pr.pdf', '36b7fdde2ccf4e78ae562c2122dcf03fba913fa283c93c9fa6b26a2363db8ed5'),
    '2012-czerwiec-stara-P': (2012, 'czerwiec', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2012/czerwiec/historia/historia_pp.pdf', '0d4b9ff8ec80d9de364c8fdfbb9160a029f68e15d146f2ba7dfe8fdc3f54441b', f'{CKE}/Arkusze_egzaminacyjne/2012/czerwiec/klucze/historia_06_pp_klucz.pdf', '27bdb33ea21cc4f3da18e967dd49d9a4c814f31d681f494012b30a3c57c2cf87'),
    '2012-czerwiec-stara-R': (2012, 'czerwiec', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2012/czerwiec/historia/historia_pr.pdf', '53a6e76e64d8bca9b978ae1a8c4492549f6b46cb66247b5c3d82ac676bca684a', f'{CKE}/Arkusze_egzaminacyjne/2012/czerwiec/klucze/historia_06_pr_klucz.pdf', 'a6d0e60d02d2d8d84033cc2b9a29ab47d61fffb3d48f767b13432d039832d12e'),
    '2012-maj-stara-P': (2012, 'maj', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2012/maj/hist/historia_pp.pdf', '68e61a3df2c367cd606b2e93d15318c13eeac6c42519d28937898c6039f30a8c', f'{CKE}/Arkusze_egzaminacyjne/2012/maj/klucze/historia_pp_klucz.pdf', 'b7d6798dd35469617586551369edbc303c770788acc542a9be6fc9dc004258b9'),
    '2012-maj-stara-R': (2012, 'maj', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2012/maj/hist/historia_pr.pdf', '7c772241cf5fb23c950d5f72de7971b2f18ce0cfc11852613323dfb8ec541c22', f'{CKE}/Arkusze_egzaminacyjne/2012/maj/klucze/historia_pr_klucz.pdf', '300b46ab3b700354583d8212a76cd49e92d3b5b272b03a1c413bfa9168e22711'),
    '2013-maj-stara-P': (2013, 'maj', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2013/historia_PP.pdf', 'eb8cc83c69152a76ef7a57cb34452eb5a47f528af3a37319257faf358360e372', f'{CKE}/Arkusze_egzaminacyjne/2013/Kryteria-Oceniania/historia_model_PP.pdf', 'fb55224de68a9d3a436182f765f08aae266b01bd0475d78cc9474c62ea236f44'),
    '2013-maj-stara-R': (2013, 'maj', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2013/historia_PR.pdf', '7f91d531878f6eaa8516ede4cee17db5daa5db74e97fdc859600f07e03f51d4d', f'{CKE}/Arkusze_egzaminacyjne/2013/Kryteria-Oceniania/historia_model_PR.pdf', 'cd7306f97af4e96f4ec3ec15fff34d702e48fcce40df6e1cfd2343609d2f5199'),
    '2014-maj-stara-P': (2014, 'maj', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2014/historia_PP_A1.pdf', '549beb92d30de3933144066a206923e5efe7961ae2533124cee793aed563166f', f'{CKE}/Arkusze_egzaminacyjne/2014/odpowiedzi/Historia_PP.pdf', '81df60479dd614d9ff7c26ae6a9d300b68e1137345176222a1eefeec567b89bc'),
    '2014-maj-stara-R': (2014, 'maj', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2014/historia_PR_A1.pdf', 'f2df36c6d6b2c4e82523e05b3f8464cba5d5a8c6043bc4f9a7186367b6c31daf', f'{CKE}/Arkusze_egzaminacyjne/2014/odpowiedzi/Historia_PR.pdf', '0575c2b33f77c8d0dccfcae1b653f036470458b270091eb9a82783f1a45ba822'),
    '2015-maj-stara-P': (2015, 'maj', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2015/formula_do_2014/MHI-P1_1P-152.pdf', 'b86e0474129b0d868b7660316e923eed8b7c62bb93e910ba9d61064e0e19c319', f'{CKE}/Arkusze_egzaminacyjne/2015/formula_do_2014/odpowiedzi/MHI-P1-S.pdf', '3184e5d3299415c76ed8fc405ab04a7a2edcd12eae9b2c55ec83478a2e7a8508'),
    '2015-maj-stara-R': (2015, 'maj', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2015/formula_do_2014/MHI-R1_1P-152.pdf', '548b2316490dbf10d53e49ff733916be8c231220111e50322abb04e6c733be8e', f'{CKE}/Arkusze_egzaminacyjne/2015/formula_do_2014/odpowiedzi/MHI-R1-S.pdf', 'caee9b89cdbc94f8f2908f06602657f0430ed077943129a3e0e0877d60aeaf71'),
    '2016-maj-stara-P': (2016, 'maj', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2016/formula_do_2014/MHI-P1_1P-162.pdf', '7c17a26ed0e8e706e4450a53b5c6b2c6c408d579ecc555a82a0310b8dc0c1f9f', f'{CKE}/Arkusze_egzaminacyjne/2016/formula_do_2014/zasady_oceniania/MHI-P1-S.pdf', '3c1917e952e617d9a063a63f5eb4acd96629d90d7795d489d7df59270493c633'),
    '2016-maj-stara-R': (2016, 'maj', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2016/formula_do_2014/MHI-R1_1P-162.pdf', 'fa33e43c95cd452678269cf513786a402d83b4f7dd653985a822f7435302d32b', f'{CKE}/Arkusze_egzaminacyjne/2016/formula_do_2014/zasady_oceniania/MHI-R1-S.pdf', 'af0bb45f5e64ac2a8c5436306b04d5d02dbcf2f1b17fae9de5f166bad1bf601c'),
    '2017-maj-stara-P': (2017, 'maj', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2017/formula_do_2014/historia/MHI-P1_1P-172.pdf', 'c8733a9c81d86f7da86198e9095b78cf8116d072724efcc92712cbaae6dec396', f'{CKE}/Arkusze_egzaminacyjne/2017/formula_do_2014/zasady_oceniania/MHI-P1-S.pdf', 'b41264792ce62cb27389fc72b2c141f2a5c014317f00a6643732e4ad3434d377'),
    '2017-maj-stara-R': (2017, 'maj', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2017/formula_do_2014/historia/MHI-R1_1P-172.pdf', '8d8f57ff34cf68df6dbabc6f269c1b17b67b8640f9e073b671d1366631bb8783', f'{CKE}/Arkusze_egzaminacyjne/2017/formula_do_2014/zasady_oceniania/MHI-R1-S.pdf', 'b8d0667e5aec6ecf75b026701219a9427a81a917270e6c19a1357b1922fcd788'),
    '2018-maj-stara-P': (2018, 'maj', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2018/formula_do_2014/historia/MHI-P1_1P-182.pdf', '2a572177bb5ad097b3ca9ba56d4da7df6f8ebbaeeed5b6860d0f6d69e023e0aa', f'{CKE}/Arkusze_egzaminacyjne/2018/formula_do_2014/Zasady_ocenienia/MHI-P1_1P-182_zasady_oceniania.pdf', 'ba605818b1fee4b42eff09f0b194450cb2013c99d25f8be74c471124825a4bb2'),
    '2018-maj-stara-R': (2018, 'maj', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2018/formula_do_2014/historia/MHI-R1_1P-182.pdf', '49cb1c9bc15a4cf0169f5438e9e1342ab5627a4af8a85367ef934a02ece0b0b2', f'{CKE}/Arkusze_egzaminacyjne/2018/formula_do_2014/Zasady_ocenienia/MHI-R1_1P-182_zasady_oceniania.pdf', 'c367c3d0a3300783a9c46067132e36e7863b63742ec8e94871a2285c0ed7ca00'),
    '2019-maj-stara-P': (2019, 'maj', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2019/formula_do_2014/historia/MHI-P1_1P-192.pdf', '3ac27915dbb44f17c6ded54dd81c0b4975275e85d172858012603ddbf748fae8', f'{CKE}/Arkusze_egzaminacyjne/2019/formula_do_2014/Zasady_ocenienia/MHI-P1_1P-192_model.pdf', 'e807a16f66e2ccc7ba658c59fe770392cbb74b7f534cd1c12e0f57177003d770'),
    '2019-maj-stara-R': (2019, 'maj', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2019/formula_do_2014/historia/MHI-R1_1P-192.pdf', '746b69f4b50ac911150c52f3b3ea400c548a64ee732b6b1a2e04a1f0fcb0eb04', f'{CKE}/Arkusze_egzaminacyjne/2019/formula_do_2014/Zasady_ocenienia/MHI-R1_1P-192_model.pdf', '9dfb5f6d7e2ab82825fd0a16e0d419196fae85b770d7c55c4dc50a896eadd600'),
    '2020-maj-stara-P': (2020, 'maj', 'stara', 'P', f'{CKE}/Arkusze_egzaminacyjne/2020/formula_do_2014/historia/MHI-P1_1P-202s.pdf', '3cf6c53ddd3c3af4d98a395be9343411acfb70c122d508bb1d85fb34a96e2265', f'{CKE}/Arkusze_egzaminacyjne/2020/formula_do_2014/Zasady_oceniania/MHI-PP-202s_zasady.pdf', '8aec1531767607762d0fb97c32a4fd9a0a4221acf2152175a1a90f425fa6a3e7'),
    '2020-maj-stara-R': (2020, 'maj', 'stara', 'R', f'{CKE}/Arkusze_egzaminacyjne/2020/formula_do_2014/historia/MHI-R1_1R-202s.pdf', '449e9d235c85ea0e0fe1ce1b065a38f46b840750ad86a95347f41ecbbcc7482f', f'{CKE}/Arkusze_egzaminacyjne/2020/formula_do_2014/Zasady_oceniania/MHI-PR-202s_zasady.pdf', '3ac087cd901f5331441b0f3d44613506c31df5db2928bec9fb33d1bbbfe82dc2'),
}
OFFICIAL_MAX = {('f2015', 'R'): 50, ('stara', 'R'): 50, ('stara', 'P'): 100}
LEVEL = {'P': 'podstawowy', 'R': 'rozszerzony'}
EXAM_MONTH = {'maj': '05', 'czerwiec': '06', 'grudzien-przykladowy': '12', 'grudzien-probny': '12'}


class Skip(Exception):
    pass


PARSE_ZASADY = P.parse_zasady


def session_key(stem):
    """Dataset session_key: formula-2015 extras follow the core naming ('2024-maj-f2015'); old formula keeps the level
    ('2015-maj-stara-P'), since P and R are separate papers of one session."""
    return stem[:-2] if stem.endswith('-f2015-R') else stem


def fetch(stem, raw, archive_dir, offline):
    for kind, url, digest in (('arkusz', ARCHIVE[stem][4], ARCHIVE[stem][5]), ('zasady', ARCHIVE[stem][6], ARCHIVE[stem][7])):
        dest = raw / f'{session_key(stem)}-{kind}.pdf'
        if not dest.exists():
            src = archive_dir / f'{stem}-{kind}.pdf' if archive_dir else None
            if src and src.exists():
                shutil.copyfile(src, dest)
            elif offline:
                raise Skip(f'missing {dest} (offline)')
            else:
                P.download(url, dest)
                print('downloaded', dest.name)
        if P.sha(dest) != digest:
            raise Skip(f'{dest.name}: sha256 differs from the archive manifest')


# ---------- old-formula arkusz rewriting ----------

OLD_HEAD = re.compile(r'^Zadanie\s*(\d+)\.?\s*\((\d+)\s*pkt\.?\s*\)$')  # also 'Zadanie17. (1 pkt)', '(3 pkt )'
OLD_SUBHEAD = re.compile(r'^Zadanie\s+(\d+)\.(\d+)\.?\s*\((\d+)\s*pkt\.?\)$')  # 2019 R: 'Zadanie 2.' + 'Zadanie 2.1. (1 pkt)'
GROUP_HEAD = re.compile(r'^Zadanie\s+(\d+)\.$')
SUB_LINE = re.compile(r'^(\d+)\.(\d+)\.?(?:\s+(.*))?$')  # '2.1. Podaj' (also '6.2 Wypisz', 2017 P)
ROMAN_TOPIC = re.compile(r'^Temat\s+(I{1,3}|IV)\b\.?\s*(.*)$')  # 2018 R: 'Temat I' / 'Temat II'
# essay criteria headings of old-formula zasady (2020: 'Kryteria szczegółowe dla poszczególnych poziomów')
# old-formula zasady introduce answers with 'Przykłady poprawnych odpowiedzi' (kept as the answer, label included)
P.EXAMPLE_M = re.compile(P.EXAMPLE_M.pattern + r'|^Przykłady?\s+poprawn\w+\s+odpowiedzi\w*\s*:?\s*$')
P.SOLUTION_M = re.compile(P.SOLUTION_M.pattern.replace('|Prawidłowa odpowiedź|', '|Prawidłowa odpowiedź|Poprawna podpowiedź|'))  # CKE typo, 2012 P
P.ESSAY_CRIT = re.compile(P.ESSAY_CRIT.pattern[:-1] + r'|Kryteri(a|um) szczegółowe\b|Poziom I{1,3}\b)')


def points_table(pdf):
    """Per-task max points from the arkusz score boxes ('Nr zadania 1.1. 1.2. 2.' / 'Maks. liczba pkt 1 1 3')."""
    import pymupdf
    res = {}
    for pg in pymupdf.open(pdf):
        ws = pg.get_text('words')
        rows = {}
        for i, w in enumerate(ws[:-1]):
            nxt = ws[i + 1]
            if (w[4], nxt[4]) == ('Nr', 'zadania'):
                rows['nr'] = (nxt[2], (nxt[1] + nxt[3]) / 2)
            if (w[4], nxt[4]) == ('liczba', 'pkt') and i and ws[i - 1][4] == 'Maks.':
                rows['pkt'] = (nxt[2], (nxt[1] + nxt[3]) / 2)
        if len(rows) < 2:
            continue
        def row(r, pat):
            x, y = rows[r]
            return sorted(((w[0] + w[2]) / 2, w[4]) for w in ws if abs((w[1] + w[3]) / 2 - y) < 4 and w[0] > x + 2
                          and re.fullmatch(pat, w[4]))
        nrs, pts = row('nr', r'\d+(\.\d+)?\.?'), row('pkt', r'\d+')
        for cx, t in nrs:
            near = min(pts, key=lambda p: abs(p[0] - cx), default=None)
            if near and abs(near[0] - cx) < 20:
                res[t.rstrip('.')] = int(near[1])
    return res
JUNK = re.compile(r'^(TEMAT:.*|Poziom (rozszerzony|podstawowy)|CZĘŚĆ\s+I{1,3}\b.*|Część\s+I{1,3}\b.*|Egzamin maturalny z historii.*)$')
SRC_LETTER = re.compile(r'^Źródło\s+([A-L])\b\.?\s*(.*)$')
SRC_INSTR = re.compile(r'^(Na podstawie|Wykorzystując|Korzystając z|Odwołując się do)\s+źród\w+\s+(.*)$')


def letters_of(instr):
    """Source letters named in 'Na podstawie źródeł A i B ...' / 'źródeł A–C' (uppercase single letters only)."""
    t = re.sub(r'([A-L])\s*[–-]\s*([A-L])', lambda m: ' '.join(chr(c) for c in range(ord(m.group(1)), ord(m.group(2)) + 1)), instr)
    t = t.split(' wykonaj')[0]
    return [x for x in re.findall(r'\b([A-L])\b', t)]


def mk(l, text):
    return dict(l, text=text, img=False)


# ---------- 2010-2014 ('Kryteria oceniania odpowiedzi'): sub-parts 'A. (0–1)' in the zasady, 'A. Podaj ...' in the arkusz

SUBPART_Z = re.compile(r'^([A-F])\.\s*\(\s*0\s*[–−-]\s*(\d+)\s*\)$')
SCORE_LINE = re.compile(r'^\d+\s*(p|pkt|punkt\w*)\.?\s*[–-]')
ZJUNK = re.compile(r'^(Kryteria oceniania odpowiedzi.*|Obszar standardów|Opis wymagań|Obszar standardów\s+Opis wymagań|\d{1,2})$')


def bucketize(lines):
    """Same states as the core parse_zasady, plus: a '1 p. – ...' line after the answer starts the scoring rule (these
    keys have no 'Schemat punktowania' heading)."""
    state, b = 'req', defaultdict(list)
    for t in lines:
        if P.SCORING_M.match(t):
            state = 'scoring'
            continue
        m = P.SOLUTION_M.match(t)
        if m:
            state = 'solution'
            if m.group(2):
                b[state].append(m.group(2))
            continue
        if P.EXAMPLE_M.match(t):
            state = 'solution'
        m = P.NOTE_M.match(t)
        if m:
            state = 'notes'
            if m.group(2):
                b[state].append(m.group(2))
            continue
        if P.REQ_M.match(t):
            state = 'req'
            continue
        if SCORE_LINE.match(t) and state in ('solution', 'req'):
            state = 'scoring'
        b[state].append(t)
    return {k: '\n'.join(b[v]).strip() for k, v in (('scoring', 'scoring'), ('answer', 'solution'), ('notes', 'notes'))}


def parse_zasady_old(path):
    """parse_zasady for the 2010-2014 keys: running page headers dropped, each lettered sub-part 'A. (0–1)' of
    'Zadanie N. (0–k)' becomes task 'N.1', 'B.' -> 'N.2', ..."""
    import pymupdf
    lines = []
    for pg in pymupdf.open(path):
        H = pg.rect.height
        for b in pg.get_text('dict')['blocks']:
            if b['type'] != 0:
                continue
            for l in b['lines']:
                t = P.clean(''.join(sp['text'] for sp in l['spans']))
                if not t or P.PAGE_JUNK.match(t) or ZJUNK.match(t) or l['bbox'][1] > H - 40 or l['bbox'][3] < 60:
                    continue
                lines.append(t)
    tasks, cur = {}, None
    for t in lines:
        m = P.HEADER_RE.match(t)
        if m and m.group(3):
            k = m.group(1) + (('.' + m.group(2)) if m.group(2) else '')
            cur = dict(points=P.max_points(m.group(3)), lines=[])
            if k in tasks and tasks[k]['points'] == cur['points']:
                cur = tasks[k]
            elif k in tasks:
                k = f'{k}@{cur["points"]}'
            tasks[k] = cur
        elif cur is not None:
            cur['lines'].append(t)
    out = {}
    for k, t in tasks.items():
        subs, pre = [], []
        for x in t['lines']:
            m = SUBPART_Z.match(x)
            if m and m.group(1) == chr(65 + len(subs)):
                subs.append(dict(points=int(m.group(2)), lines=[]))
            elif subs:
                subs[-1]['lines'].append(x)
            else:
                pre.append(x)
        if subs:
            if sum(sp['points'] for sp in subs) != t['points']:
                raise Skip(f'zasady zad {k}: sub-parts sum {sum(sp["points"] for sp in subs)} != {t["points"]}')
            for n, sp in enumerate(subs, 1):
                out[f'{k}.{n}'] = dict(sp, **bucketize(sp['lines']), raw='\n'.join(pre + sp['lines']).strip())
        else:
            out[k] = dict(t, **bucketize(t['lines']), raw='\n'.join(t['lines']).strip())
    return out


def stara_hook(level, report_w, key, table, lettered=False):
    """lettered=True (2010-2014): sub-tasks are 'A. Podaj ...' lines (first pass: only lines starting with an instruction;
    second pass: any line with the next expected letter)."""
    strict = [True]

    def is_sub(l, n):
        if l['img']:
            return None
        if not lettered:
            ms = SUB_LINE.match(l['text'])
            return (int(ms.group(2)), ms.group(3)) if ms and int(ms.group(1)) == n else None
        ms = re.match(r'^([A-F])\.\s+(\S.*)$', l['text'])
        if not ms:
            return None
        rest = ms.group(2)
        if strict[0] and not (P.INSTR.match(rest) or P.QWORD.match(rest) or rest.startswith(('Na podstawie', 'Wykorzystując'))):
            return None
        return ord(ms.group(1)) - 64, rest

    def pts_of(n, s, head=None):
        """Max points of sub-task n.s: the arkusz score box (or sub-task header), confirmed by the zasady header or its
        scoring levels. Without a score box: the zasady header if the scoring levels agree, else the scoring levels
        (the group total in the arkusz is checked afterwards)."""
        z = zas_[0][f'{n}.{s}']
        t, h, lv = table.get(f'{n}.{s}', head), z['points'], P.scoring_points(z['scoring'])
        if t is None:
            if h is not None and (lv is None or lv == h):
                return h
            if (n, s) in resolved:
                return resolved[(n, s)]
            raise Skip(f'zad {n}.{s}: no score box, zasady header {h} vs scoring levels {lv}')
        if t != h and t != lv:
            raise Skip(f'zad {n}.{s}: score box {t}, zasady header {h}, scoring levels {lv}')
        if t != h:
            report_w.append(f'{key}: zad {n}.{s} points: score box {t}, zasady header {h}, scoring levels {lv}; kept {t}')
        return t

    zas_ = [None]
    resolved = {}

    def resolve(n, k, zsubs):
        """Sub-tasks without a score box whose zasady header and scoring levels disagree: pick the only combination
        that adds up to the arkusz group total k."""
        import itertools
        opts = []
        for s in zsubs:
            z = zas_[0][f'{n}.{s}']
            t, h, lv = table.get(f'{n}.{s}'), z['points'], P.scoring_points(z['scoring'])
            opts.append([t] if t is not None else sorted({x for x in (h, lv) if x is not None}))
        if k is None or all(len(o) == 1 for o in opts):
            return
        ok = [c for c in itertools.product(*opts) if sum(c) == k]
        lvs = tuple(P.scoring_points(zas_[0][f'{n}.{s}']['scoring']) for s in zsubs)
        if len(ok) > 1 and lvs in ok:  # headers swapped (2019 P zad 23): the scoring levels are what graders use
            ok = [lvs]
        if len(ok) != 1:
            raise Skip(f'zad {n}: sub-task points ambiguous {opts} for total {k}')
        for s, v in zip(zsubs, ok[0]):
            resolved[(n, s)] = v
            report_w.append(f'{key}: zad {n}.{s} points: no score box; {v} makes the arkusz total {k}')

    def hook(lines, zas):
        if not lettered:
            return run(lines, zas)
        n0 = len(report_w)
        try:
            return run(lines, zas)
        except Skip:
            del report_w[n0:]
            strict[0] = False
            return run(lines, zas)

    def run(lines, zas):
        zas_[0] = zas
        ls = []
        for l in lines:
            if not l['img'] and re.fullmatch(r'\d{1,2}', l['text']) and l['y0'] < 70:  # page number (2010-2014)
                continue
            m = None if l['img'] else ROMAN_TOPIC.match(l['text'])
            if m:
                l = mk(l, f'Temat {"I II III IV".split().index(m.group(1)) + 1}. {m.group(2)}'.strip())
            if l['img'] or not JUNK.match(l['text']):
                ls.append(l)
        # part II of R: remember where it starts/ends (on the unfiltered lines)
        p2 = next((i for i, l in enumerate(lines) if not l['img'] and re.match(r'^(CZĘŚĆ|Część)\s+II\b(?!I)', l['text'])), None)
        p3 = next((i for i, l in enumerate(lines) if not l['img'] and re.match(r'^(CZĘŚĆ|Część)\s+III\b', l['text'])), None)
        if level == 'R' and (p2 is None or p3 is None):
            raise Skip('R paper without CZĘŚĆ II/III markers')
        if level == 'R':
            for l in lines[p2:p3]:
                l['_p2'] = True  # survives the dict copies below
        # 1. rewrite headers and inline sub-tasks, verifying against the zasady
        out, cur, subs_seen, zsubs, heads, essay_alias = [], None, [], [], [], []

        def close():
            if cur is None:
                return
            n, k = cur
            if zsubs:
                if subs_seen != [s for s in zsubs]:
                    raise Skip(f'zad {n}: sub-tasks in arkusz {subs_seen} != zasady {zsubs}')
                tot = sum(pts_of(n, s) for s in zsubs)
                if k is not None and tot != k:
                    raise Skip(f'zad {n}: arkusz {k} pkt, sub-tasks sum {tot}')
            else:
                if k is None:
                    raise Skip(f'zad {n}: group header without sub-tasks')
                if str(n) not in zas:
                    ess = [z for z, v in zas.items() if (v['points'] or 0) >= 10]
                    if k >= 10 and len(ess) == 1:  # essay numbered differently in the zasady (2017 R: 27 vs 24)
                        essay_alias.append(ess[0])
                        return
                    raise Skip(f'zad {n}: missing in zasady')
                if table.get(str(n), k) != k:
                    raise Skip(f'zad {n}: header {k} pkt, score box {table[str(n)]}')

        for l in ls:
            m = None if l['img'] else GROUP_HEAD.match(l['text'])
            if m and cur is not None and int(m.group(1)) == cur[0] + 1:  # explicit group header (2019 R)
                close()
                n = int(m.group(1))
                cur, subs_seen = (n, None), []
                zsubs = sorted((int(z.split('.')[1]) for z in zas if re.fullmatch(rf'{n}\.\d+', z)))
                heads.append(n)
                out.append(mk(l, f'Zadanie {n}.'))
                out[-1]['_head'] = n
                continue
            m = None if l['img'] or cur is None else OLD_SUBHEAD.match(l['text'])
            if m and int(m.group(1)) == cur[0] and cur[1] is None and int(m.group(2)) in zsubs:
                s_ = int(m.group(2))
                subs_seen.append(s_)
                out.append(mk(l, f'Zadanie {cur[0]}.{s_}. (0–{pts_of(cur[0], s_, int(m.group(3)))})'))
                continue
            m = None if l['img'] else OLD_HEAD.match(l['text'])
            if m:
                close()
                n, k = int(m.group(1)), int(m.group(2))
                cur, subs_seen = (n, k), []
                zsubs = sorted((int(z.split('.')[1]) for z in zas if re.fullmatch(rf'{n}\.\d+', z)))
                if len(zsubs) == 1 and str(n) not in zas and f'{n}.1' not in table:  # 2019 R: zasady 'Zadanie 8.2.' for zad 8
                    zas[str(n)] = zas.pop(f'{n}.{zsubs[0]}')
                    report_w.append(f'{key}: zasady zad {n}.{zsubs[0]} is arkusz zad {n} (single sub-task number; matched)')
                    zsubs = []
                resolve(n, k, zsubs)
                heads.append(n)
                out.append(mk(l, f'Zadanie {n}.' if zsubs else f'Zadanie {n}. (0–{k})'))
                out[-1]['_head'] = n
                continue
            ms = None if cur is None else is_sub(l, cur[0])
            if ms and ms[0] in zsubs and ms[0] not in subs_seen and (not lettered or ms[0] == len(subs_seen) + 1):
                s = ms[0]
                subs_seen.append(s)
                out.append(mk(l, f'Zadanie {cur[0]}.{s}. (0–{pts_of(cur[0], s)})'))
                if ms[1]:
                    out.append(mk(l, ms[1]))
                continue
            out.append(l)
        close()
        # essay without the 'Zadanie zawiera N tematy' line (2019 R): add the standard instruction (topics counted)
        for i, l in enumerate(out):
            m = None if l['img'] else re.match(r'^Zadanie (\d+)\. \(0–(\d+)\)$', l['text'])
            if m and int(m.group(2)) >= 10 and not any(x['text'].startswith('Zadanie zawiera') for x in out[i + 1:i + 3]):
                nt = len({x['text'][:8] for x in out[i + 1:] if not x['img'] and re.match(r'^Temat \d\.', x['text'])})
                word = {2: 'dwa', 3: 'trzy'}.get(nt)
                if word:
                    out.insert(i + 1, mk(l, f'Zadanie zawiera {word} tematy. Wybierz jeden z nich do opracowania.'))
                    report_w.append(f'{key}: zad {m.group(1)}: essay has no topic-count line; added the standard one ({nt} topics)')
                break
        if heads != list(range(1, len(heads) + 1)):
            raise Skip(f'task numbering not consecutive: {heads}')
        extra = {z.split('.')[0].split('@')[0] for z in zas if z not in essay_alias} - {str(h) for h in heads}
        if extra:
            raise Skip(f'zasady tasks not in arkusz: {sorted(extra)}')
        if level != 'R':
            return out
        # 2. part II: move shared sources to each task that uses them
        srcs = {}
        res_before, part2, res_after = [], [], []
        for l in out:
            (part2 if l.get('_p2') else (res_before if not part2 else res_after)).append(l)
        new2, cur_src, pending_instr, new_src, prev_letters = [], None, None, [], []
        tasks = []  # [head_line, instr_line, letters, [task lines]]
        for l in part2:
            ms = None if l['img'] else SRC_LETTER.match(l['text'])
            mi = None if l['img'] else SRC_INSTR.match(l['text'])
            if ms:
                cur_src = ms.group(1)
                srcs[cur_src] = [l]
                new_src.append(cur_src)
                continue
            if mi and letters_of(l['text']):
                pending_instr, cur_src = l, None
                continue
            if l.get('_head'):
                if pending_instr is not None:
                    letters = letters_of(pending_instr['text'])
                elif new_src:
                    letters = list(new_src)
                else:
                    letters = list(prev_letters)
                if not letters or any(x not in srcs for x in letters):
                    raise Skip(f'part II zad {l["_head"]}: cannot resolve sources {letters}')
                tasks.append([l, pending_instr, letters, []])
                prev_letters, pending_instr, new_src, cur_src = letters, None, [], None
                continue
            if cur_src is not None:
                srcs[cur_src].append(l)
            elif tasks:
                tasks[-1][3].append(l)
            elif not l['img']:
                report_w.append(f'{key}: part II text before first source dropped: {l["text"][:60]}')
        for head, instr, letters, body in tasks:
            new2.append(head)
            # the task's own lines may start with sub-task headers; sources go into the (group) context first
            for x in letters:
                new2.extend(srcs[x])
            if instr is not None:
                new2.append(instr)
            new2.extend(body)
        return res_before + new2 + res_after
    return hook


MATERIAL = re.compile(r'^Materiały?\s+do\s+zadania\s+(\d+)\.?$')


def material_hook(report_w, key):
    """Formula-2015 extras (2013-12, 2014-12) print 'Materiał(y) do zadania N.' BEFORE the 'Zadanie N.' header; move
    that block after the header so it becomes the task's context instead of the previous task's question."""
    def hook(lines, zas):
        out, buf, want = [], [], None
        for l in lines:
            m = None if l['img'] else MATERIAL.match(l['text'])
            if m:
                if buf:
                    raise Skip(f'material for zad {want} not followed by its header')
                want, buf = m.group(1), [l]
                continue
            h = None if l['img'] else P.HEADER_RE.match(l['text'])
            if h and buf:
                if h.group(1) != want:
                    raise Skip(f'material for zad {want} followed by header zad {h.group(1)}')
                out.append(l)
                out.extend(buf)
                buf, want = [], None
                continue
            (buf if buf else out).append(l)
        if buf:
            raise Skip(f'material for zad {want} without header')
        return out
    return hook


def fix_key(d):
    """Key formats the core normaliser misses: true/false as '1. P' lines; matching as 'A. answer' lines."""
    for f in ('answer', 'scoring', 'scoring_notes', 'scoring_raw'):  # section headings of the zasady ('Część II')
        if d.get(f):
            d[f] = re.sub(r'(\n\s*(Część|CZĘŚĆ)\s+I{1,3}\.?\s*)+$', '', d[f])
    if d['type'] == 'ordering':  # the core normaliser reads only the first answer line; rebuild from 'A. ... n' pairs
        pairs = re.findall(r'(?m)^\s*([A-F])\.[^\n]*\n\s*(\d)\s*$', d['answer'])
        opts = re.findall(r'(?m)^\s*([A-F])\.\s', d['question'])
        if pairs and sorted(k for k, _ in pairs) == sorted(opts) and sorted(int(v) for _, v in pairs) == list(range(1, len(opts) + 1)):
            d['answer_key'] = [k for k, _ in sorted(pairs, key=lambda p: int(p[1]))]
        else:
            d['answer_key'] = None
        d['auto_gradable'] = d['answer_key'] is not None and not d.get('requires_justification')
        return
    if d['answer_key'] is not None or d['type'] not in ('true_false', 'matching'):
        return
    a = d['answer'].split('Przykładowe uzasadnienie')[0].split('Uzasadnienie')[0].strip()
    ls = [x.strip() for x in a.split('\n') if x.strip()]
    if d['type'] == 'true_false':
        pairs = [re.fullmatch(r'(\d)\.?\s*(P|F|prawda|fałsz)', x, re.I) for x in ls]
        if ls and all(pairs):
            d['answer_key'] = {m.group(1): m.group(2)[0].upper() for m in pairs}
    else:
        pairs = [re.fullmatch(r'([A-F1-9])\.\s+(\S.*)', x) for x in ls]
        if len(ls) >= 2 and all(pairs):
            d['answer_key'] = {m.group(1): m.group(2).rstrip(',;') for m in pairs}
    if d['answer_key'] is not None:
        d['auto_gradable'] = not d.get('requires_justification')


# ---------- leak check ----------

def norm(s):
    s = (s or '').lower().replace('[obraz]', ' ')
    s = re.sub(r'[^0-9a-ząćęłńóśźżäöüéè]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def leak_check(new, held):
    """Drop new items that near-duplicate any dev/test item. Returns (kept, dropped[(id, held_id, why)])."""
    H = []
    for h in held:
        H.append((h['id'], norm(h['question']), norm(h['context']), norm(h['context'] + ' ' + h['question'])))
    SH = 60
    shingles = defaultdict(set)
    for n, (_, _, _, full) in enumerate(H):
        for i in range(0, max(1, len(full) - SH + 1), 5):
            shingles[full[i:i + SH]].add(n)
    kept, dropped = [], []
    for d in new:
        q, c = norm(d['question']), norm(d['context'])
        full = norm(d['context'] + ' ' + d['question'])
        hit = None
        cands = set()
        for i in range(0, max(1, len(full) - SH + 1)):
            cands |= shingles.get(full[i:i + SH], set())
        for n in cands:
            sm = difflib.SequenceMatcher(None, full, H[n][3], autojunk=False)
            m = sm.find_longest_match(0, len(full), 0, len(H[n][3]))
            if m.size >= 200:
                hit = (H[n][0], f'shared passage {m.size} chars')
                break
        if not hit:
            for hid, hq, hc, _ in H:
                for a, b, what in ((q, hq, 'question'), (c, hc, 'context')):
                    if len(a) >= 80 and len(b) >= 80 and difflib.SequenceMatcher(None, a, b).quick_ratio() >= 0.8:
                        r = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
                        if r >= 0.8:
                            hit = (hid, f'{what} ratio {r:.2f}')
                            break
                if hit:
                    break
        if hit:
            dropped.append((d['id'], *hit))
        else:
            kept.append(d)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', default='data/matura-historia')
    ap.add_argument('--archive-dir', type=Path, help='local copy of the archive raw/ folder (files <stem>-arkusz.pdf)')
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--only', nargs='*', help='restrict to these archive stems (debug; does not write data.json)')
    a = ap.parse_args()
    out = Path(a.output)
    P.OUT = out
    raw, img_dir = out / 'raw-archiwum', out / 'images'
    raw.mkdir(parents=True, exist_ok=True)
    data = json.loads((out / 'data.json').read_text(encoding='utf-8'))
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    arch_keys = {session_key(s) for s in ARCHIVE}
    data['train'] = [d for d in data['train'] if d['session_key'] not in arch_keys]  # idempotent re-run
    report = dict(sessions={}, warnings=[], adapted_660={})
    new, skipped, status, no_answer = [], {}, {}, []
    for stem in a.only or ARCHIVE:
        year, session, formula, level = ARCHIVE[stem][:4]
        key = session_key(stem)
        spec = (year, session, '2015', f'{year}-{EXAM_MONTH[session]}', ARCHIVE[stem][4], ARCHIVE[stem][6], None)
        w0 = len(report['warnings'])
        try:
            fetch(stem, raw, a.archive_dir, a.offline)
            if formula == 'f2015':
                items = P.build_session(key, spec, raw, img_dir, report, line_hook=material_hook(report['warnings'], key))
            else:
                old = year < 2015
                P.parse_zasady = parse_zasady_old if old else PARSE_ZASADY
                try:
                    items = P.build_session(key, spec, raw, img_dir, report, prefix=key, level=LEVEL[level],
                                            line_hook=stara_hook(level, report['warnings'], key,
                                                                 points_table(raw / f'{key}-arkusz.pdf'), lettered=old))
                finally:
                    P.parse_zasady = PARSE_ZASADY
            got, want = P.session_score_max(items), OFFICIAL_MAX[(formula, level)]  # checked before dropping empty answers
            if got != want:
                raise Skip(f'{got} pts (one item per choice_group), official max {want}')
            bad = [w for w in report['warnings'][w0:] if 'no zasady' in w or 'not found in arkusz' in w or 'duplicate header' in w]
            if bad:
                raise Skip('; '.join(bad))
        except (Skip, SystemExit, StopIteration) as e:
            skipped[key] = str(e) or type(e).__name__
            report['sessions'].pop(key, None)
            print('SKIP', key, '-', skipped[key])
            continue
        empty = [d['id'] for d in items if d['type'] != 'essay' and not d['answer'].strip()]
        if empty:  # the key text did not land in 'answer' (answer inside the scoring table etc.): not shipped
            report['warnings'].append(f'{key}: dropped {len(empty)} items without a parsed answer: {empty}')
            no_answer.extend(empty)
            items = [d for d in items if d['id'] not in empty]
        for d in items:
            fix_key(d)
            d['formula'] = 'stara' if formula == 'stara' else '2015'
            d['level'] = LEVEL[level]
            d['split'] = 'train'
        status[key] = dict(report['sessions'][key], formula=items[0]['formula'], level=LEVEL[level], official_max=want,
                           arkusz_url=ARCHIVE[stem][4], zasady_url=ARCHIVE[stem][6])
        print(f'{key}: {len(items)} items, {got} pts')
        new.extend(items)
    held = data['dev'] + data['test']
    kept, dropped = leak_check(new, held)
    # a session that mostly re-uses held-out sources (2024-maj-f2015 = same exam day as dev 2024-maj) is dropped whole:
    # its remaining items share themes/sources with dev even where the text check does not fire
    n_sess = Counter(d['session_key'] for d in new)
    sk = {d['id']: d['session_key'] for d in new}
    n_leak = Counter(sk[i] for i, _, _ in dropped)
    for k, n in n_leak.items():
        if n >= 0.5 * n_sess[k]:
            rest = [d for d in kept if d['session_key'] == k]
            dropped += [(d['id'], None, f'session {k}: {n}/{n_sess[k]} items leak, whole session dropped') for d in rest]
            kept = [d for d in kept if d['session_key'] != k]
            skipped[k] = f'leak check: {n}/{n_sess[k]} items near-duplicate dev/test items (same sources as dev 2024-maj); whole session dropped'
            status.pop(k, None)
    for d in dropped:
        print('LEAK drop', *d)
    if a.only:
        return
    # drop crops of leaked items (unless shared with a kept item)
    keep_imgs = {p for d in data['train'] + data['dev'] + data['test'] + kept for p in d['images']}
    for f in img_dir.glob('*.png'):  # stale crops of archival sessions (leaked, dropped or renamed items)
        if f.name.startswith(tuple(f'{k}-' for k in arch_keys)) and f'images/{f.name}' not in keep_imgs:
            f.unlink()
    data['train'] = data['train'] + kept
    splits = data
    (out / 'data.json').write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    by_era = lambda d: 'formula 2015 extras' if d['formula'] == '2015' else f'stara {d["year"]}'  # noqa: E731
    manifest['archival'] = dict(
        description='Archival CKE papers added to TRAIN only by scripts/prepare_matura_historia_archiwum.py. PDFs are not '
                    'committed (cache: raw-archiwum/, git-ignored); URLs and SHA-256 per session below.',
        sessions=status, skipped=skipped, parse_warnings=report['warnings'][:],
        items=len(kept), points=sum(d['max_points'] for d in kept), dropped_no_answer=no_answer,
        type_counts=dict(Counter(d['type'] for d in kept)),
        level_counts=dict(Counter(d['level'] for d in kept)),
        formula_counts=dict(Counter(d['formula'] for d in kept)),
        needs_visual=sum(d['needs_visual'] for d in kept), auto_gradable=sum(d['auto_gradable'] for d in kept),
        leak_check=dict(rule='drop a new item if its normalised question or context has difflib ratio >= 0.8 with a dev/test '
                             'item (texts >= 80 chars), or it shares a passage >= 200 chars with a dev/test item',
                        checked=len(new), dropped=len(dropped), dropped_items=[dict(id=i, held_out=h, why=w) for i, h, w in dropped]),
    )
    tr = splits['train']
    manifest['split_sessions']['train'] = sorted({d['session_key'] for d in tr})
    manifest['counts']['train'] = len(tr)
    manifest['type_counts']['train'] = dict(Counter(d['type'] for d in tr))
    manifest['needs_visual']['train'] = sum(d['needs_visual'] for d in tr)
    manifest['auto_gradable']['train'] = sum(d['auto_gradable'] for d in tr)
    manifest['points']['train'] = sum(d['max_points'] or 0 for d in tr)
    for k, v in status.items():
        manifest['session_max_points'][k] = v['max_points']
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps({k: manifest['archival'][k] for k in ('items', 'points', 'type_counts', 'level_counts', 'formula_counts')},
                     ensure_ascii=False))
    print('skipped:', json.dumps(skipped, ensure_ascii=False, indent=1))
    print('leak check: checked', len(new), 'dropped', len(dropped))


if __name__ == '__main__':
    main()
