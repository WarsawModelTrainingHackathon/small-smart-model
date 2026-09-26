# /// script
# requires-python = ">=3.11"
# ///
"""Leak check of data/matura-historia: every TRAIN item (core + archival) against every dev/test item.

    uv run scripts/check_matura_historia_leaks.py [--data data/matura-historia/data.json] [--near]

A train item leaks if its normalised question or context has difflib ratio >= 0.8 with the question or context of a
dev/test item (both texts >= 80 chars), or if its context+question shares >= 200 chars of source text with a dev/test
item's context+question: one contiguous passage, or the sum of common runs of >= 30 chars between a context (the
sources) and the other item's context+question (so one excerpt quoted with different elisions '[…]' counts, but shared
instruction boilerplate in two questions does not). Exit code 1 if any leak is found. --near also lists near misses (ratio >= 0.6 or
>= 100 shared chars) for manual review. Prefilters (speed only): the character ratio is computed only for pairs whose
word multisets overlap >= 0.4 (a character ratio >= 0.8 needs far more), shared text only for pairs with >= 30 common
20-char shingles.
"""
import argparse
from collections import Counter, defaultdict
import difflib
import json
from pathlib import Path
import re
import sys


def norm(s):
    s = (s or '').lower().replace('[obraz]', ' ')
    s = re.sub(r'[^0-9a-ząćęłńóśźżäöüéè]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def check(train, held, ratio=0.8, passage=200, min_len=80, sh=20, run=30):
    H = [(h['id'], norm(h['question']), norm(h['context']), norm(h['context'] + ' ' + h['question'])) for h in held]
    index = defaultdict(set)
    for n, (*_, full) in enumerate(H):
        for i in range(0, max(1, len(full) - sh + 1)):
            index[full[i:i + sh]].add(n)
    M = []  # one matcher per held-out text (difflib caches the analysis of seq2)
    for hid, hq, hc, _ in H:
        for b, what in ((hq, 'question'), (hc, 'context')):
            if len(b) >= min_len:
                sm = difflib.SequenceMatcher(None, autojunk=False)
                sm.set_seq2(b)
                M.append((hid, len(b), what, sm, Counter(b.split())))
    out = []
    for d in train:
        q, c, full = norm(d['question']), norm(d['context']), norm(d['context'] + ' ' + d['question'])
        best = None  # (score kind, value, held id)
        hits = Counter()
        for x in {full[i:i + sh] for i in range(0, max(1, len(full) - sh + 1))}:
            hits.update(index.get(x, ()))
        for n in sorted(n for n, k in hits.items() if k >= run):
            sm = difflib.SequenceMatcher(None, full, H[n][3], autojunk=False)
            size = sm.find_longest_match(0, len(full), 0, len(H[n][3])).size  # one contiguous passage
            # source text quoted with different elisions: sum of common runs, where one side is a context (the
            # sources); question-vs-question overlap is instruction boilerplate ('Rozstrzygnij, czy ... Odpowiedź
            # uzasadnij', essay instructions) and is not counted
            for a, b in ((c, H[n][3]), (full, H[n][2])):
                if a and b:
                    size = max(size, sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False)
                                         .get_matching_blocks() if m.size >= run))
            if size >= passage // 2 and (best is None or best[0] != 'shared text' or size > best[1]):
                best = ('shared text', size, H[n][0])
        for a, what_a in ((q, 'question'), (c, 'context')):
            if len(a) < min_len:
                continue
            wa = Counter(a.split())
            na = sum(wa.values())
            for hid, lb, what_b, sm, wb in M:
                if 2 * min(len(a), lb) / (len(a) + lb) < 0.6:  # = real_quick_ratio
                    continue
                # word-multiset overlap: character quick_ratio hardly filters same-language text
                if 2 * sum((wa & wb).values()) / (na + sum(wb.values())) < 0.4:
                    continue
                sm.set_seq1(a)
                if sm.quick_ratio() >= 0.6:
                    r = sm.ratio()
                    what = what_a if what_a == what_b else f'{what_a}~{what_b}'
                    if r >= 0.6 and (best is None or (best[0] != 'shared text' and r > best[1])):
                        best = (f'{what} ratio', round(r, 3), hid)
        if best:
            leak = best[1] >= passage if best[0] == 'shared text' else best[1] >= ratio
            out.append(dict(id=d['id'], session_key=d['session_key'], held_out=best[2], kind=best[0], value=best[1], leak=leak))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=Path('data/matura-historia/data.json'))
    ap.add_argument('--near', action='store_true', help='also print near misses')
    a = ap.parse_args()
    data = json.loads(a.data.read_text(encoding='utf-8'))
    res = check(data['train'], data['dev'] + data['test'])
    leaks = [r for r in res if r['leak']]
    for r in (res if a.near else leaks):
        print(('LEAK ' if r['leak'] else 'near ') + json.dumps(r, ensure_ascii=False))
    print(f'train items checked: {len(data["train"])}, held-out items: {len(data["dev"]) + len(data["test"])}, '
          f'leaks: {len(leaks)}, near misses: {len(res) - len(leaks)}')
    sys.exit(1 if leaks else 0)


if __name__ == '__main__':
    main()
