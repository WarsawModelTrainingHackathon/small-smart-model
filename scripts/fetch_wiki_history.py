#!/usr/bin/env python3
"""Grow the Polish-Wikipedia history slice (CC BY-SA). Resume-safe, appends chunks.

  python3 scripts/fetch_wiki_history.py --out harness/wiki/minicorpus.jsonl --max-pages 120
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_wiki_minicorpus as fw
import wiki_bm25

EXTRA_TITLES = [
    'Historia Polski', 'Rozbiory Polski', 'Konstytucja 3 maja', 'Insurekcja kościuszkowska',
    'Księstwo Warszawskie', 'Kongres wiedeński', 'Powstanie listopadowe', 'Wiosna Ludów',
    'Powstanie styczniowe', 'I wojna światowa', 'Odzyskanie niepodległości przez Polskę',
    'Wojna polsko-bolszewicka', 'Bitwa warszawska', 'II Rzeczpospolita', 'Przewrót majowy',
    'Kampania wrześniowa', 'Polskie Państwo Podziemne', 'Armia Krajowa', 'Powstanie warszawskie',
    'Zbrodnia katyńska', 'Holokaust w Polsce', 'PRL', 'Odwilż gomułkowska', 'Grudzień 1970',
    'NSZZ Solidarność', 'Okrągły Stół', 'III Rzeczpospolita', 'NATO', 'Unia Europejska',
    'Mieszko I', 'Chrzest Polski', 'Kazimierz III Wielki', 'Unia lubelska', 'Jan III Sobieski',
    'Bitwa pod Wiedniem', 'Stanisław August Poniatowski', 'Tadeusz Kościuszko',
    'Adam Mickiewicz', 'Józef Bem', 'Romuald Traugutt', 'Roman Dmowski', 'Ignacy Jan Paderewski',
    'Władysław Sikorski', 'Stefan Wyszyński', 'Lech Wałęsa', 'Wojciech Jaruzelski',
    'Aleksander Macedoński', 'Cesarstwo rzymskie', 'Rewolucja francuska', 'Napoleon Bonaparte',
    'Wiosna Ludów', 'Zjednoczenie Niemiec', 'I wojna światowa', 'Traktat wersalski',
    'Rewolucja październikowa', 'Adolf Hitler', 'Związek Socjalistycznych Republik Radzieckich',
    'Zimna wojna', 'Kryzys kubański', 'Jesień Ludów',
]

CATEGORIES = [
    'Kategoria:Władcy Polski',
    'Kategoria:Powstania polskie',
    'Kategoria:Bitwy Polski',
    'Kategoria:II wojna światowa',
]


def category_titles(cat, limit=40):
    data = fw.api({
        'action': 'query', 'list': 'categorymembers', 'cmtitle': cat,
        'cmnamespace': 0, 'cmlimit': str(limit),
    })
    return [x['title'] for x in data.get('query', {}).get('categorymembers', [])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'harness' / 'wiki' / 'minicorpus.jsonl'))
    p.add_argument('--max-pages', type=int, default=100)
    p.add_argument('--sleep', type=float, default=1.3)
    p.add_argument('--chunks-per-page', type=int, default=8)
    a = p.parse_args()
    out = Path(a.out)
    fw.PAGES  # noqa: ensure import
    seen = set()
    n = 0
    if out.exists():
        for line in out.read_text(encoding='utf-8').splitlines():
            if line.strip():
                seen.add(json.loads(line)['title'])
                n += 1
    titles = list(dict.fromkeys(EXTRA_TITLES))
    for cat in CATEGORIES:
        try:
            titles.extend(category_titles(cat))
            time.sleep(a.sleep)
        except Exception as e:
            print('category fail', cat, e, flush=True)
    titles = list(dict.fromkeys(titles))
    added_pages = 0
    for title in titles:
        if added_pages >= a.max_pages:
            break
        if title in seen:
            continue
        art = fw.extract_title(title)
        time.sleep(a.sleep)
        if not art or art['title'] in seen:
            continue
        parts = fw.chunk(art['text'])[:a.chunks_per_page]
        if not parts:
            continue
        with open(out, 'a', encoding='utf-8') as f:
            for i, c in enumerate(parts):
                if wiki_bm25.is_junk_chunk(c):
                    continue
                rec = {'id': f"{art['pageid']}#{i}", 'title': art['title'], 'url': art['url'],
                       'license': fw.LICENSE, 'text': c}
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n += 1
        seen.add(art['title'])
        added_pages += 1
        print(f'+ {art["title"]}  pages={added_pages} chunks={n}', flush=True)
    print(f'done: +{added_pages} pages, corpus chunks ~{n} -> {out}')


if __name__ == '__main__':
    main()
