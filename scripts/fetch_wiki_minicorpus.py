#!/usr/bin/env python3
"""Download a small Polish Wikipedia slice (CC BY-SA 4.0) and chunk it.

  python3 scripts/fetch_wiki_minicorpus.py
  python3 scripts/fetch_wiki_minicorpus.py --out harness/wiki/minicorpus.jsonl

Pages target the base-dev misses (Augustus vs Caesar, Jaruzelski, 1968, etc.).
Do not scrape CKE PDFs here. Wikipedia API only.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = 'z-abka-matura/0.1 (Warsaw Model Trainers hackathon; educational; contact: team z-abka)'
API = 'https://pl.wikipedia.org/w/api.php'
LICENSE = 'CC BY-SA 4.0 (https://pl.wikipedia.org/wiki/Wikipedia:Tekst_licencji_Creative_Commons)'

# Title must exist (or redirect) on pl.wikipedia.org
PAGES = [
    'Oktawian August',
    'Juliusz Cezar',
    'Henryk V Salicki',
    'Henryk V Lancaster',
    'Jan II Kazimierz Waza',
    'Ludwika Maria Gonzaga',
    'Wojciech Jaruzelski',
    'Stan wojenny w Polsce (1981–1983)',
    'Interwencja Układu Warszawskiego w Czechosłowacji',
    'Praska Wiosna',
    'Marzec 1968',
    'Henryk IV Burbon',
    'Edykt nantejski',
    'Wikingowie',
    'Wyprawy krzyżowe',
    'Bolesław I Chrobry',
    'Zjazd gnieźnieński',
    'Powstanie wielkopolskie',
    'Józef Dowbor-Muśnicki',
    'Józef Piłsudski',
    'Koronacja na króla Polski',
]

_WS = re.compile(r'\n{3,}')


def api(params, retries=6):
    q = urllib.parse.urlencode({**params, 'format': 'json'})
    req = urllib.request.Request(f'{API}?{q}', headers={'User-Agent': UA})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or i == retries - 1:
                raise
            time.sleep(15 * (i + 1))
    raise RuntimeError('wikipedia api failed')


def extract_title(title):
    data = api({
        'action': 'query',
        'prop': 'extracts|info',
        'explaintext': '1',
        'exsectionformat': 'plain',
        'redirects': '1',
        'inprop': 'url',
        'titles': title,
    })
    pages = data.get('query', {}).get('pages', {})
    page = next(iter(pages.values()))
    if page.get('missing') or 'extract' not in page:
        print(f'SKIP missing Wikipedia page: {title!r}', flush=True)
        return None
    text = _WS.sub('\n\n', (page.get('extract') or '').strip())
    return {
        'title': page.get('title', title),
        'pageid': page.get('pageid'),
        'url': page.get('fullurl') or f'https://pl.wikipedia.org/wiki/{urllib.parse.quote(title)}',
        'text': text,
    }


def chunk(text, size=900, overlap=80):
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    buf, n, out = [], 0, []
    for p in paras:
        if n + len(p) > size and buf:
            out.append('\n'.join(buf))
            # overlap: keep last short paragraph
            keep = buf[-1] if overlap and len(buf[-1]) <= overlap * 2 else ''
            buf, n = ([keep] if keep else []), (len(keep) if keep else 0)
        buf.append(p)
        n += len(p) + 1
    if buf:
        out.append('\n'.join(buf))
    return [c for c in out if len(c) > 80]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'harness' / 'wiki' / 'minicorpus.jsonl'))
    p.add_argument('--sleep', type=float, default=1.2)
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = []
    if out.exists():
        for line in out.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            seen.add(r['title'])
        print(f'resume {len(rows)} chunks, titles {sorted(seen)}', flush=True)
    for title in PAGES:
        # skip if we already have this requested title or the resolved wiki title
        if title in seen:
            print(f'skip have {title}', flush=True)
            continue
        art = extract_title(title)
        if not art:
            continue
        if art['title'] in seen:
            print(f'skip resolved {art["title"]}', flush=True)
            continue
        parts = chunk(art['text'])[:12]
        print(f'{art["title"]}: {len(art["text"])} chars -> {len(parts)} chunks', flush=True)
        new = []
        for i, c in enumerate(parts):
            new.append({
                'id': f"{art['pageid']}#{i}",
                'title': art['title'],
                'url': art['url'],
                'license': LICENSE,
                'text': c,
            })
        rows.extend(new)
        seen.add(art['title'])
        seen.add(title)
        with open(out, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        time.sleep(a.sleep)
    (out.parent / 'SOURCES.md').write_text(
        'Polish Wikipedia extracts, CC BY-SA 4.0.\n'
        f'Rebuild: python3 scripts/fetch_wiki_minicorpus.py --out {out}\n'
        'Pages:\n' + '\n'.join(f'- {t}' for t in PAGES) + '\n',
        encoding='utf-8',
    )
    print(f'wrote {len(rows)} chunks -> {out}')


if __name__ == '__main__':
    main()
