# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tantivy==0.26.2",
# ]
# ///
"""Offline Polish Wikipedia retrieval (BM25 over a local tantivy index).

The index is built by scripts/wiki_build_index.py (plwiki 20231101 from the
Hugging Face dataset wikimedia/wikipedia). Passages are ~120-250 words, each
prefixed with "<article title>" or "<article title> — <section>".

Polish is highly inflected, so both the index and the queries go through the same
crude normalisation (normalize_tokens): lowercase, strip diacritics, drop
stopwords, strip one common inflectional ending, truncate to PREFIX_LEN chars.
Scoring is BM25 over three fields: normalised passage text (title + section +
body), normalised article title (boosted) and adjacent-term bigrams of the
passage (boosted; rewards "Kazimierza Wielkiego" over "Wielka Reforma Teatru").
Exam boilerplate ("na podstawie źródła i własnej wiedzy wyjaśnij ...") is dropped
from queries, so a whole exam question can be passed as the query.

Python API:
    from wiki_rag import WikiIndex
    idx = WikiIndex("/team/wiki/plwiki-20231101-v2")
    for hit in idx.search("unia lubelska 1569 skutki", k=5):
        print(hit["score"], hit["title"], hit["text"][:80])

CLI:
    uv run scripts/wiki_rag.py --index /team/wiki/plwiki-20231101-v2 --query "Sejm Wielki" -k 5
    uv run scripts/wiki_rag.py --index /team/wiki/plwiki-20231101-v2 --bench   # p50 latency
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import tantivy

# ---------------------------------------------------------------------------
# Normalisation (shared by the index builder and the query side; changing it
# requires rebuilding the index -- NORM_VERSION is stored in the build info).
# ---------------------------------------------------------------------------
NORM_VERSION = 2
PREFIX_LEN = 7
MIN_STEM = 3
TITLE_BOOST = 1.5
BIGRAM_BOOST = 1.0

_FOLD = str.maketrans("ąćęłńóśźż", "acelnoszz")  # applied after .lower()
_TOKEN_RE = re.compile(r"\w+")
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$")

# Folded (no diacritics) Polish function words.
STOPWORDS = frozenset(
    """a aby ale albo ani az bez bo by byc byl byla byli bylo byly bedzie beda
    co czy dla do gdy gdzie go i ich ile im ja jak jako je jego jej jest jesli
    jeszcze jednak juz kiedy ktora ktore ktorego ktorej ktory ktorych ktorym
    ktorzy lub ma mu na nad nie niz nich o od oraz po pod poza przed przez
    przy roku sa sie sobie ta tak takze tam te tego tej ten to tu tym w we wiec
    wraz z za ze zas""".split()
)

# Common inflectional endings (folded), longest first; one is stripped if the
# remaining stem keeps >= MIN_STEM chars. The "ow*" family keeps
# Kraków/Krakowa/Krakowie together; "i*" keeps polski/polskiego/polskie together.
_ENDINGS = sorted(
    """owymi owych owego owemu iego iemu owej owym owie owem
    ymi ych ego emu iej ich imi ami ach iem iom owa owe owi owy owo
    ym im ej om em ie ia ii iu io ow
    a e i o u y""".split(),
    key=len,
    reverse=True,
)

_cache: dict[str, str] = {}
_CACHE_MAX = 2_000_000


def _fold_nonascii(tok: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", tok) if not unicodedata.combining(c))


def _norm_token(tok: str) -> str:
    """Folded, lowercased token -> index term ('' = drop)."""
    if not tok.isascii():
        tok = _fold_nonascii(tok)
    if tok in STOPWORDS:
        return ""
    if tok.isdigit():
        return tok[:8]
    if len(tok) < 2:
        return ""
    if any(c.isdigit() for c in tok):
        return tok[:12]
    if _ROMAN_RE.match(tok):  # XIII vs XII must stay distinct
        return tok
    for e in _ENDINGS:
        if tok.endswith(e) and len(tok) - len(e) >= MIN_STEM:
            tok = tok[: -len(e)]
            break
    return tok[:PREFIX_LEN]


def normalize_tokens(text: str) -> list[str]:
    """Text -> list of normalised index terms (same function for docs and queries)."""
    cache = _cache
    out = []
    for t in _TOKEN_RE.findall(text.lower().translate(_FOLD)):
        s = cache.get(t)
        if s is None:
            s = _norm_token(t)
            if len(cache) >= _CACHE_MAX:
                cache.clear()
            cache[t] = s
        if s:
            out.append(s)
    return out


def bigrams(terms: list[str]) -> list[str]:
    """Adjacent normalised terms glued together (safe for tantivy's 'default' tokenizer)."""
    return [a + b for a, b in zip(terms, terms[1:])]


# Query-only stopwords: matura task boilerplate (normalised at import time).
QUERY_STOP = frozenset(
    normalize_tokens(
        """podstawie źródła źródło źródeł źródłach własnej wiedzy wyjaśnij wyjaśnić
        odpowiedź odpowiedzi uwzględnij podaj podać wymień oceń uzasadnij określ
        zadanie zadania zadaniu fragment fragmentu tekst tekstu ilustracja ilustracji
        tabela tabeli poniższy poniższego powyższy przedstaw przedstawionego
        przedstawiony rozstrzygnij zaznacz wpisz uzupełnij prawdziwe fałszywe
        jakie jaki jaka jakich dlaczego np itp itd
        źródło źródła źródle fragment fragmenty tekst tekstu ilustracja ilustracji
        mapa mapie mapę fotografia fotografii rycina karykatura drzeworyt plakat
        rozstrzygnij rozstrzygnąć zaznacz dokończ sformułuj argument uzasadnij
        odpowiedź odpowiedzi uzasadnienie rozstrzygnięcie przedstawiony przedstawione
        opisz podaj wymień wskaż określ wyjaśnij porównaj oceń opowiedz scharakteryzuj
        wydarzenie wydarzenia działania działaniach polityk polityka struktura
        władca władcy nazwa imię nazwisko przykład przykłady źródłach informacji
        historii historiografii zadanie zadania temat tematy"""
    )
)

_CITATION_LINE = re.compile(
    r"(?im)^\s*(?:[A-ZĄĆĘŁŃÓŚŹŻ][^\n]{0,110},\s*)?"
    r"(?:Kraków|Warszawa|Poznań|Łódź|Wrocław|Gdańsk|Lublin|London|Paris|Oxford|Cambridge)\s+\d{4}[^\n]*$"
)
_URL = re.compile(r"https?://\S+", re.I)
_SOURCE_LABEL = re.compile(
    r"(?i)\b(?:Źródło\s*\d*\.?|Fragment(?:y)?(?:\s+opracowania\s+historycznego|"
    r"\s+dokumentu|\s+tekstu|\s+mowy|\s+pamiętników\s+z\s+epoki)?|"
    r"Kadr\s+z\s+filmu|Karykatura\s+polityczna|Plakat|Mapa|Rycina|Drzeworyt)\b"
)
_CAPITALIZED_PHRASE = re.compile(
    r"\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,})+\b"
)
_DATE_OR_QUOTE = re.compile(r"\b(?:1[0-9]{3}|20[0-2][0-9]|[0-9]{3})\b|„[^”]{3,}”|\"[^\"]{3,}\"")


def build_query(item: dict) -> str:
    """Build a retrieval query from the question and source text only.

    Use the question (including multiple-choice options), or the sole selected
    essay theme when one is marked. Remove empty answer scaffolds, image
    placeholders, URLs and bibliographic lines, then add a short source excerpt
    for generic questions. Gold answers, keys and scoring fields are never read.
    """
    question = str(item.get("question") or "")
    question = re.sub(r"(?im)^\s*(?:Rozstrzygnięcie|Uzasadnienie|Odpowiedź|Wydarzenie)\s*:\s*.*$", " ", question)
    question = question.replace("[OBRAZ]", " ")
    question = re.sub(r"\s+", " ", question).strip()

    # Essay prompts often prepend generic instructions (including the minimum
    # word count) before a single selected theme. Search the theme itself so
    # boilerplate terms like "temat" and "wyrazów" do not dominate BM25.
    if str(item.get("type") or "").lower() == "essay":
        topic_markers = list(re.finditer(r"\bTemat\s+\d+\s*[.:—-]\s*", question, re.I))
        if len(topic_markers) == 1:
            question = question[topic_markers[0].end():].strip()

    context = str(item.get("context") or "")
    context = _CITATION_LINE.sub(" ", context)
    context = _URL.sub(" ", context.replace("[OBRAZ]", " "))
    context = _SOURCE_LABEL.sub(" ", context)
    context = re.sub(r"\s+", " ", context).strip()

    # Specific names, dates and quoted concepts in the question already make
    # good search keys; extra source prose can then add distracting terms.
    # Generic prompts ("which ruler?", "what event?") need a short source
    # excerpt to identify the subject. Ignore multiple-choice options when
    # deciding whether the question itself has an anchor.
    anchor_text = re.split(r"\bA\.\s+", question, maxsplit=1)[0]
    anchored = bool(_DATE_OR_QUOTE.search(anchor_text))
    if not anchored:
        for match in _CAPITALIZED_PHRASE.finditer(anchor_text):
            prefix = anchor_text[: match.start()].rstrip()
            if prefix and prefix[-1] not in ".!?;:\n":
                anchored = True
                break

    # The dev comparison favored question-only for already named topics, and
    # question + the first 65 source words for generic references to a source.
    excerpt = "" if anchored else " ".join(context.split()[:65])
    return (question + " " + excerpt).strip()[:1600]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class WikiIndex:
    """Read-only BM25 search over the local Polish Wikipedia passage index."""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        path = Path(index_dir)
        info_path = path / "wiki_build_info.json"
        if not path.is_dir() or not info_path.is_file():
            raise FileNotFoundError(
                f"Wikipedia index not found at {index_dir!r}; build it with scripts/wiki_build_index.py"
            )
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("norm_version") != NORM_VERSION:
            raise ValueError(
                f"Index at {index_dir!r} uses normalization version {info.get('norm_version')}; "
                f"this code needs version {NORM_VERSION}. Rebuild the index."
            )
        self.index = tantivy.Index.open(index_dir)
        self.index.reload()
        self.searcher = self.index.searcher()
        self.schema = self.index.schema
        try:  # indexes built with NORM_VERSION 1 have no bigram field
            tantivy.Query.term_query(self.schema, "body_bi", "x", index_option="freq")
            self.has_bigrams = True
        except ValueError:
            self.has_bigrams = False

    def _term(self, field: str, term: str, boost: float = 1.0):
        q = tantivy.Query.term_query(self.schema, field, term, index_option="freq")
        return q if boost == 1.0 else tantivy.Query.boost_query(q, boost)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Top-k passages for a free-text Polish query: [{title, text, score}, ...]."""
        seq = normalize_tokens(query)
        terms = [t for t in dict.fromkeys(seq) if t not in QUERY_STOP]  # dedupe, keep order
        if not terms or k <= 0:
            return []
        should = tantivy.Occur.Should
        subs = []
        for t in terms:
            subs.append((should, self._term("body_norm", t)))
            subs.append((should, self._term("title_norm", t, TITLE_BOOST)))
        if self.has_bigrams:
            pairs = [(a, c) for a, c in zip(seq, seq[1:]) if a not in QUERY_STOP and c not in QUERY_STOP]
            for b in dict.fromkeys(a + c for a, c in pairs):
                subs.append((should, self._term("body_bi", b, BIGRAM_BOOST)))
        q = tantivy.Query.boolean_query(subs)
        # Several adjacent chunks from one long article can crowd out distinct
        # sources. Scan a few extra matches and prefer one passage per title.
        res = self.searcher.search(q, limit=max(k * 4, k), count=False)
        hits = []
        repeats = []
        seen_titles = set()
        for score, addr in res.hits:
            doc = self.searcher.doc(addr)
            title = doc.get_first("title")
            text = doc.get_first("text")
            if isinstance(text, (bytes, bytearray)):
                text = bytes(text).decode("utf-8")
            hit = {"title": title, "text": text, "score": float(score)}
            title_key = title.casefold()
            if title_key in seen_titles:
                repeats.append(hit)
                continue
            seen_titles.add(title_key)
            hits.append(hit)
            if len(hits) == k:
                break
        hits.extend(repeats[: max(0, k - len(hits))])
        return hits


BENCH_QUERIES = [
    "Sejm Wielki zniesienie liberum veto",
    "unia lubelska 1569 skutki",
    "Konstytucja 3 maja postanowienia",
    "pokój toruński 1466 Prusy Królewskie",
    "hołd pruski Albrecht Hohenzollern",
    "bitwa pod Grunwaldem 1410",
    "przywilej koszycki Ludwik Węgierski szlachta",
    "konfederacja barska 1768",
    "powstanie styczniowe branka Wielopolski",
    "Księstwo Warszawskie Napoleon kodeks",
    "reformy Kazimierza Wielkiego",
    "chrzest Polski 966 Mieszko I",
    "rozbicie dzielnicowe testament Bolesława Krzywoustego",
    "pierwszy rozbiór Polski 1772",
    "traktat wersalski Gdańsk wolne miasto",
    "przewrót majowy 1926 Piłsudski sanacja",
    "reforma rolna PKWN 1944",
    "Solidarność porozumienia sierpniowe 1980",
    "wojna trzydziestoletnia pokój westfalski",
    "rewolucja francuska Deklaracja praw człowieka i obywatela",
]


def _bench(idx: WikiIndex, k: int) -> dict:
    idx.search("rozgrzewka indeksu", k=k)  # first query pays for opening segments
    lat = []
    tops = []
    for q in BENCH_QUERIES:
        t0 = time.perf_counter()
        hits = idx.search(q, k=k)
        lat.append((time.perf_counter() - t0) * 1000)
        tops.append((q, [h["title"] for h in hits[:3]]))
    for (q, titles), ms in zip(tops, lat):
        print(f"{ms:7.1f} ms  {q!r:60} -> {titles}")
    lat_sorted = sorted(lat)
    out = {
        "n": len(lat),
        "p50_ms": round(statistics.median(lat), 2),
        "p90_ms": round(lat_sorted[int(0.9 * (len(lat) - 1))], 2),
        "max_ms": round(max(lat), 2),
    }
    print(json.dumps(out))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="/team/wiki/plwiki-20231101-v2", help="index directory")
    ap.add_argument("--query", "-q", help="free-text query (Polish)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="print hits as JSON")
    ap.add_argument("--bench", action="store_true", help="latency on 20 built-in history queries")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    idx = WikiIndex(args.index)
    print(f"[wiki_rag] opened {args.index} in {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    if args.bench:
        _bench(idx, args.k)
        return 0
    if not args.query:
        ap.error("--query or --bench required")
    t0 = time.perf_counter()
    hits = idx.search(args.query, k=args.k)
    ms = (time.perf_counter() - t0) * 1000
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=1))
    else:
        for i, h in enumerate(hits, 1):
            print(f"#{i} [{h['score']:.2f}] {h['title']}\n{h['text']}\n")
    print(f"[wiki_rag] {len(hits)} hits in {ms:.1f} ms", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
