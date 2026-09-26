# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyarrow>=15",
#   "tantivy==0.26.2",
# ]
# ///
"""Build the offline Polish Wikipedia BM25 index used by scripts/wiki_rag.py.

Source: Hugging Face dataset wikimedia/wikipedia, config 20231101.pl (6 parquet
files, ~1.8 GB, 1.59M articles). Each article is cut at the first reference-type
heading (Przypisy, Bibliografia, Linki zewnętrzne, Zobacz też, ...), split into
sections and packed into ~120-250 word passages, each prefixed with
"<title>" or "<title> — <section>". Disambiguation pages and very short stubs
are dropped. Indexed fields (see wiki_rag.normalize_tokens): body_norm (title +
section + body terms), title_norm (title terms; empty for year pages), body_bi
(adjacent-term bigrams). The index is a tantivy (Rust, disk-based, mmap)
index; RAM use is bounded by --heap-mb.

Gentle run on a shared GPU box (CPU only):
  nice -n 19 ionice -c3 uv run scripts/wiki_build_index.py \
      --raw-dir /scratch/wiki/raw --out /scratch/wiki/plwiki-20231101-v2 --threads 1 --heap-mb 1024
  cp -r /scratch/wiki/plwiki-20231101-v2 /team/wiki/   # persistent copy
Missing parquet files are downloaded into --raw-dir first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pyarrow.parquet as pq  # noqa: E402
import tantivy  # noqa: E402

import wiki_rag  # noqa: E402  (shared normalisation)
from wiki_rag import bigrams, normalize_tokens  # noqa: E402

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
DATASET = "wikimedia/wikipedia"

# Headings after which the rest of the article is references/links/categories.
TAIL_HEADINGS = {
    "przypisy", "bibliografia", "linki zewnętrzne", "zobacz też", "uwagi", "literatura",
    "źródła", "bibliografia uzupełniająca", "literatura uzupełniająca", "przypisy i uwagi",
    "uwagi i przypisy", "źródła i bibliografia", "bibliografia i linki zewnętrzne",
    "zewnętrzne linki", "linki", "noty", "objaśnienia", "przypisy bibliograficzne",
}
DISAMBIG_MARKERS = ("może oznaczać", "może dotyczyć", "może odnosić się", "może się odnosić")
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9„\"(\[])")
_HEAD_END_PUNCT = tuple(".,;:!?…\"”»")


def download(config: str, raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    api = f"{HF}/api/datasets/{DATASET}/tree/main/{config}"
    with urllib.request.urlopen(api, timeout=60) as r:
        files = sorted(x["path"] for x in json.load(r) if x["path"].endswith(".parquet"))
    out = []
    for rel in files:
        dst = raw_dir / Path(rel).name
        if not dst.exists():
            url = f"{HF}/datasets/{DATASET}/resolve/main/{rel}"
            print(f"[download] {url}", flush=True)
            tmp = dst.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.rename(dst)
        out.append(dst)
    return out


def _is_heading(raw: str, s: str, isolated: bool) -> bool:
    # plwiki plain text renders section titles as short lines: either with a
    # trailing space (section with direct content) or alone between blank lines
    # (section whose first child is a subsection). List items start with a space.
    if not s or len(s) > 90 or raw.startswith(" ") or s.endswith(_HEAD_END_PUNCT):
        return False
    nw = len(s.split())
    if nw > 10 or not any(c.isalpha() for c in s):
        return False
    return raw.endswith(" ") or s.lower() in TAIL_HEADINGS or (isolated and nw <= 8)


def sections(text: str) -> list[tuple[str, list[str]]]:
    secs: list[tuple[str, list[str]]] = [("", [])]
    lines = text.split("\n")
    n = len(lines)
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        isolated = 0 < i < n - 1 and not lines[i - 1].strip() and not lines[i + 1].strip()
        if _is_heading(raw, s, isolated):
            if s.lower() in TAIL_HEADINGS:
                break
            secs.append((s, []))
        else:
            secs[-1][1].append(s)
    return [(h, p) for h, p in secs if p]


def _split_long(p: str, cmax: int) -> list[str]:
    """Split a paragraph longer than cmax words into sentence-packed pieces."""
    pieces, buf, bw = [], [], 0
    for sent in _SENT_SPLIT.split(p):
        words = sent.split()
        while len(words) > cmax:  # pathological sentence: hard word windows
            if buf:
                pieces.append(" ".join(buf))
                buf, bw = [], 0
            pieces.append(" ".join(words[:cmax]))
            words = words[cmax:]
        if not words:
            continue
        if bw and bw + len(words) > cmax * 0.8:
            pieces.append(" ".join(buf))
            buf, bw = [], 0
        buf.append(" ".join(words))
        bw += len(words)
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def article_chunks(text: str, cmin: int, cmax: int) -> list[tuple[str, str, int]]:
    """-> [(section_heading, body, n_words)], bodies of ~cmin..cmax words."""
    out: list[tuple[str, str, int]] = []
    buf: list[str] = []
    bw = 0
    head = ""

    def flush():
        nonlocal buf, bw
        if buf:
            out.append((head, "\n".join(buf), bw))
        buf, bw = [], 0

    for h, paras in sections(text):
        if bw >= cmin // 2:  # a tiny section is merged into the next one
            flush()
        if not buf:
            head = h
        for p in paras:
            pw = len(p.split())
            for piece in (_split_long(p, cmax) if pw > cmax else [p]):
                w = len(piece.split()) if pw > cmax else pw
                if bw and bw + w > cmax:
                    flush()
                    head = h
                buf.append(piece)
                bw += w
    flush()
    # merge a short tail into the previous passage of the same article
    if len(out) >= 2 and out[-1][2] < cmin // 2 and out[-2][2] + out[-1][2] <= cmax + cmin // 2:
        (h1, b1, w1), (_, b2, w2) = out[-2], out[-1]
        out[-2:] = [(h1, b1 + "\n" + b2, w1 + w2)]
    return out


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build_schema() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("title", stored=True, tokenizer_name="raw", index_option="basic")
    sb.add_bytes_field("text", stored=True, indexed=False)
    sb.add_unsigned_field("art_id", stored=True)
    sb.add_unsigned_field("chunk", stored=True)
    sb.add_text_field("title_norm", stored=False, tokenizer_name="default", index_option="freq")
    sb.add_text_field("body_norm", stored=False, tokenizer_name="default", index_option="freq")
    sb.add_text_field("body_bi", stored=False, tokenizer_name="default", index_option="freq")
    return sb.build()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="20231101.pl")
    ap.add_argument("--raw-dir", default="/scratch/wiki/raw")
    ap.add_argument("--out", default="/scratch/wiki/plwiki-20231101-v2")
    ap.add_argument("--heap-mb", type=int, default=1024, help="tantivy writer heap (bounds RAM)")
    ap.add_argument("--threads", type=int, default=1, help="tantivy indexing threads")
    ap.add_argument("--chunk-min", type=int, default=120)
    ap.add_argument("--chunk-max", type=int, default=250)
    ap.add_argument("--min-words", type=int, default=20, help="drop articles shorter than this")
    ap.add_argument("--limit", type=int, default=0, help="stop after N articles (testing)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            sys.exit(f"{out} exists and is not empty (use --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    files = download(args.config, Path(args.raw_dir))
    index = tantivy.Index(build_schema(), path=str(out))
    writer = index.writer(heap_size=args.heap_mb * 2**20, num_threads=args.threads)

    t0 = time.time()
    st = dict(articles=0, kept=0, disambig=0, short=0, chunks=0, words=0)
    done = False
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=256, columns=["id", "title", "text"]):
            ids = batch.column("id").to_pylist()
            titles = batch.column("title").to_pylist()
            texts = batch.column("text").to_pylist()
            for aid, title, text in zip(ids, titles, texts):
                st["articles"] += 1
                head = text[:300].lower()
                if title.endswith("(ujednoznacznienie)") or any(m in head for m in DISAMBIG_MARKERS):
                    st["disambig"] += 1
                    continue
                chunks = article_chunks(text, args.chunk_min, args.chunk_max)
                if sum(c[2] for c in chunks) < args.min_words:
                    st["short"] += 1
                    continue
                st["kept"] += 1
                tt = normalize_tokens(title)
                # year pages ("1569", "966") are event lists; a title-boosted year
                # match would crowd out real articles, so they match on body only
                title_norm = "" if all(t.isdigit() for t in tt) else " ".join(tt)
                for i, (sec, body, nw) in enumerate(chunks):
                    prefix = f"{title} — {sec}" if sec else title
                    doc = tantivy.Document()
                    doc.add_text("title", title)
                    doc.add_bytes("text", f"{prefix}\n{body}".encode("utf-8"))
                    doc.add_unsigned("art_id", int(aid) if aid.isdigit() else 0)
                    doc.add_unsigned("chunk", i)
                    doc.add_text("title_norm", title_norm)
                    toks = normalize_tokens(f"{prefix}\n{body}")
                    doc.add_text("body_norm", " ".join(toks))
                    doc.add_text("body_bi", " ".join(bigrams(toks)))
                    writer.add_document(doc)
                    st["chunks"] += 1
                    st["words"] += nw
                if st["articles"] % 50000 == 0:
                    el = time.time() - t0
                    print(f"[build] {el/60:6.1f} min {st} rate={st['articles']/el:.0f} art/s "
                          f"rss={_rss_mb():.0f}MB file={fp.name}", flush=True)
                if args.limit and st["articles"] >= args.limit:
                    done = True
                    break
            if done:
                break
        if done:
            break

    print(f"[build] committing... {st}", flush=True)
    writer.commit()
    writer.wait_merging_threads()
    minutes = (time.time() - t0) / 60
    size_gb = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e9
    info = {
        "source": f"{DATASET}:{args.config}",
        "files": [f.name for f in files],
        "stats": st,
        "build_minutes": round(minutes, 1),
        "index_gb": round(size_gb, 2),
        "norm_version": wiki_rag.NORM_VERSION,
        "prefix_len": wiki_rag.PREFIX_LEN,
        "chunk_min": args.chunk_min,
        "chunk_max": args.chunk_max,
        "min_words": args.min_words,
        "limit": args.limit,
        "tantivy": getattr(tantivy, "__version__", "0.26.2"),
    }
    (out / "wiki_build_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print("[build] DONE " + json.dumps(info, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
