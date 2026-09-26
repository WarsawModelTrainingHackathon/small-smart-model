from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import wiki_rag  # noqa: E402


def test_build_query_adds_short_clean_source_excerpt_for_generic_question():
    item = {
        "question": "Podaj nazwę władcy, którego dotyczą oba źródła.\nOdpowiedź:",
        "context": (
            "Źródło 1. Fragment opracowania historycznego\n"
            "Po śmierci Ludwiki Marii sytuacja polityczna Rzeczypospolitej uległa zmianie.\n"
            "Jan Kowalski, Historia Polski, Warszawa 2001, s. 12.\n"
            "[OBRAZ] https://example.invalid/image.png"
        ),
        "answer": "SECRET GOLD ANSWER",
        "answer_key": "SECRET KEY",
        "scoring": "SECRET SCORING",
    }

    query = wiki_rag.build_query(item)

    assert "Ludwiki Marii" in query
    assert "SECRET" not in query
    assert "[OBRAZ]" not in query
    assert "https://" not in query
    assert "Warszawa 2001" not in query
    assert "Źródło 1" not in query


def test_build_query_uses_specific_quoted_topic_without_noisy_context():
    item = {
        "question": "Wyjaśnij sens sformułowania „Paryż wart mszy”.",
        "context": "NOISY NAME, Source text that must not be added.",
    }

    query = wiki_rag.build_query(item)

    assert "Paryż wart mszy" in query
    assert "NOISY NAME" not in query


def test_build_query_ignores_gold_fields():
    base = {"question": "Podaj nazwę wydarzenia.", "context": "Tekst źródła o wydarzeniu."}
    changed_gold = dict(base, answer="different", answer_key={"A": "different"}, scoring="different")

    assert wiki_rag.build_query(base) == wiki_rag.build_query(changed_gold)


def test_missing_or_unbuilt_index_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="wiki_build_index.py"):
        wiki_rag.WikiIndex(str(tmp_path / "missing"))

    unbuilt = tmp_path / "unbuilt"
    unbuilt.mkdir()
    with pytest.raises(FileNotFoundError, match="wiki_build_index.py"):
        wiki_rag.WikiIndex(str(unbuilt))


def test_stale_normalization_index_has_rebuild_error(tmp_path):
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "wiki_build_info.json").write_text('{"norm_version": 999}', encoding="utf-8")

    with pytest.raises(ValueError, match="Rebuild the index"):
        wiki_rag.WikiIndex(str(stale))


def test_search_prefers_distinct_article_titles(tmp_path):
    path = tmp_path / "index"
    path.mkdir()
    sb = wiki_rag.tantivy.SchemaBuilder()
    sb.add_text_field("title", stored=True, tokenizer_name="raw", index_option="basic")
    sb.add_bytes_field("text", stored=True, indexed=False)
    sb.add_unsigned_field("art_id", stored=True)
    sb.add_unsigned_field("chunk", stored=True)
    sb.add_text_field("title_norm", stored=False, tokenizer_name="default", index_option="freq")
    sb.add_text_field("body_norm", stored=False, tokenizer_name="default", index_option="freq")
    sb.add_text_field("body_bi", stored=False, tokenizer_name="default", index_option="freq")
    index = wiki_rag.tantivy.Index(sb.build(), path=str(path))
    writer = index.writer(heap_size=32 * 1024 * 1024, num_threads=1)
    for title, text in (
        ("Historia Polski", "pierwszy fragment"),
        ("Historia Polski", "drugi fragment"),
        ("Powstanie listopadowe", "inne źródło"),
        ("Powstanie styczniowe", "kolejne źródło"),
    ):
        doc = wiki_rag.tantivy.Document()
        doc.add_text("title", title)
        doc.add_bytes("text", text.encode("utf-8"))
        doc.add_unsigned("art_id", 1)
        doc.add_unsigned("chunk", 0)
        tokens = wiki_rag.normalize_tokens("Historia Polska Powstanie")
        doc.add_text("title_norm", " ".join(tokens))
        doc.add_text("body_norm", " ".join(tokens))
        doc.add_text("body_bi", " ".join(wiki_rag.bigrams(tokens)))
        writer.add_document(doc)
    writer.commit()
    (path / "wiki_build_info.json").write_text(
        '{"norm_version": %d}' % wiki_rag.NORM_VERSION, encoding="utf-8"
    )

    hits = wiki_rag.WikiIndex(str(path)).search("historia powstanie", k=3)

    assert len(hits) == 3
    assert len({h["title"].casefold() for h in hits}) == 3
