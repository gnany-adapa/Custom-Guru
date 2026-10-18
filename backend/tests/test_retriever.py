"""Tests for the retriever: embed a query and pull the nearest transcript chunks."""
from __future__ import annotations

from backend.app.rag.ingest import ingest
from backend.app.rag.retriever import retrieve


def test_retrieve_returns_nearest_chunks_first(tmp_path):
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    (tdir / "pricing.txt").write_text(" ".join(["pricing offers value money"] * 80))
    (tdir / "fitness.txt").write_text(" ".join(["morning workout sleep protein"] * 80))
    cdir = tmp_path / "chroma"
    ingest(
        transcripts_dir=str(tdir),
        chroma_dir=str(cdir),
        collection_name="testcol",
        chunk_size_words=50,
        overlap_words=10,
    )

    hits = retrieve(
        "how should I price my product?",
        top_k=2,
        chroma_dir=str(cdir),
        collection_name="testcol",
    )

    assert len(hits) == 2
    # Each hit carries the text, its source file, and a distance.
    assert set(hits[0]) >= {"text", "source", "distance"}
    # Nearest-first ordering.
    assert hits[0]["distance"] <= hits[1]["distance"]
    # The pricing question should surface the pricing transcript as the top hit.
    assert hits[0]["source"] == "pricing.txt"
