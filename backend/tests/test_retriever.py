"""Tests for the retriever: embed a query and pull the nearest transcript chunks."""
from __future__ import annotations

import pytest

from backend.app.llm.base import LLMProvider, Message
from backend.app.rag.ingest import ingest
from backend.app.rag.retriever import VectorStoreNotFoundError, retrieve


class _UnusedProvider(LLMProvider):
    """Fails if used — proves a missing collection is detected before embedding."""

    def generate(self, messages: list[Message], system: str | None = None, max_tokens: int = 1000) -> str:
        raise AssertionError("generate should not be called")

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed should not be called when the collection is missing")


def test_retrieve_missing_collection_raises_clear_error(tmp_path):
    with pytest.raises(VectorStoreNotFoundError):
        retrieve(
            "anything",
            chroma_dir=str(tmp_path / "empty"),
            collection_name="does_not_exist",
            provider=_UnusedProvider(),
        )


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
