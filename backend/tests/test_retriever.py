"""Tests for the retriever: embed a query and pull the nearest transcript chunks."""
from __future__ import annotations

import pytest

from backend.app.llm.base import LLMProvider, Message
from backend.app.rag.ingest import ingest
from backend.app.rag.retriever import VectorStoreNotFoundError, retrieve, retrieve_principles


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


@pytest.mark.live
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


def _build_principle_store(tmp_path, provider):
    import json
    import chromadb

    principles = [
        {"id": "price-ladder", "name": "Price is a ladder", "scope": "pricing",
         "principle": "Start price low to get flow, then raise it systematically.",
         "rationale": "Easy to go up, hard to come down.", "confidence": 0.9,
         "quotes": [{"text": "easy to go up in price, hard to come down", "source": "marketing.txt"}]},
        {"id": "maker-time", "name": "Protect maker time", "scope": "time",
         "principle": "Guard large uninterrupted blocks of deep work.",
         "rationale": "Meetings destroy maker output.", "confidence": 0.95,
         "quotes": [{"text": "a meeting destroys the whole block", "source": "time.txt"}]},
    ]
    col = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_or_create_collection(
        "demo_principles", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[p["id"] for p in principles],
        embeddings=provider.embed([p["principle"] for p in principles]),
        documents=[p["principle"] for p in principles],
        metadatas=[{"id": p["id"], "name": p["name"], "principle": p["principle"],
                    "rationale": p["rationale"], "scope": p["scope"], "confidence": p["confidence"],
                    "quotes": json.dumps(p["quotes"])} for p in principles],
    )


@pytest.mark.live
def test_retrieve_principles_returns_nearest_with_parsed_quotes(tmp_path):
    from backend.app.llm.factory import get_provider
    provider = get_provider()
    _build_principle_store(tmp_path, provider)

    hits = retrieve_principles(
        "how should I set the price on my new offer?",
        top_k=2, chroma_dir=str(tmp_path / "chroma"), persona="demo", provider=provider)

    assert len(hits) == 2
    assert hits[0]["id"] == "price-ladder"  # pricing question -> pricing principle first
    assert hits[0]["quotes"][0]["source"] == "marketing.txt"  # quotes parsed from JSON metadata
    assert set(hits[0]) >= {"id", "name", "principle", "quotes", "distance"}


def test_retrieve_principles_missing_store_returns_empty(tmp_path):
    hits = retrieve_principles(
        "anything", chroma_dir=str(tmp_path / "empty"), persona="nope", provider=_UnusedProvider())
    assert hits == []  # no principle store yet -> reasoning path simply can't run
