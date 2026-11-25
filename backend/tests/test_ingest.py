"""Tests for the ingestion pipeline: text cleaning, chunking, and vector-store build.

`chunk_text` / `clean_text` are pure and network-free. `ingest` hits the live embeddings
proxy and writes a temporary Chroma store, so the proxy must be running.
"""
from __future__ import annotations

import chromadb
import pytest

from backend.app.rag.ingest import chunk_text, clean_text, ingest


def test_clean_text_collapses_whitespace():
    raw = "  um,   hello\n\n\n  world  \t "
    assert clean_text(raw) == "um, hello world"


def test_chunk_text_splits_with_overlap():
    text = " ".join(str(i) for i in range(100))  # 100 "words": 0..99
    chunks = chunk_text(text, chunk_size_words=40, overlap_words=10)
    assert len(chunks) >= 3
    # Consecutive chunks overlap: the tail of one reappears at the head of the next.
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-10:] == second_words[:10]


def test_chunk_text_short_input_single_chunk():
    chunks = chunk_text("just a few words here", chunk_size_words=40, overlap_words=10)
    assert chunks == ["just a few words here"]


@pytest.mark.live
def test_ingest_builds_persistent_collection(tmp_path):
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    (tdir / "talk1.txt").write_text(" ".join(["value"] * 300))
    (tdir / "talk2.txt").write_text(" ".join(["offer"] * 300))
    cdir = tmp_path / "chroma"

    count = ingest(
        transcripts_dir=str(tdir),
        chroma_dir=str(cdir),
        collection_name="test_persona",
        chunk_size_words=100,
        overlap_words=20,
    )

    assert count > 0
    client = chromadb.PersistentClient(path=str(cdir))
    col = client.get_collection("test_persona")
    assert col.count() == count
    # Metadata records which file each chunk came from.
    sample = col.get(limit=1, include=["metadatas"])
    assert sample["metadatas"][0]["source"] in {"talk1.txt", "talk2.txt"}


@pytest.mark.parametrize("chunk_size,overlap", [(40, 40), (40, 50), (0, 0)])
def test_chunk_text_rejects_bad_overlap(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some words here to split", chunk_size_words=chunk_size, overlap_words=overlap)


def test_ingest_rejects_transcripts_with_no_usable_text(tmp_path):
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    (tdir / "blank.txt").write_text("   \n\n\t  ")  # whitespace only
    with pytest.raises(ValueError, match="no usable text"):
        ingest(
            transcripts_dir=str(tdir),
            chroma_dir=str(tmp_path / "chroma"),
            collection_name="blankpersona",
        )
