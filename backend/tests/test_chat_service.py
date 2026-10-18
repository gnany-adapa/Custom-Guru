"""Tests for the chat service: retrieve context, then answer grounded in it."""
from __future__ import annotations

from backend.app.chat.service import answer
from backend.app.rag.ingest import ingest


def test_answer_covered_question_is_grounded(tmp_path):
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    (tdir / "delegation.txt").write_text(
        "The way I think about delegation is simple. Delegate any task someone else can "
        "do at least eighty percent as well as you. Your time is your most valuable "
        "asset, so protect it ruthlessly and buy back your time. " * 20
    )
    cdir = tmp_path / "chroma"
    ingest(
        transcripts_dir=str(tdir),
        chroma_dir=str(cdir),
        collection_name="guru",
        chunk_size_words=60,
        overlap_words=15,
    )

    result = answer(
        "How do I decide what to delegate?",
        chroma_dir=str(cdir),
        collection_name="guru",
    )

    assert result["covered"] is True
    assert isinstance(result["answer"], str) and result["answer"].strip() != ""
    # Any citation the model emits must point at the only transcript we ingested.
    assert all(c["source"] == "delegation.txt" for c in result["citations"])
