"""Tests for the grounded guard (hybrid behavior).

When the nearest retrieved chunk is too far, the service no longer returns a canned
refusal. Instead it calls the LLM with a *fallback* prompt (covered=False, no sources),
letting the model greet small talk warmly or decline a genuine off-topic question.
"""
from __future__ import annotations

import pytest

from backend.app.chat.service import answer
from backend.app.llm.base import LLMProvider, Message
from backend.app.llm.factory import get_provider
from backend.app.rag.ingest import ingest


class _FallbackSpyProvider(LLMProvider):
    """Real embeddings; records the generate() call and the system prompt it received."""

    def __init__(self, real: LLMProvider) -> None:
        self._real = real
        self.generate_called = False
        self.system_seen: str | None = None

    def generate(self, messages: list[Message], system: str | None = None, max_tokens: int = 1000) -> str:
        self.generate_called = True
        self.system_seen = system
        return "FALLBACK_REPLY"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._real.embed(texts)


@pytest.mark.live
def test_low_match_uses_fallback_llm_not_canned_refusal(tmp_path):
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    (tdir / "delegation.txt").write_text(
        "Delegate any task someone else can do eighty percent as well as you. "
        "Protect your maker time and buy back your calendar. " * 20
    )
    cdir = tmp_path / "chroma"
    ingest(
        transcripts_dir=str(tdir),
        chroma_dir=str(cdir),
        collection_name="guru",
        chunk_size_words=60,
        overlap_words=15,
    )

    spy = _FallbackSpyProvider(get_provider())
    result = answer(
        "What is the best recipe for pasta carbonara?",
        provider=spy,
        chroma_dir=str(cdir),
        collection_name="guru",
    )

    assert result["covered"] is False
    assert result["citations"] == []
    assert spy.generate_called is True  # hybrid: the LLM is consulted on a low match
    assert result["answer"] == "FALLBACK_REPLY"
    # ...and it was handed the fallback prompt, not the grounded one.
    assert "no relevant material" in (spy.system_seen or "").lower()
