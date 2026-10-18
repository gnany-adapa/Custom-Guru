"""Tests for the grounded guard: off-topic questions are refused without an LLM call."""
from __future__ import annotations

from backend.app.chat.service import answer
from backend.app.llm.base import LLMProvider, Message
from backend.app.llm.factory import get_provider
from backend.app.rag.ingest import ingest


class _GenerateSpyProvider(LLMProvider):
    """Real embeddings, but fails loudly if generate() is ever called."""

    def __init__(self, real: LLMProvider) -> None:
        self._real = real
        self.generate_called = False

    def generate(self, messages: list[Message], system: str | None = None, max_tokens: int = 1000) -> str:
        self.generate_called = True
        raise AssertionError("LLM must not be called for an uncovered question")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._real.embed(texts)


def test_uncovered_question_refused_without_llm(tmp_path):
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

    spy = _GenerateSpyProvider(get_provider())
    result = answer(
        "What is the best recipe for pasta carbonara?",
        provider=spy,
        chroma_dir=str(cdir),
        collection_name="guru",
    )

    assert result["covered"] is False
    assert spy.generate_called is False
    assert result["answer"].strip() != ""
    assert result["sources"] == []
