"""Offline unit tests for the chat service: grounded/fallback branch + citation handling.

Retrieval is stubbed and the provider is faked, so these run with no proxy or vector
store and pin down the core logic deterministically.
"""
from __future__ import annotations

import backend.app.chat.service as service
from backend.app.chat.service import answer
from backend.app.llm.base import LLMProvider, Message


class _FakeProvider(LLMProvider):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.systems_seen: list[str | None] = []

    def generate(self, messages: list[Message], system: str | None = None, max_tokens: int = 1000) -> str:
        self.systems_seen.append(system)
        return self.reply

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed must not run when retrieve is stubbed")


def test_grounded_answer_extracts_citations(monkeypatch):
    hits = [
        {"text": "Raise your price; compete on value not cost.", "source": "a.txt", "distance": 0.20},
        {"text": "Protect your maker time from meetings.", "source": "b.txt", "distance": 0.30},
    ]
    monkeypatch.setattr(service, "retrieve", lambda *a, **k: hits)
    fp = _FakeProvider("Raise your price [1]. Protect your time [2].")

    result = answer("how?", provider=fp, threshold=0.75)

    assert result["covered"] is True
    assert [c["n"] for c in result["citations"]] == [1, 2]
    assert {c["source"] for c in result["citations"]} == {"a.txt", "b.txt"}
    assert all(c["excerpt"] for c in result["citations"])
    assert "[1]" in result["answer"] and "[2]" in result["answer"]
    # grounded prompt carried the numbered passages
    assert "Raise your price" in (fp.systems_seen[0] or "")


def test_invalid_citation_numbers_are_stripped(monkeypatch):
    monkeypatch.setattr(
        service, "retrieve", lambda *a, **k: [{"text": "only passage", "source": "a.txt", "distance": 0.2}]
    )
    fp = _FakeProvider("Solid point [1], and also [5].")  # only 1 passage exists

    result = answer("q", provider=fp, threshold=0.75)

    assert [c["n"] for c in result["citations"]] == [1]
    assert "[1]" in result["answer"]
    assert "[5]" not in result["answer"]  # invalid marker removed


def test_fallback_has_no_citations(monkeypatch):
    monkeypatch.setattr(
        service, "retrieve", lambda *a, **k: [{"text": "x", "source": "a.txt", "distance": 0.95}]
    )
    fp = _FakeProvider("Hey, what do you want to work on?")

    result = answer("hi", provider=fp, threshold=0.75)

    assert result["covered"] is False
    assert result["citations"] == []


def test_no_hits_is_fallback(monkeypatch):
    monkeypatch.setattr(service, "retrieve", lambda *a, **k: [])
    fp = _FakeProvider("hello")

    result = answer("anything", provider=fp, threshold=0.75)

    assert result["covered"] is False
    assert result["citations"] == []
