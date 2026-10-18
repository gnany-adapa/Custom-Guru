"""Offline unit tests for the chat service's grounded-vs-fallback decision.

Retrieval is stubbed and the provider is faked, so these run with no proxy or vector
store and nail down the core branch logic deterministically (fast, network-free).
"""
from __future__ import annotations

import backend.app.chat.service as service
from backend.app.chat.service import answer
from backend.app.llm.base import LLMProvider, Message


class _FakeProvider(LLMProvider):
    def __init__(self) -> None:
        self.systems_seen: list[str | None] = []

    def generate(self, messages: list[Message], system: str | None = None, max_tokens: int = 1000) -> str:
        self.systems_seen.append(system)
        return "REPLY"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed must not run when retrieve is stubbed")


def test_grounded_branch_when_match_is_close(monkeypatch):
    hits = [
        {"text": "chunk about pricing", "source": "a.txt", "distance": 0.20},
        {"text": "more pricing", "source": "a.txt", "distance": 0.30},
    ]
    monkeypatch.setattr(service, "retrieve", lambda *a, **k: hits)
    fp = _FakeProvider()

    result = answer("how do I price?", provider=fp, threshold=0.75)

    assert result["covered"] is True
    assert result["sources"] == ["a.txt"]  # de-duplicated
    assert "chunk about pricing" in (fp.systems_seen[0] or "")  # grounded prompt has context


def test_fallback_branch_when_match_is_far(monkeypatch):
    monkeypatch.setattr(
        service, "retrieve", lambda *a, **k: [{"text": "x", "source": "a.txt", "distance": 0.92}]
    )
    fp = _FakeProvider()

    result = answer("pasta recipe?", provider=fp, threshold=0.75)

    assert result["covered"] is False
    assert result["sources"] == []
    assert "no relevant material" in (fp.systems_seen[0] or "").lower()  # fallback prompt


def test_fallback_branch_when_no_hits(monkeypatch):
    monkeypatch.setattr(service, "retrieve", lambda *a, **k: [])
    fp = _FakeProvider()

    result = answer("anything", provider=fp, threshold=0.75)

    assert result["covered"] is False
    assert result["sources"] == []
