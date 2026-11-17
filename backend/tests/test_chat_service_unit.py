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
    monkeypatch.setattr(service, "retrieve_principles", lambda *a, **k: [])  # no principles → decline
    fp = _FakeProvider("Hey, what do you want to work on?")

    result = answer("hi", provider=fp, threshold=0.75)

    assert result["covered"] is False
    assert result["mode"] == "fallback"
    assert result["citations"] == []


def test_no_hits_is_fallback(monkeypatch):
    monkeypatch.setattr(service, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(service, "retrieve_principles", lambda *a, **k: [])
    fp = _FakeProvider("hello")

    result = answer("anything", provider=fp, threshold=0.75)

    assert result["covered"] is False
    assert result["mode"] == "fallback"
    assert result["citations"] == []


def test_reasoning_branch_when_principles_relevant(monkeypatch):
    # Not grounded (far chunk), but a relevant principle exists → reason from it.
    monkeypatch.setattr(
        service, "retrieve", lambda *a, **k: [{"text": "ctx", "source": "a.txt", "distance": 0.95}]
    )
    principles = [
        {"id": "price-ladder", "name": "Price is a ladder", "principle": "Start low, raise.",
         "rationale": "", "scope": "", "confidence": 0.9,
         "quotes": [{"text": "easy up, hard down", "source": "m.txt"}], "distance": 0.50},
    ]
    monkeypatch.setattr(service, "retrieve_principles", lambda *a, **k: principles)
    fp = _FakeProvider(
        '{"answer":"Raise it, add value.","trace":[{"factor":"new offer","principle":1,"inference":"start low then climb"}]}'
    )

    result = answer("should I launch a $5/mo app?", provider=fp, threshold=0.75)

    assert result["mode"] == "reasoned"
    assert result["covered"] is False
    assert len(result["trace"]) == 1
    assert result["trace"][0]["principle_id"] == "price-ladder"
    assert {c["name"] for c in result["citations"]} == {"Price is a ladder"}
