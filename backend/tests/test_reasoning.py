"""Unit tests for the reasoning engine: build a trace from referenced principles, validate
principle references (drop fabricated ones), label the answer as extrapolation. Offline."""
from __future__ import annotations

from backend.app.chat.reasoning import reason_from_principles
from backend.app.llm.base import LLMProvider, Message

_PRINCIPLES = [
    {"id": "price-ladder", "name": "Price is a ladder", "scope": "pricing", "confidence": 0.9,
     "principle": "Start price low to get flow, then raise it systematically.",
     "rationale": "Easy to go up, hard to come down.",
     "quotes": [{"text": "easy to go up in price, hard to come down", "source": "m.txt"}], "distance": 0.3},
    {"id": "value-not-price", "name": "Compete on value", "scope": "pricing", "confidence": 0.9,
     "principle": "Compete on value, not on being the cheapest.",
     "rationale": "Price wars kill margin.",
     "quotes": [{"text": "never try to be the cheapest", "source": "m.txt"}], "distance": 0.4},
]


class _FakeReasoner(LLMProvider):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.system_seen: str | None = None

    def generate(self, messages: list[Message], system: str | None = None, max_tokens: int = 1000) -> str:
        self.system_seen = system
        return self.reply

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("reasoning does not embed")


def test_reason_builds_trace_and_citations_from_referenced_principles():
    reply = (
        '{"answer":"Don\'t cut price — raise it and add value.",'
        '"trace":[{"factor":"undercutting a competitor","principle":2,"inference":"he competes on value, not price"},'
        '{"factor":"a new offer","principle":1,"inference":"start low then climb"}]}'
    )
    fp = _FakeReasoner(reply)
    out = reason_from_principles("Should I cut my price to beat a competitor?", _PRINCIPLES, [], fp)

    assert out["mode"] == "reasoned" and out["covered"] is False
    assert out["answer"].strip() != ""
    assert len(out["trace"]) == 2
    assert {c["name"] for c in out["citations"]} == {"Price is a ladder", "Compete on value"}
    assert out["trace"][0]["principle_id"] == "value-not-price"  # mapped P2 -> its id
    assert "Price is a ladder" in (fp.system_seen or "")  # prompt carried numbered principles


def test_reason_drops_fabricated_principle_refs():
    reply = '{"answer":"X","trace":[{"factor":"f","principle":5,"inference":"i"},{"factor":"g","principle":1,"inference":"j"}]}'
    out = reason_from_principles("q", _PRINCIPLES, [], _FakeReasoner(reply))
    assert len(out["trace"]) == 1  # P5 doesn't exist -> dropped
    assert out["trace"][0]["principle_id"] == "price-ladder"
    assert {c["name"] for c in out["citations"]} == {"Price is a ladder"}


def test_reason_tolerates_empty_or_garbage_output():
    out = reason_from_principles("q", _PRINCIPLES, [], _FakeReasoner("not json"))
    assert out["mode"] == "reasoned"
    assert out["trace"] == [] and out["citations"] == []
