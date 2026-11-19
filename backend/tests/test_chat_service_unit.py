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

    result = answer("should I launch a $5/mo app?", provider=fp, threshold=0.75, critic=False)

    assert result["mode"] == "reasoned"
    assert result["covered"] is False
    assert len(result["trace"]) == 1
    assert result["trace"][0]["principle_id"] == "price-ladder"
    assert {c["name"] for c in result["citations"]} == {"Price is a ladder"}


def test_reasoning_threshold_is_overridable_per_call(monkeypatch):
    # The reasoning bar is a per-call argument (like the grounding `threshold`), so routing
    # is decided independently of the global default. Same principle at distance 0.50:
    monkeypatch.setattr(
        service, "retrieve", lambda *a, **k: [{"text": "ctx", "source": "a.txt", "distance": 0.95}]
    )
    principles = [
        {"id": "price-ladder", "name": "Price is a ladder", "principle": "Start low, raise.",
         "rationale": "", "scope": "", "confidence": 0.9,
         "quotes": [{"text": "easy up, hard down", "source": "m.txt"}], "distance": 0.50},
    ]
    monkeypatch.setattr(service, "retrieve_principles", lambda *a, **k: principles)
    fp = _FakeProvider('{"answer":"Raise it.","trace":[{"factor":"x","principle":1,"inference":"y"}]}')

    # Bar below the principle's distance → not relevant enough → decline.
    strict = answer("q", provider=fp, threshold=0.75, reasoning_threshold=0.40)
    assert strict["mode"] == "fallback"

    # Bar above it → reason from the principle.
    lenient = answer("q", provider=fp, threshold=0.75, reasoning_threshold=0.60)
    assert lenient["mode"] == "reasoned"


# ---------------------------------------------------------------------------
# Faithfulness critic wiring tests (Task 2)
# ---------------------------------------------------------------------------

_PRINCIPLES = [
    {
        "id": "price-ladder",
        "name": "Price is a ladder",
        "principle": "Start low, raise.",
        "rationale": "",
        "scope": "",
        "confidence": 0.9,
        "quotes": [{"text": "easy up, hard down", "source": "m.txt"}],
        "distance": 0.50,
    }
]
_FAR_HIT = [{"text": "ctx", "source": "a.txt", "distance": 0.95}]
_REASONING_REPLY = (
    '{"answer":"Raise it, add value.",'
    '"trace":[{"factor":"new offer","principle":1,"inference":"start low then climb"}]}'
)
_REVISED_TEXT = "Raise it carefully."


def _stub_reasoned(monkeypatch, fp):
    """Shared setup for reasoned-path tests."""
    monkeypatch.setattr(service, "retrieve", lambda *a, **k: _FAR_HIT)
    monkeypatch.setattr(service, "retrieve_principles", lambda *a, **k: _PRINCIPLES)


def test_reasoned_result_carries_critique(monkeypatch):
    """Critic finds all claims supported → faithful, not declined, answer kept as-is."""
    fp = _FakeProvider(_REASONING_REPLY)
    _stub_reasoned(monkeypatch, fp)
    monkeypatch.setattr(
        service,
        "critique",
        lambda q, a, p, prov, **kw: {
            "checked": True,
            "faithful": True,
            "flagged": [],
            "answer": "Raise it, add value.",
            "salvageable": True,
        },
    )

    result = answer("should I raise my price?", provider=fp, threshold=0.75)

    assert result["mode"] == "reasoned"
    assert result["critique"] == {"checked": True, "faithful": True, "flagged": [], "declined": False}


def test_unsupported_claim_is_revised_in_reasoned_answer(monkeypatch):
    """Critic flags a claim but answer is salvageable → revised text, not declined."""
    fp = _FakeProvider(_REASONING_REPLY)
    _stub_reasoned(monkeypatch, fp)
    monkeypatch.setattr(
        service,
        "critique",
        lambda q, a, p, prov, **kw: {
            "checked": True,
            "faithful": False,
            "flagged": ["some unsupported claim"],
            "answer": _REVISED_TEXT,
            "salvageable": True,
        },
    )

    result = answer("should I raise my price?", provider=fp, threshold=0.75)

    assert result["critique"]["flagged"]  # non-empty
    assert result["critique"]["declined"] is False
    assert result["answer"] == _REVISED_TEXT


def test_unsalvageable_reasoned_answer_declines_but_stays_reasoned(monkeypatch):
    """Critic marks unsalvageable → mode stays 'reasoned', answer replaced, trace/citations cleared."""
    fp = _FakeProvider(_REASONING_REPLY)
    _stub_reasoned(monkeypatch, fp)
    monkeypatch.setattr(
        service,
        "critique",
        lambda q, a, p, prov, **kw: {
            "checked": True,
            "faithful": False,
            "flagged": ["bad claim 1", "bad claim 2"],
            "answer": "original",
            "salvageable": False,
        },
    )

    result = answer("should I raise my price?", provider=fp, threshold=0.75)

    assert result["mode"] == "reasoned"  # NOT fallback — routing was correct
    assert result["covered"] is False
    assert result["critique"]["declined"] is True
    assert result["critique"]["faithful"] is False
    assert result["answer"] == service._CRITIC_DECLINE
    assert result["citations"] == [] and result["trace"] == []


def test_critic_disabled_leaves_answer_unchanged(monkeypatch):
    """Passing critic=False keeps the reasoned answer and attaches the default no-op critique."""
    fp = _FakeProvider(_REASONING_REPLY)
    _stub_reasoned(monkeypatch, fp)

    result = answer("q", provider=fp, threshold=0.75, critic=False)

    assert result["critique"] == {"checked": False, "faithful": None, "flagged": [], "declined": False}


def test_all_claims_supported_does_not_swap_answer(monkeypatch):
    """When critic returns flagged=[] (all claims supported), the original reasoned answer is kept.

    The revised critic answer ("SOMETHING ELSE") must NOT replace the original even though
    the salvageable branch ran — an empty flagged list means nothing needed to change.
    """
    fp = _FakeProvider(_REASONING_REPLY)
    _stub_reasoned(monkeypatch, fp)
    monkeypatch.setattr(
        service,
        "critique",
        lambda q, a, p, prov, **kw: {
            "checked": True,
            "faithful": True,
            "flagged": [],
            "answer": "SOMETHING ELSE",
            "salvageable": True,
        },
    )

    result = answer("should I raise my price?", provider=fp, threshold=0.75)

    # The original reasoned answer must be delivered unchanged.
    assert result["answer"] == "Raise it, add value."
    assert result["answer"] != "SOMETHING ELSE"
    assert result["critique"]["flagged"] == []
    assert result["critique"]["declined"] is False


def test_grounded_and_fallback_carry_uniform_critique_default(monkeypatch):
    """Both grounded and fallback paths attach the default (checked=False) critique dict."""
    # Grounded path
    hits = [{"text": "value advice", "source": "a.txt", "distance": 0.20}]
    monkeypatch.setattr(service, "retrieve", lambda *a, **k: hits)
    fp = _FakeProvider("Value answer.")
    grounded = answer("how?", provider=fp, threshold=0.75)

    # Fallback path
    monkeypatch.setattr(
        service, "retrieve", lambda *a, **k: [{"text": "x", "source": "a.txt", "distance": 0.95}]
    )
    monkeypatch.setattr(service, "retrieve_principles", lambda *a, **k: [])
    fallback = answer("hi", provider=fp, threshold=0.75)

    assert grounded["critique"]["checked"] is False
    assert fallback["critique"]["checked"] is False
