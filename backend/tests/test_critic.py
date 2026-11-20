"""Offline unit tests for the faithfulness critic.

The LLM provider is faked (reply string injected; embed raises), so these
run with no proxy and pin down the core logic deterministically.
"""
from __future__ import annotations

import json

from backend.app.llm.base import LLMProvider, Message

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PRINCIPLES = [
    {
        "id": "price-value",
        "name": "Price for value",
        "principle": "Raise your price when you deliver more value.",
        "rationale": "Value captures profit",
        "scope": "pricing",
        "confidence": 0.9,
        "quotes": [{"text": "charge what you're worth", "source": "a.txt"}],
        "distance": 0.3,
    }
]

# All claims backed by principle 1 — faithful answer, unchanged revised_answer.
all_supported = json.dumps(
    {
        "claims": [{"claim": "Raise price.", "supported": True, "principle": 1}],
        "revised_answer": "Raise price.",
    }
)

# One supported, one unsupported — model strips the bad claim in revised_answer.
one_unsupported = json.dumps(
    {
        "claims": [
            {"claim": "Raise price.", "supported": True, "principle": 1},
            {"claim": "hire a CFO", "supported": False, "principle": None},
        ],
        "revised_answer": "Raise price.",
    }
)

# All unsupported and revised_answer is empty — not salvageable.
all_unsupported_empty_revised = json.dumps(
    {
        "claims": [{"claim": "some unsupported claim", "supported": False, "principle": None}],
        "revised_answer": "",
    }
)


# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------


class _FakeProvider(LLMProvider):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, messages: list[Message], system: str | None = None, max_tokens: int = 1000) -> str:
        return self.reply

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed must not be called in critic tests")


fp = _FakeProvider  # shorthand for test call sites


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_faithful_answer_passes_through():
    from backend.app.chat.critic import critique

    out = critique("q", "Raise price.", PRINCIPLES, fp(all_supported))
    assert out["checked"] and out["faithful"] and out["flagged"] == []
    assert out["salvageable"] and out["answer"] == "Raise price."


def test_unsupported_claim_is_flagged_and_revised():
    from backend.app.chat.critic import critique

    out = critique("q", "Raise price and hire a CFO.", PRINCIPLES, fp(one_unsupported))
    assert out["faithful"] is False
    assert "hire a CFO" in " ".join(out["flagged"])
    assert out["answer"] == "Raise price."  # revised_answer from the critic
    assert out["salvageable"] is True


def test_all_unsupported_is_unsalvageable():
    from backend.app.chat.critic import critique

    out = critique("q", "...", PRINCIPLES, fp(all_unsupported_empty_revised))
    assert out["salvageable"] is False
    assert out["faithful"] is False


def test_malformed_json_is_safe_noop():
    from backend.app.chat.critic import critique

    out = critique("q", "original", PRINCIPLES, fp("not json"))
    assert out["checked"] is False and out["faithful"] is None
    assert out["answer"] == "original" and out["salvageable"] is True


def test_claim_missing_key_does_not_raise():
    """LLM returns a claim object without the 'claim' key — must not KeyError."""
    from backend.app.chat.critic import critique

    # 'claim' key is absent; only 'supported' and 'principle' are present.
    reply = json.dumps(
        {
            "claims": [{"supported": False, "principle": None}],
            "revised_answer": "",
        }
    )
    out = critique("q", "original", PRINCIPLES, fp(reply))
    # Should not raise; the missing-key claim is silently skipped (blank text filtered).
    assert out["checked"] is True
    assert out["flagged"] == []  # blank claim string was filtered out
    assert out["salvageable"] is False  # 0 supported, empty revised


def test_braces_in_question_and_answer_do_not_raise():
    """User text containing literal { } must not blow up .format() in build_critic_prompt."""
    from backend.app.chat.critic import critique

    out = critique(
        "set fee to {floor} or {ceiling}?",
        'answer with {"json": "ish"} content',
        PRINCIPLES,
        fp(all_supported),
    )
    # Simply must not raise; correctness of the result is a bonus.
    assert out["checked"] is True
