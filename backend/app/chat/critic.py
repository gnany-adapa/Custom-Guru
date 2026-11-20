"""Faithfulness critic.

Checks a reasoned answer against the principles it was built from. Each substantive
claim in the answer is verified: if no listed principle supports it the claim is
flagged, and the model returns a revised answer that keeps only supported content.

The critic is a single LLM call; it never raises on bad model output — unparseable
responses produce a safe no-op result.
"""
from __future__ import annotations

import json
import re

from backend.app.llm.base import LLMProvider
from backend.app.persona.prompts import build_critic_prompt


def _parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def critique(
    question: str,
    answer: str,
    principles: list[dict],
    provider: LLMProvider,
    *,
    min_supported: int = 1,
) -> dict:
    """Check `answer` against `principles` using `provider`.

    Returns:
        {
            "checked": bool,           # False on unparseable model output
            "faithful": bool | None,   # None when checked is False
            "flagged": list[str],      # claim texts that are not principle-backed
            "answer": str,             # revised answer (or original on no-op / unsalvageable)
            "salvageable": bool,       # True when at least min_supported claims hold up
        }
    """
    system = build_critic_prompt(question, answer, principles)
    raw = provider.generate(
        messages=[{"role": "user", "content": question}],
        system=system,
    )

    data = _parse_json_object(raw)
    if not data:
        return {
            "checked": False,
            "faithful": None,
            "flagged": [],
            "answer": answer,
            "salvageable": True,
        }

    claims = data.get("claims")
    if not isinstance(claims, list):
        claims = []

    flagged: list[str] = [
        text
        for c in claims
        if isinstance(c, dict) and not c.get("supported")
        for text in (str(c.get("claim", "")),)
        if text  # skip blank claim strings so we don't flag empty noise
    ]
    supported_count: int = sum(
        1 for c in claims if isinstance(c, dict) and c.get("supported")
    )

    revised = str(data.get("revised_answer", "")).strip()

    faithful = len(flagged) == 0
    salvageable = supported_count >= min_supported and bool(revised)
    final_answer = revised if salvageable else answer

    return {
        "checked": True,
        "faithful": faithful,
        "flagged": flagged,
        "answer": final_answer,
        "salvageable": salvageable,
    }
