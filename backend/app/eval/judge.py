"""LLM-as-judge scoring for eval answers.

For a no-ground-truth task we score measurable proxies, not "correctness":
  - groundedness: are the answer's claims supported by the cited evidence (vs. fabricated)?
  - voice:        does it read like the persona (direct, concrete, practical)?
  - transparency: is it honest about what's grounded vs. extrapolated?
Each is 1-5. The judge is a proxy (LLM + eyeball), not an oracle.
"""
from __future__ import annotations

import json
import re

from backend.app.llm.base import LLMProvider

_JUDGE_SYSTEM = (
    "You are a strict evaluator. You score one answer and reply with ONLY a JSON object, "
    "no prose. Keys: groundedness (1-5), voice (1-5), transparency (1-5), note (short string)."
)

_RUBRIC = """Question:
{question}

Answer being judged:
{answer}

Evidence the answer was allowed to use (cited passages):
{evidence}

Score 1-5 each:
- groundedness: are the answer's claims supported by the evidence above and not invented? \
If there is no evidence and the answer still makes confident claims, score low. If the answer \
appropriately declines or greets (no factual claims), score groundedness 5.
- voice: does it sound like a direct, concrete, practical business mentor (not generic)?
- transparency: is it honest about what is grounded vs. extrapolated / what it doesn't know?

Reply with ONLY the JSON object."""


def _parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _as_int(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def judge_answer(question: str, answer: str, citations: list[dict], provider: LLMProvider) -> dict:
    evidence = (
        "\n".join(f"- \"{c.get('excerpt', '')}\" ({c.get('source', '?')})" for c in citations)
        or "(no sources were cited)"
    )
    prompt = _RUBRIC.format(question=question, answer=answer, evidence=evidence)
    raw = provider.generate(
        messages=[{"role": "user", "content": prompt}], system=_JUDGE_SYSTEM, max_tokens=300
    )
    data = _parse_json_object(raw)
    return {
        "groundedness": _as_int(data.get("groundedness")),
        "voice": _as_int(data.get("voice")),
        "transparency": _as_int(data.get("transparency")),
        "note": str(data.get("note", "")),
    }
