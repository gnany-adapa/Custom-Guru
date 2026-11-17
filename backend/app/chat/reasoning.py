"""Principle-based reasoning.

Answer a question the persona never directly addressed by reasoning from their extracted
principles. Produces a validated premise→principle→conclusion trace: every step must cite a
real retrieved principle (fabricated references are dropped), and the result is labelled as
extrapolation — never presented as a quote.
"""
from __future__ import annotations

import json
import re

from backend.app.llm.base import LLMProvider
from backend.app.persona.prompts import build_reasoning_prompt


def _parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def reason_from_principles(
    question: str,
    principles: list[dict],
    cases: list[dict],
    provider: LLMProvider,
    max_tokens: int = 900,
) -> dict:
    """Reason from `principles` (and optional `cases` for voice) to answer `question`.

    Returns {"answer", "covered": False, "mode": "reasoned", "trace": [...], "citations": [...]}.
    Each trace step is {factor, principle_id, principle_name, inference}; citations are the
    principles actually referenced, with their supporting quote.
    """
    system = build_reasoning_prompt(principles, cases)
    raw = provider.generate(
        messages=[{"role": "user", "content": question}],
        system=system,
        max_tokens=max_tokens,
    )
    data = _parse_json_object(raw)
    answer = str(data.get("answer", "")).strip()

    raw_trace = data.get("trace")
    steps = raw_trace if isinstance(raw_trace, list) else []
    trace: list[dict] = []
    used_ids: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        try:
            idx = int(step.get("principle")) - 1
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(principles):
            continue  # fabricated / out-of-range principle reference → drop the step
        p = principles[idx]
        trace.append(
            {
                "factor": str(step.get("factor", "")).strip(),
                "principle_id": p["id"],
                "principle_name": p["name"],
                "inference": str(step.get("inference", "")).strip(),
            }
        )
        if p["id"] not in used_ids:
            used_ids.append(p["id"])

    by_id = {p["id"]: p for p in principles}
    citations: list[dict] = []
    for pid in used_ids:
        p = by_id[pid]
        quote = (p.get("quotes") or [{}])[0]
        citations.append(
            {
                "name": p["name"],
                "principle": p.get("principle", ""),
                "quote": quote.get("text", ""),
                "source": quote.get("source", ""),
            }
        )

    return {
        "answer": answer,
        "covered": False,
        "mode": "reasoned",
        "trace": trace,
        "citations": citations,
    }
