"""Eval runner: run the question set through the pipeline and score it.

Run a baseline against the current chat service:
    python -m backend.app.eval.runner
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable

from backend.app.chat.service import answer as default_answer
from backend.app.eval.judge import judge_answer as default_judge
from backend.app.llm.base import LLMProvider
from backend.app.llm.factory import get_provider

_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
_METRICS = ("groundedness", "voice", "transparency")


def load_questions(path: str | None = None) -> list[dict]:
    with open(path or _QUESTIONS_PATH, encoding="utf-8") as fh:
        return json.load(fh)["questions"]


def behavior_ok(kind: str, mode: str) -> bool | None:
    """Did the router do the expected thing for this question kind?

    Scored on the routing `mode`, since the whole point is the 3-way split:
    covered questions should ground in his words, novel ones should be labeled
    extrapolation, off-topic ones should decline. Returns None for kinds we
    don't score behaviorally (e.g. small talk).
    """
    if kind == "off_topic":
        return mode == "fallback"  # should decline
    if kind == "covered":
        return mode == "grounded"  # answer from his actual words
    if kind == "novel":
        return mode == "reasoned"  # extrapolate from principles, labeled as such
    return None


def aggregate(results: list[dict]) -> dict:
    agg: dict = {}
    for metric in _METRICS:
        vals = [r["scores"][metric] for r in results if r.get("scores") and metric in r["scores"]]
        agg[metric] = round(sum(vals) / len(vals), 2) if vals else None
    checks = [behavior_ok(r["kind"], r["mode"]) for r in results]
    checks = [c for c in checks if c is not None]
    agg["behavior_pass_rate"] = round(sum(checks) / len(checks), 2) if checks else None
    agg["n"] = len(results)
    return agg


def critique_counts(results: list[dict]) -> dict:
    """Classify reasoned results by their critic outcome.

    Only reasoned-mode results are counted; grounded/fallback are ignored.
    Returns {"unchanged": int, "revised": int, "declined": int}.
    """
    unchanged = revised = declined = 0
    for r in results:
        if r.get("mode") != "reasoned":
            continue
        c = r.get("critique", {})
        if c.get("declined"):
            declined += 1
        elif c.get("flagged"):
            revised += 1
        else:
            unchanged += 1
    return {"unchanged": unchanged, "revised": revised, "declined": declined}


def run_eval(
    answer_fn: Callable[[str], dict] = default_answer,
    judge_fn: Callable = default_judge,
    questions: list[dict] | None = None,
    provider: LLMProvider | None = None,
) -> dict:
    questions = questions if questions is not None else load_questions()
    results = []
    for q in questions:
        out = answer_fn(q["question"])
        scores = judge_fn(q["question"], out.get("answer", ""), out.get("citations", []), provider)
        results.append(
            {
                "id": q["id"],
                "kind": q["kind"],
                "question": q["question"],
                "covered": out.get("covered"),
                "mode": out.get("mode", "grounded"),
                "answer": out.get("answer", ""),
                "scores": scores,
                "critique": out.get("critique", {"checked": False, "faithful": None, "flagged": [], "declined": False}),
            }
        )
    return {"results": results, "aggregates": aggregate(results)}


def compare_critic(
    questions: list[dict] | None = None,
    provider: LLMProvider | None = None,
) -> dict:
    """Run the eval twice — critic off then on — and return both aggregates plus counts.

    Returns {"off": <aggregate>, "on": <aggregate>, "counts": <critique_counts on on-results>}.
    """
    off_report = run_eval(
        answer_fn=lambda q: default_answer(q, critic=False),
        provider=provider,
        questions=questions,
    )
    on_report = run_eval(
        answer_fn=lambda q: default_answer(q, critic=True),
        provider=provider,
        questions=questions,
    )
    return {
        "off": off_report["aggregates"],
        "on": on_report["aggregates"],
        "counts": critique_counts(on_report["results"]),
    }


if __name__ == "__main__":
    provider = get_provider()
    comparison = compare_critic(provider=provider)
    off = comparison["off"]
    on = comparison["on"]
    counts = comparison["counts"]

    print(f"\n=== Custom Guru eval — critic A/B ({on['n']} questions) ===")
    print(f"{'metric':<20} {'critic OFF':>12} {'critic ON':>12}")
    print("-" * 46)
    print(f"{'groundedness':<20} {str(off['groundedness']):>12} {str(on['groundedness']):>12}")
    print(f"{'transparency':<20} {str(off['transparency']):>12} {str(on['transparency']):>12}")
    print(f"{'voice':<20} {str(off['voice']):>12} {str(on['voice']):>12}")
    print(f"{'behavior_pass_rate':<20} {str(off['behavior_pass_rate']):>12} {str(on['behavior_pass_rate']):>12}")
    print()
    print(f"Reasoned answers (critic ON): "
          f"unchanged={counts['unchanged']}  revised={counts['revised']}  declined={counts['declined']}")
    print()
