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


def behavior_ok(kind: str, covered: bool) -> bool | None:
    """Did the system do the expected thing for this question kind?

    Returns None for kinds we don't score behaviorally (e.g. small talk).
    """
    if kind == "off_topic":
        return covered is False  # should decline
    if kind in ("covered", "novel"):
        return covered is True  # should answer/ground (novel: eventually via reasoning)
    return None


def aggregate(results: list[dict]) -> dict:
    agg: dict = {}
    for metric in _METRICS:
        vals = [r["scores"][metric] for r in results if r.get("scores") and metric in r["scores"]]
        agg[metric] = round(sum(vals) / len(vals), 2) if vals else None
    checks = [behavior_ok(r["kind"], r["covered"]) for r in results]
    checks = [c for c in checks if c is not None]
    agg["behavior_pass_rate"] = round(sum(checks) / len(checks), 2) if checks else None
    agg["n"] = len(results)
    return agg


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
                "answer": out.get("answer", ""),
                "scores": scores,
            }
        )
    return {"results": results, "aggregates": aggregate(results)}


if __name__ == "__main__":
    report = run_eval(provider=get_provider())
    agg = report["aggregates"]
    print(f"\n=== Custom Guru eval — {agg['n']} questions ===")
    print(
        f"groundedness={agg['groundedness']}  voice={agg['voice']}  "
        f"transparency={agg['transparency']}  behavior_pass_rate={agg['behavior_pass_rate']}\n"
    )
    for r in report["results"]:
        s = r["scores"]
        ok = behavior_ok(r["kind"], r["covered"])
        flag = {True: "ok", False: "MISS", None: "-"}[ok]
        print(
            f"[{flag:4}] {r['kind']:10} g={s['groundedness']} v={s['voice']} t={s['transparency']}  {r['id']}"
        )
