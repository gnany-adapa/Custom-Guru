"""Offline unit tests for the eval harness: aggregation, behavior check, judge parsing."""
from __future__ import annotations

from backend.app.eval.judge import judge_answer
from backend.app.eval.runner import aggregate, behavior_ok, critique_counts, run_eval
from backend.app.llm.base import LLMProvider, Message


def test_behavior_ok_by_kind():
    assert behavior_ok("off_topic", "fallback") is True
    assert behavior_ok("off_topic", "grounded") is False
    assert behavior_ok("covered", "grounded") is True
    assert behavior_ok("covered", "reasoned") is False
    assert behavior_ok("novel", "reasoned") is True
    assert behavior_ok("novel", "grounded") is False
    assert behavior_ok("small_talk", "grounded") is None  # not scored


def test_aggregate_means_and_pass_rate():
    results = [
        {"kind": "covered", "mode": "grounded", "scores": {"groundedness": 4, "voice": 2, "transparency": 4}},
        {"kind": "novel", "mode": "reasoned", "scores": {"groundedness": 2, "voice": 4, "transparency": 2}},
    ]
    agg = aggregate(results)
    assert agg["groundedness"] == 3.0
    assert agg["voice"] == 3.0
    assert agg["transparency"] == 3.0
    assert agg["behavior_pass_rate"] == 1.0
    assert agg["n"] == 2


def test_run_eval_with_fakes():
    qs = [
        {"id": "a", "kind": "covered", "question": "q1"},
        {"id": "b", "kind": "off_topic", "question": "q2"},
    ]

    def fake_answer(q):
        return {"answer": "A", "mode": "grounded" if q == "q1" else "fallback", "citations": []}

    def fake_judge(question, answer, citations, provider):
        return {"groundedness": 4, "voice": 4, "transparency": 4}

    report = run_eval(fake_answer, fake_judge, questions=qs, provider=None)
    assert len(report["results"]) == 2
    assert report["aggregates"]["groundedness"] == 4.0
    assert report["aggregates"]["behavior_pass_rate"] == 1.0  # covered→grounded, off_topic→fallback


class _FakeJudgeProvider(LLMProvider):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, messages: list[Message], system: str | None = None, max_tokens: int = 1000) -> str:
        return self.reply

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("judge does not embed")


def test_judge_answer_parses_scores():
    fp = _FakeJudgeProvider('Sure: {"groundedness": 4, "voice": 5, "transparency": 3, "note": "solid"}')
    out = judge_answer("q", "ans", [{"excerpt": "x", "source": "a.txt"}], fp)
    assert out["groundedness"] == 4
    assert out["voice"] == 5
    assert out["transparency"] == 3


def test_critique_counts_classifies_reasoned_results():
    results = [
        {"mode": "reasoned", "critique": {"checked": False, "flagged": [], "declined": False}},
        {"mode": "reasoned", "critique": {"checked": True, "flagged": ["x"], "declined": False}},
        {"mode": "reasoned", "critique": {"checked": True, "flagged": ["y"], "declined": True}},
        {"mode": "grounded", "critique": {"checked": False, "flagged": [], "declined": False}},
    ]
    assert critique_counts(results) == {"unchanged": 1, "revised": 1, "declined": 1}
