"""Unit tests for principle extraction helpers + the quote-verification safeguard.

Pure logic and the candidate filter run offline (fake provider); the full build is an
integration run (python -m backend.app.rag.principles), not a unit test.
"""
from __future__ import annotations

from backend.app.llm.base import LLMProvider, Message
from backend.app.rag.principles import (
    _cluster,
    _cosine,
    _parse_json_array,
    _parse_json_object,
    _quote_supported,
    _resolve_cluster,
    _slug,
    compress_to_core,
    extract_candidates,
)


def test_slug():
    assert _slug("Price is a Ladder!") == "price-is-a-ladder"
    assert _slug("   ") == "principle"


def test_quote_supported_is_tolerant_but_real():
    source = "so we compete on value not on price, because margins matter a lot"
    assert _quote_supported("Compete on Value, not on Price!", source) is True  # case/punct tolerant
    assert _quote_supported("always raise prices every quarter", source) is False  # not present
    assert _quote_supported("price", source) is False  # too short to verify


def test_cosine_and_cluster():
    assert round(_cosine([1, 0], [1, 0]), 3) == 1.0
    assert round(_cosine([1, 0], [0, 1]), 3) == 0.0
    clusters = _cluster([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], threshold=0.86)
    assert len(clusters) == 2
    assert [0, 1] in clusters  # the two near-identical vectors group together


def test_json_parsers_are_defensive():
    assert _parse_json_array('noise [{"a":1}] trailing') == [{"a": 1}]
    assert _parse_json_array("not json") == []
    assert _parse_json_object('```\n{"a": 2}\n```') == {"a": 2}
    assert _parse_json_object("nope") == {}


class _FakeExtractProvider(LLMProvider):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, messages: list[Message], system: str | None = None, max_tokens: int = 1000) -> str:
        return self.reply

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("extraction does not embed")


def test_extract_candidates_drops_unverifiable_quote():
    window = "we compete on value not on price because a price war kills your margins"
    reply = (
        '[{"name":"Compete on value","principle":"Compete on value, not price.",'
        '"rationale":"margins","scope":"pricing","confidence":0.9,'
        '"quote":"compete on value not on price"},'
        '{"name":"Raise quarterly","principle":"Raise prices every quarter.",'
        '"rationale":"x","scope":"x","confidence":0.5,'
        '"quote":"always raise prices every quarter"}]'
    )
    out = extract_candidates(window, "talk.txt", _FakeExtractProvider(reply))
    assert len(out) == 1  # the fabricated-quote candidate is dropped
    assert out[0]["name"] == "Compete on value"
    assert out[0]["source"] == "talk.txt"


class _FakeResolveProvider(LLMProvider):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.generate_called = False

    def generate(self, messages: list[Message], system: str | None = None, max_tokens: int = 1000) -> str:
        self.generate_called = True
        return self.reply

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("resolve does not embed")


def _cand(name, principle, quote):
    return {
        "name": name, "principle": principle, "rationale": "", "scope": "",
        "confidence": 0.9, "quote": quote, "source": "t.txt",
    }


def test_resolve_cluster_merges_same_keeps_distinct():
    cands = [
        _cand("Batch A", "Batch meetings onto one day.", "batch meetings onto one day"),
        _cand("Batch B", "Cluster meetings to protect flow.", "cluster meetings to protect flow"),
        _cand("Delete", "Delete meetings entirely.", "delete meetings entirely please"),
    ]
    reply = (
        '[{"name":"Batch meetings","principle":"Batch meetings to protect focus.",'
        '"members":[0,1]},'
        '{"name":"Delete meetings","principle":"Delete meetings entirely.","members":[2]}]'
    )
    out = _resolve_cluster(cands, _FakeResolveProvider(reply))
    assert len(out) == 2
    batch = next(p for p in out if p["name"] == "Batch meetings")
    assert len(batch["quotes"]) == 2  # the two same-meaning candidates merged
    delete = next(p for p in out if "Delete" in p["name"])
    assert len(delete["quotes"]) == 1  # the distinct one stayed separate


def test_resolve_cluster_singleton_skips_ai():
    prov = _FakeResolveProvider("SHOULD NOT BE CALLED")
    out = _resolve_cluster([_cand("X", "Some rule.", "a long enough verbatim quote")], prov)
    assert len(out) == 1
    assert prov.generate_called is False  # no LLM call for a lone candidate


def test_resolve_cluster_recovers_dropped_candidate():
    cands = [
        _cand("A", "Rule A.", "quote a long enough here"),
        _cand("B", "Rule B.", "quote b long enough here"),
    ]
    reply = '[{"name":"A","principle":"Rule A.","members":[0]}]'  # model forgot candidate 1
    out = _resolve_cluster(cands, _FakeResolveProvider(reply))
    assert len(out) == 2  # B recovered as its own principle, not lost


def _principle(name, n_quotes, confidence):
    return {
        "name": name, "principle": f"{name} rule.", "rationale": "", "scope": "",
        "confidence": confidence,
        "quotes": [{"text": f"{name} quote {i}", "source": "t.txt"} for i in range(n_quotes)],
        "id": _slug(name),
    }


def test_compress_noop_when_within_target():
    ps = [_principle("A", 1, 0.9), _principle("B", 1, 0.8)]
    prov = _FakeResolveProvider("x")
    out = compress_to_core(ps, prov, target=5)
    assert out == ps
    assert prov.generate_called is False  # no LLM call when already small enough


def test_compress_merges_and_prunes():
    ps = [_principle("A", 1, 0.9), _principle("B", 1, 0.8), _principle("C", 1, 0.7), _principle("D", 1, 0.6)]
    reply = (
        '[{"name":"Core1","principle":"Core one.","members":[0,1]},'
        '{"name":"Core2","principle":"Core two.","members":[2,3]}]'
    )
    out = compress_to_core(ps, _FakeResolveProvider(reply), target=2)
    assert len(out) == 2
    core1 = next(p for p in out if p["name"] == "Core1")
    assert len(core1["quotes"]) == 2  # A + B quotes folded in
    assert all("id" in p for p in out)


def test_compress_falls_back_to_top_n_on_garbage():
    ps = [_principle("A", 3, 0.9), _principle("B", 1, 0.8), _principle("C", 2, 0.7)]
    out = compress_to_core(ps, _FakeResolveProvider("not json at all"), target=2)
    assert len(out) == 2
    assert {p["name"] for p in out} == {"A", "C"}  # top-2 by quote count
