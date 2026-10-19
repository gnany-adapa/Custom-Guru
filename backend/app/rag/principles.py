"""Principle extraction: transcripts -> durable decision principles (quote-verified).

Two-stage map->consolidate pipeline:
  1. MAP: over word-windows, an LLM proposes candidate principles, each with a verbatim
     supporting quote. We DROP any candidate whose quote isn't actually in the source
     (anti-hallucination), mirroring the citation discipline in the chat path.
  2. CONSOLIDATE: embed candidates, cluster near-duplicates, and merge each cluster into
     one canonical principle that keeps all its supporting quotes.

Output: an inspectable `data/personas/<persona>/principles.json` + a Chroma collection
`<persona>_principles` (the statement is embedded for retrieval).

Run:  python -m backend.app.rag.principles
"""
from __future__ import annotations

import glob
import json
import math
import os
import re

import chromadb

from backend.app.config import settings
from backend.app.llm.base import LLMProvider
from backend.app.llm.factory import get_provider
from backend.app.rag.ingest import chunk_text, clean_text

_WINDOW_WORDS = 800
_WINDOW_OVERLAP = 100
_GROUP_THRESHOLD = 0.72  # wide net: loosely bucket possible duplicates; the LLM then splits
_CORE_TARGET = 30  # final compression target — keep only the most important durable principles
_EMBED_BATCH = 64
_MAX_QUOTES = 6

# --------------------------------------------------------------------------- pure helpers
def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "principle"


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for tolerant quote matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _quote_supported(quote: str, source_text: str) -> bool:
    """True if the (normalized) quote actually appears in the (normalized) source."""
    nq = _normalize(quote)
    if len(nq) < 12:  # too short to be a meaningful, verifiable quote
        return False
    return nq in _normalize(source_text)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _cluster(vectors: list[list[float]], threshold: float) -> list[list[int]]:
    """Greedy clustering: each vector joins the first cluster it's similar enough to."""
    clusters: list[dict] = []
    for i, v in enumerate(vectors):
        for cl in clusters:
            if _cosine(v, cl["rep"]) >= threshold:
                cl["members"].append(i)
                break
        else:
            clusters.append({"rep": v, "members": [i]})
    return [cl["members"] for cl in clusters]


def _parse_json_array(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------- extraction (map)
_EXTRACT_SYSTEM = (
    "You extract a speaker's DURABLE, HIGH-LEVEL decision principles from a transcript "
    "excerpt — the core mental models they apply across many situations — NOT specific "
    "tactics, tips, facts, or one-off examples. Reply with ONLY a JSON array (no prose)."
)

_EXTRACT_PROMPT = """From the excerpt below, extract AT MOST 2 durable, high-level \
principles the speaker holds. A principle is a general rule they'd apply across many \
different situations (e.g. "compete on value, not price"). It is NOT a narrow tactic \
("write at a third-grade reading level"), a metric, or a fact ("I sold my company for \
$46M"). Prefer fewer, bigger principles; if nothing rises to that level, return [].

For each, output an object:
{{"name": short label, "principle": one-sentence imperative rule, "rationale": why it \
holds (short), "scope": when it applies (short), "confidence": 0.0-1.0, "quote": a VERBATIM \
span copied exactly from the excerpt that supports it}}

Rules: the quote MUST be copied word-for-word from the excerpt. Reply with ONLY the JSON array.

EXCERPT:
{text}"""


def extract_candidates(window_text: str, source: str, provider: LLMProvider) -> list[dict]:
    """Propose candidate principles for one window; drop any whose quote isn't verifiable."""
    raw = provider.generate(
        messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(text=window_text)}],
        system=_EXTRACT_SYSTEM,
        max_tokens=1200,
    )
    candidates = []
    for item in _parse_json_array(raw):
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote", "")).strip()
        principle = str(item.get("principle", "")).strip()
        if not principle or not _quote_supported(quote, window_text):
            continue
        candidates.append(
            {
                "name": str(item.get("name", principle[:40])).strip(),
                "principle": principle,
                "rationale": str(item.get("rationale", "")).strip(),
                "scope": str(item.get("scope", "")).strip(),
                "confidence": _as_float(item.get("confidence"), 0.6),
                "quote": quote,
                "source": source,
            }
        )
    return candidates


def _as_float(value, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- consolidate (reduce)
_RESOLVE_SYSTEM = (
    "You decide which candidate principles express the SAME underlying rule and merge only "
    "those. Keep genuinely different rules separate. Reply with ONLY a JSON array."
)
_RESOLVE_PROMPT = """Here are candidate principles — some are the same rule worded \
differently, some are genuinely different:

{items}

Group ONLY the ones that mean the same rule; keep distinct rules separate. Output a JSON \
array where each element is one canonical principle:
{{"name": short label, "principle": one-sentence imperative rule, "rationale": short, \
"scope": short, "confidence": 0.0-1.0, "members": [the numbers of the candidates merged here]}}
Every candidate number must appear in exactly one group. Reply with ONLY the JSON array."""


def _embed_statements(provider: LLMProvider, candidates: list[dict]) -> list[list[float]]:
    statements = [c["principle"] for c in candidates]
    vectors: list[list[float]] = []
    for i in range(0, len(statements), _EMBED_BATCH):
        vectors.extend(provider.embed(statements[i : i + _EMBED_BATCH]))
    return vectors


def _dedup_quotes(cands: list[dict]) -> list[dict]:
    seen, quotes = set(), []
    for c in cands:
        key = _normalize(c["quote"])
        if key and key not in seen:
            seen.add(key)
            quotes.append({"text": c["quote"], "source": c["source"]})
    return quotes


def _canonical_from_single(c: dict) -> dict:
    return {
        "name": c["name"],
        "principle": c["principle"],
        "rationale": c["rationale"],
        "scope": c["scope"],
        "confidence": c["confidence"],
        "quotes": _dedup_quotes([c]),
    }


def _resolve_cluster(cands: list[dict], provider: LLMProvider) -> list[dict]:
    """Within a loosely-grouped bucket, let the LLM merge same-meaning candidates and keep
    distinct ones separate. Returns one OR MORE canonical principles (the AI re-splits the
    over-eager math grouping). Any candidate the model drops is recovered as its own
    principle, so no principle is ever silently lost.
    """
    if len(cands) == 1:
        return [_canonical_from_single(cands[0])]

    listing = "\n".join(
        f'{i}. {c["principle"]} (quote: "{c["quote"]}")' for i, c in enumerate(cands)
    )
    raw = provider.generate(
        messages=[{"role": "user", "content": _RESOLVE_PROMPT.format(items=listing)}],
        system=_RESOLVE_SYSTEM,
        max_tokens=800,
    )

    resolved, assigned = [], set()
    for group in _parse_json_array(raw):
        if not isinstance(group, dict):
            continue
        members = [
            m
            for m in group.get("members", [])
            if isinstance(m, int) and 0 <= m < len(cands) and m not in assigned
        ]
        if not members:
            continue
        assigned.update(members)
        resolved.append(
            {
                "name": str(group.get("name") or cands[members[0]]["name"]).strip(),
                "principle": str(group.get("principle") or cands[members[0]]["principle"]).strip(),
                "rationale": str(group.get("rationale", "")).strip(),
                "scope": str(group.get("scope", "")).strip(),
                "confidence": _as_float(group.get("confidence"), 0.7),
                "quotes": _dedup_quotes([cands[m] for m in members]),
            }
        )
    # Safety net: candidates the model forgot become their own principle (no data loss).
    for i, c in enumerate(cands):
        if i not in assigned:
            resolved.append(_canonical_from_single(c))
    return resolved


def consolidate(candidates: list[dict], provider: LLMProvider) -> list[dict]:
    if not candidates:
        return []
    vectors = _embed_statements(provider, candidates)
    buckets = _cluster(vectors, _GROUP_THRESHOLD)  # math: loose grouping (wide net)

    principles, used_ids = [], set()
    for members in buckets:
        for record in _resolve_cluster([candidates[i] for i in members], provider):  # AI: split/merge
            pid = _slug(record["name"])
            while pid in used_ids:
                pid += "-x"
            used_ids.add(pid)
            record["id"] = pid
            principles.append(record)
    # Most-supported (and most-confident) principles first.
    principles.sort(key=lambda p: (len(p["quotes"]), p["confidence"]), reverse=True)
    return principles


_COMPRESS_SYSTEM = (
    "You distill a long list of principles down to a person's core, durable mental models. "
    "Merge overlapping or granular ones into the broader principle they belong to, and drop "
    "trivial one-off tactics. Reply with ONLY a JSON array."
)
_COMPRESS_PROMPT = """From the {count} candidate principles below, produce AT MOST {target} \
of the MOST important, durable, high-level principles this person holds. Merge overlapping \
or narrow ones into the broader principle they belong to; drop trivial tactics entirely.

{items}

Output a JSON array; each element is one core principle:
{{"name": short label, "principle": one-sentence imperative rule, "rationale": short, \
"scope": short, "confidence": 0.0-1.0, "members": [the numbers of the candidates folded into this]}}
Aim for {target} or fewer. Reply with ONLY the JSON array."""


def _collect_quotes(principles: list[dict], members: list[int]) -> list[dict]:
    seen, quotes = set(), []
    for m in members:
        for q in principles[m]["quotes"]:
            key = _normalize(q["text"])
            if key and key not in seen:
                seen.add(key)
                quotes.append(q)
    return quotes[:_MAX_QUOTES]


def _assign_ids(records: list[dict]) -> list[dict]:
    used: set[str] = set()
    for rec in records:
        pid = _slug(rec["name"])
        while pid in used:
            pid += "-x"
        used.add(pid)
        rec["id"] = pid
    return records


def compress_to_core(principles: list[dict], provider: LLMProvider, target: int = _CORE_TARGET) -> list[dict]:
    """Reduce a large principle list to the ~target most important, durable principles.

    Folds overlapping/granular principles into broader ones and drops trivial tactics. Falls
    back to a deterministic top-`target` (by support then confidence) if the LLM output is
    unusable, so we always return a bounded set.
    """
    if len(principles) <= target:
        return principles

    listing = "\n".join(f"{i}. {p['name']}: {p['principle']}" for i, p in enumerate(principles))
    raw = provider.generate(
        messages=[{"role": "user", "content": _COMPRESS_PROMPT.format(
            count=len(principles), target=target, items=listing)}],
        system=_COMPRESS_SYSTEM,
        max_tokens=4000,
    )

    core = []
    for group in _parse_json_array(raw):
        if not isinstance(group, dict):
            continue
        members = [
            m for m in group.get("members", []) if isinstance(m, int) and 0 <= m < len(principles)
        ]
        if not members:
            continue
        core.append(
            {
                "name": str(group.get("name") or principles[members[0]]["name"]).strip(),
                "principle": str(group.get("principle") or principles[members[0]]["principle"]).strip(),
                "rationale": str(group.get("rationale", "")).strip(),
                "scope": str(group.get("scope", "")).strip(),
                "confidence": _as_float(
                    group.get("confidence"), max(principles[m]["confidence"] for m in members)
                ),
                "quotes": _collect_quotes(principles, members),
            }
        )

    if not core:  # LLM output unusable → deterministic fallback
        core = sorted(principles, key=lambda p: (len(p["quotes"]), p["confidence"]), reverse=True)[:target]

    core.sort(key=lambda p: (len(p["quotes"]), p["confidence"]), reverse=True)
    return _assign_ids(core)


# --------------------------------------------------------------------------- build the store
def build_principle_store(
    transcripts_dir: str | None = None,
    persona: str | None = None,
    chroma_dir: str | None = None,
    provider: LLMProvider | None = None,
) -> list[dict]:
    transcripts_dir = transcripts_dir or settings.transcripts_dir
    persona = persona or settings.persona
    chroma_dir = chroma_dir or settings.chroma_dir
    provider = provider or get_provider()

    paths = sorted(glob.glob(os.path.join(transcripts_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt transcripts found in {transcripts_dir}")

    candidates: list[dict] = []
    for path in paths:
        source = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            text = clean_text(fh.read())
        for window in chunk_text(text, _WINDOW_WORDS, _WINDOW_OVERLAP):
            candidates.extend(extract_candidates(window, source, provider))

    candidates_merged = consolidate(candidates, provider)  # dedup near-duplicates
    principles = compress_to_core(candidates_merged, provider)  # keep the durable core

    # Write the inspectable JSON (persona data stays under gitignored data/).
    out_path = os.path.join(transcripts_dir, "..", "principles.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"persona": persona, "principles": principles}, fh, indent=2)

    # Build the retrieval collection (embed the statement).
    if principles:
        client = chromadb.PersistentClient(path=chroma_dir)
        name = f"{persona}_principles"
        try:
            client.delete_collection(name)
        except Exception:
            pass
        col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=[p["id"] for p in principles],
            embeddings=provider.embed([p["principle"] for p in principles]),
            documents=[p["principle"] for p in principles],
            metadatas=[
                {
                    "id": p["id"],
                    "name": p["name"],
                    "principle": p["principle"],
                    "rationale": p["rationale"],
                    "scope": p["scope"],
                    "confidence": p["confidence"],
                    "quotes": json.dumps(p["quotes"]),
                }
                for p in principles
            ],
        )
    return principles


if __name__ == "__main__":
    ps = build_principle_store()
    print(f"\nExtracted {len(ps)} principles for '{settings.persona}':\n")
    for p in ps:
        print(f"  [{len(p['quotes'])}q, conf {p['confidence']:.2f}] {p['name']}: {p['principle']}")
