"""Chat orchestration: retrieve grounded context, then generate the persona's answer."""
from __future__ import annotations

import re

from backend.app.chat.reasoning import reason_from_principles
from backend.app.config import settings
from backend.app.llm.base import LLMProvider
from backend.app.llm.factory import get_provider
from backend.app.obs import observe, update_trace
from backend.app.persona.prompts import build_fallback_prompt, build_system_prompt
from backend.app.rag.retriever import retrieve, retrieve_principles

_CITE_RE = re.compile(r"\[(\d+)\]")
_EXCERPT_CHARS = 180


def _excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _apply_citations(answer_text: str, hits: list[dict]) -> tuple[str, list[dict]]:
    """Keep valid [n] markers, strip invalid ones, and build the citation list.

    A citation is only included if the model actually referenced it, so the model
    can't fabricate a source and we don't show passages it never used.
    """
    k = len(hits)
    cited: list[int] = []

    def _keep(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= k:
            if n not in cited:
                cited.append(n)
            return match.group(0)
        return ""  # invalid number -> drop the marker entirely

    cleaned = _CITE_RE.sub(_keep, answer_text)
    citations = [
        {"n": n, "source": hits[n - 1]["source"], "excerpt": _excerpt(hits[n - 1]["text"])}
        for n in sorted(cited)
    ]
    return cleaned, citations


@observe(name="chat.answer")
def answer(
    question: str,
    top_k: int | None = None,
    threshold: float | None = None,
    reasoning_threshold: float | None = None,
    provider: LLMProvider | None = None,
    chroma_dir: str | None = None,
    collection_name: str | None = None,
    max_tokens: int = 800,
) -> dict:
    """Answer `question` as the persona, grounded in retrieved transcript chunks.

    If the nearest chunk is farther than `threshold` (cosine distance), there's no
    relevant material: the model is given a fallback prompt so it can greet small talk
    warmly or decline a genuine off-topic question — without inventing advice.

    Returns {"answer": str, "covered": bool, "mode": str, "citations": list[dict],
    "trace": list[dict]}. `mode` is one of "grounded" | "reasoned" | "fallback".
    Grounded answers carry citations of {"n": int, "source": str, "excerpt": str} and an
    empty `trace`; reasoned answers carry a premise→principle→conclusion `trace` and
    principle-based citations (see reason_from_principles); fallback answers have empty
    `citations` and `trace`.
    """
    provider = provider or get_provider()
    threshold = settings.similarity_threshold if threshold is None else threshold
    reasoning_threshold = (
        settings.reasoning_threshold if reasoning_threshold is None else reasoning_threshold
    )
    hits = retrieve(
        question,
        top_k=top_k,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        provider=provider,
    )

    grounded = bool(hits) and hits[0]["distance"] <= threshold
    update_trace(
        metadata={
            "grounded": grounded,
            "threshold": threshold,
            "retrieved": [
                {"source": h["source"], "distance": round(h["distance"], 3)} for h in hits
            ],
        }
    )
    if grounded:
        reply = provider.generate(
            messages=[{"role": "user", "content": question}],
            system=build_system_prompt(hits),
            max_tokens=max_tokens,
        )
        answer_text, citations = _apply_citations(reply, hits)
        return {
            "answer": answer_text, "covered": True, "mode": "grounded",
            "citations": citations, "trace": [],
        }

    # Not directly covered — try reasoning from the persona's principles before declining.
    persona = collection_name or settings.persona
    principles = retrieve_principles(
        question, chroma_dir=chroma_dir, persona=persona, provider=provider
    )
    if principles and principles[0]["distance"] <= reasoning_threshold:
        update_trace(metadata={"mode": "reasoned", "principles": [p["id"] for p in principles[:3]]})
        # Nearest chunks give the model the persona's voice, even if beyond the grounding bar.
        return reason_from_principles(question, principles, hits[:2], provider)

    # Nothing relevant even among principles: greet small talk or decline.
    reply = provider.generate(
        messages=[{"role": "user", "content": question}],
        system=build_fallback_prompt(),
        max_tokens=300,
    )
    return {"answer": reply, "covered": False, "mode": "fallback", "citations": [], "trace": []}
