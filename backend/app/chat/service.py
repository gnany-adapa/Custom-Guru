"""Chat orchestration: retrieve grounded context, then generate the persona's answer."""
from __future__ import annotations

from backend.app.config import settings
from backend.app.llm.base import LLMProvider
from backend.app.llm.factory import get_provider
from backend.app.persona.prompts import build_fallback_prompt, build_system_prompt
from backend.app.rag.retriever import retrieve


def answer(
    question: str,
    top_k: int | None = None,
    threshold: float | None = None,
    provider: LLMProvider | None = None,
    chroma_dir: str | None = None,
    collection_name: str | None = None,
    max_tokens: int = 800,
) -> dict:
    """Answer `question` as the persona, grounded in retrieved transcript chunks.

    If the nearest chunk is farther than `threshold` (cosine distance), there's no
    relevant material: the model is instead given a fallback prompt so it can greet
    small talk warmly or decline a genuine off-topic question — without inventing advice.

    Returns {"answer": str, "covered": bool, "sources": list[str]}.
    """
    provider = provider or get_provider()
    threshold = settings.similarity_threshold if threshold is None else threshold
    hits = retrieve(
        question,
        top_k=top_k,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        provider=provider,
    )

    grounded = bool(hits) and hits[0]["distance"] <= threshold
    if grounded:
        system = build_system_prompt(hits)
        reply = provider.generate(
            messages=[{"role": "user", "content": question}],
            system=system,
            max_tokens=max_tokens,
        )
        # De-duplicate sources while preserving retrieval order.
        sources = list(dict.fromkeys(h["source"] for h in hits))
        return {"answer": reply, "covered": True, "sources": sources}

    # No relevant material: let the model greet small talk or decline, via the fallback.
    reply = provider.generate(
        messages=[{"role": "user", "content": question}],
        system=build_fallback_prompt(),
        max_tokens=300,
    )
    return {"answer": reply, "covered": False, "sources": []}
