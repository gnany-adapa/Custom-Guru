"""Chat orchestration: retrieve grounded context, then generate the persona's answer."""
from __future__ import annotations

from backend.app.config import settings
from backend.app.llm.base import LLMProvider
from backend.app.llm.factory import get_provider
from backend.app.persona.prompts import build_system_prompt
from backend.app.rag.retriever import retrieve

# Returned verbatim (in persona voice) when nothing relevant is retrieved, so the avatar
# declines instead of inventing advice.
_NOT_COVERED = (
    "Honestly, I haven't talked about that one. I'm not going to make something up just "
    "to give you an answer — that's how people end up with bad advice. Ask me about "
    "something I've actually covered and I'll give it to you straight."
)


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

    If the nearest chunk is farther than `threshold` (cosine distance), the question is
    treated as uncovered: the avatar declines and no LLM call is made.

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

    # Grounded guard: refuse when nothing retrieved is close enough.
    if not hits or hits[0]["distance"] > threshold:
        return {"answer": _NOT_COVERED, "covered": False, "sources": []}

    system = build_system_prompt(hits)
    reply = provider.generate(
        messages=[{"role": "user", "content": question}],
        system=system,
        max_tokens=max_tokens,
    )
    # De-duplicate sources while preserving retrieval order.
    sources = list(dict.fromkeys(h["source"] for h in hits))
    return {"answer": reply, "covered": True, "sources": sources}
