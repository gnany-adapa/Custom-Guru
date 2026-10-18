"""Query-time retrieval: embed the question and fetch the nearest transcript chunks."""
from __future__ import annotations

import chromadb

from backend.app.config import settings
from backend.app.llm.base import LLMProvider
from backend.app.llm.factory import get_provider


def retrieve(
    query: str,
    top_k: int | None = None,
    chroma_dir: str | None = None,
    collection_name: str | None = None,
    provider: LLMProvider | None = None,
) -> list[dict]:
    """Return up to `top_k` chunks nearest to `query`, nearest first.

    Each item: {"text": str, "source": str, "distance": float} where distance is cosine
    distance (0 = identical, higher = less related).
    """
    top_k = top_k or settings.top_k
    chroma_dir = chroma_dir or settings.chroma_dir
    collection_name = collection_name or settings.persona
    provider = provider or get_provider()

    col = chromadb.PersistentClient(path=chroma_dir).get_collection(collection_name)
    query_vec = provider.embed([query])[0]
    res = col.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": doc, "source": meta.get("source"), "distance": dist})
    return hits
