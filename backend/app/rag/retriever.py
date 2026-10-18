"""Query-time retrieval: embed the question and fetch the nearest transcript chunks."""
from __future__ import annotations

import chromadb

from backend.app.config import settings
from backend.app.llm.base import LLMProvider
from backend.app.llm.factory import get_provider


class VectorStoreNotFoundError(RuntimeError):
    """Raised when the persona's Chroma collection hasn't been built yet."""


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

    Raises VectorStoreNotFoundError if the collection doesn't exist yet (e.g. ingest
    hasn't been run).
    """
    top_k = top_k or settings.top_k
    chroma_dir = chroma_dir or settings.chroma_dir
    collection_name = collection_name or settings.persona
    provider = provider or get_provider()

    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        col = client.get_collection(collection_name)
    except Exception as exc:  # chroma raises a few different types for "missing"
        raise VectorStoreNotFoundError(
            f"Vector store '{collection_name}' not found at {chroma_dir}. "
            f"Run `python -m backend.app.rag.ingest` to build it."
        ) from exc

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
