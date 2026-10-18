"""Ingestion pipeline: transcripts -> cleaned text -> chunks -> embeddings -> Chroma.

Run as a script to (re)build the vector store for the configured persona:

    python -m backend.app.rag.ingest
"""
from __future__ import annotations

import glob
import os
import re

import chromadb

from backend.app.config import settings
from backend.app.llm.base import LLMProvider
from backend.app.llm.factory import get_provider

_EMBED_BATCH = 64


def clean_text(raw: str) -> str:
    """Collapse all runs of whitespace to single spaces and trim the ends."""
    return re.sub(r"\s+", " ", raw).strip()


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    """Split text into overlapping word windows.

    Overlap keeps ideas that straddle a boundary retrievable from either chunk.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    step = chunk_size_words - overlap_words
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size_words]))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def _embed_all(provider: LLMProvider, documents: list[str]) -> list[list[float]]:
    """Embed in batches so a long transcript doesn't become one huge request."""
    vectors: list[list[float]] = []
    for i in range(0, len(documents), _EMBED_BATCH):
        vectors.extend(provider.embed(documents[i : i + _EMBED_BATCH]))
    return vectors


def ingest(
    transcripts_dir: str | None = None,
    chroma_dir: str | None = None,
    collection_name: str | None = None,
    provider: LLMProvider | None = None,
    chunk_size_words: int = 200,
    overlap_words: int = 40,
) -> int:
    """Build (or rebuild) the persona's Chroma collection. Returns the chunk count."""
    transcripts_dir = transcripts_dir or settings.transcripts_dir
    chroma_dir = chroma_dir or settings.chroma_dir
    collection_name = collection_name or settings.persona
    provider = provider or get_provider()

    paths = sorted(glob.glob(os.path.join(transcripts_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt transcripts found in {transcripts_dir}")

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    for path in paths:
        source = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            text = clean_text(fh.read())
        for i, chunk in enumerate(chunk_text(text, chunk_size_words, overlap_words)):
            documents.append(chunk)
            metadatas.append({"source": source, "chunk": i})
            ids.append(f"{source}-{i}")

    embeddings = _embed_all(provider, documents)

    client = chromadb.PersistentClient(path=chroma_dir)
    # Rebuild from scratch so re-ingesting never duplicates chunks.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(documents)


if __name__ == "__main__":
    n = ingest()
    print(f"Ingested {n} chunks into collection '{settings.persona}' at {settings.chroma_dir}")
