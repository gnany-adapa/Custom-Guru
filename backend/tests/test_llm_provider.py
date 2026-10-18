"""Integration tests for the LLM provider adapter layer.

These hit the live Hyperspace proxy, so it must be running locally with a valid key
in `.env`. They verify the adapter contract: generate() returns text, embed() returns
numeric vectors.
"""
from __future__ import annotations

from backend.app.llm.factory import get_provider


def test_hyperspace_generate_returns_text():
    provider = get_provider()
    out = provider.generate(
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        system="You reply with exactly one lowercase word.",
        max_tokens=10,
    )
    assert isinstance(out, str)
    assert out.strip() != ""


def test_hyperspace_embed_returns_vectors():
    provider = get_provider()
    vecs = provider.embed(["hello world", "a second piece of text"])
    assert isinstance(vecs, list)
    assert len(vecs) == 2
    assert all(isinstance(v, list) and len(v) > 0 for v in vecs)
    assert all(isinstance(x, float) for x in vecs[0][:5])
    # text-embedding-3-large returns 3072-dim vectors
    assert len(vecs[0]) == len(vecs[1])
