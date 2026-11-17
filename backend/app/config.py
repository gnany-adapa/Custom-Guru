"""Environment-driven configuration for Custom Guru.

All tunables live here and are read from environment variables (loaded from a local
`.env` file via python-dotenv). Nothing secret is hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # Which LLM adapter to activate.
    llm_provider: str = _get("LLM_PROVIDER", "hyperspace")

    # Hyperspace local proxy.
    hyperspace_base_url: str = _get("HYPERSPACE_BASE_URL", "http://localhost:6655")
    hyperspace_api_key: str = _get("HYPERSPACE_API_KEY", "")
    hyperspace_llm_model: str = _get("HYPERSPACE_LLM_MODEL", "claude-sonnet-4-6")
    hyperspace_embed_model: str = _get("HYPERSPACE_EMBED_MODEL", "text-embedding-3-large")

    # Persona + RAG.
    persona: str = _get("PERSONA", "hormozi")
    transcripts_dir: str = _get("TRANSCRIPTS_DIR", "data/personas/hormozi/transcripts")
    chroma_dir: str = _get("CHROMA_DIR", "data/chroma")
    top_k: int = int(_get("TOP_K", "4"))
    # Max cosine distance for the nearest chunk to count as "covered" (grounded path).
    # PROVISIONAL at 2-transcript scale: measured covered questions land ~0.56, in-domain
    # questions he hasn't covered ~0.60-0.73, off-topic ~0.89. 0.58 sits in the gap so the
    # uncovered-but-in-domain ones fall through to the reasoning path instead of grounding
    # and then declining. Re-calibrate from the eval harness as transcripts are added.
    similarity_threshold: float = float(_get("SIMILARITY_THRESHOLD", "0.58"))
    # Max cosine distance for the nearest *principle* to be relevant enough to reason from.
    # Calibrated: novel on-domain questions land ~0.55-0.75; off-topic ~0.81+.
    reasoning_threshold: float = float(_get("REASONING_THRESHOLD", "0.78"))

    # Langfuse observability (optional — the app runs fine without it).
    langfuse_public_key: str = _get("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = _get("LANGFUSE_SECRET_KEY", "")
    # Langfuse's UI emits LANGFUSE_BASE_URL; its SDK wants LANGFUSE_HOST — accept either.
    langfuse_host: str = _get("LANGFUSE_HOST", "") or _get("LANGFUSE_BASE_URL", "http://localhost:3000")

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
