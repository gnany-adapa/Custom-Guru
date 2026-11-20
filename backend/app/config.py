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

    # Persona + RAG. The public demo ships persona-neutral: identity is config-driven
    # with generic defaults, and no persona data is included. Point these at your own
    # corpus and set the display identity below (e.g. in a local, gitignored .env).
    persona: str = _get("PERSONA", "example")
    transcripts_dir: str = _get("TRANSCRIPTS_DIR", "data/personas/example/transcripts")
    # Display identity shown in the UI and used in prompts. Neutral by default.
    persona_name: str = _get("PERSONA_NAME", "Example Mentor")
    persona_initials: str = _get("PERSONA_INITIALS", "EM")
    persona_role: str = _get("PERSONA_ROLE", "Grounded in their own words")
    chroma_dir: str = _get("CHROMA_DIR", "data/chroma")
    top_k: int = int(_get("TOP_K", "4"))
    # Max cosine distance for the nearest chunk to count as "covered" (grounded path).
    # Calibrated on the 12-question eval set (backend.app.eval.runner) at 2-transcript
    # scale: covered questions land 0.48-0.58, novel (uncovered-but-in-domain) ones
    # 0.55-0.68 — these OVERLAP, so no single threshold separates them perfectly. 0.60
    # sits just above the covered cluster: grounds all 4 covered, routes 3/4 novel to
    # reasoning (the 4th, at 0.545, falls inside the covered band). The real fix is more
    # transcripts — denser coverage pulls covered hits nearer and opens the gap. Re-run
    # the eval as data grows. (An env SIMILARITY_THRESHOLD overrides this default.)
    similarity_threshold: float = float(_get("SIMILARITY_THRESHOLD", "0.60"))
    # Max cosine distance for the nearest *principle* to be relevant enough to reason from.
    # Calibrated: novel on-domain questions land ~0.55-0.75; off-topic ~0.81+.
    reasoning_threshold: float = float(_get("REASONING_THRESHOLD", "0.78"))

    # Faithfulness critic: when enabled, reasoned answers are checked against principles.
    critic_enabled: bool = _get("CRITIC_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

    # Langfuse observability (optional — the app runs fine without it).
    langfuse_public_key: str = _get("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = _get("LANGFUSE_SECRET_KEY", "")
    # Langfuse's UI emits LANGFUSE_BASE_URL; its SDK wants LANGFUSE_HOST — accept either.
    langfuse_host: str = _get("LANGFUSE_HOST", "") or _get("LANGFUSE_BASE_URL", "http://localhost:3000")

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
