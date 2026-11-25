"""Selects the active LLM provider from configuration.

Add a new provider by implementing `LLMProvider` and registering it in `_PROVIDERS`.
Users then switch to it by setting `LLM_PROVIDER` in their `.env` — no code changes.
"""
from __future__ import annotations

from backend.app.config import settings
from backend.app.llm.base import LLMProvider
from backend.app.llm.hyperspace import HyperspaceProvider
from backend.app.llm.openai_provider import OpenAIProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "hyperspace": HyperspaceProvider,
    "openai": OpenAIProvider,
    # "claude": ClaudeProvider,
}


def get_provider(name: str | None = None) -> LLMProvider:
    key = (name or settings.llm_provider).lower()
    try:
        provider_cls = _PROVIDERS[key]
    except KeyError:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown LLM_PROVIDER '{key}'. Known providers: {known}") from None
    return provider_cls()
