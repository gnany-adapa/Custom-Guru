"""The provider-agnostic LLM interface.

Every concrete provider (Hyperspace, OpenAI, Claude, ...) implements this contract.
The rest of the app depends only on `LLMProvider` and never on a vendor SDK, so swapping
providers is a config change, not a code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

# A chat message, e.g. {"role": "user", "content": "..."}.
Message = dict[str, str]


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 1000,
    ) -> str:
        """Return the model's text reply to `messages`, optionally steered by `system`."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string, in the same order."""
