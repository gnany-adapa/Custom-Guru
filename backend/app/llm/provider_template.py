"""Copy-me template for adding a new LLM provider.

HOW TO USE
----------
1. Copy this file to a new name, e.g. `openai_provider.py`, and rename the class.
2. Implement the two methods below for your provider (REST via `requests`, or its SDK).
3. Register it in `factory.py`:
       _PROVIDERS = {..., "yourname": YourProvider}
4. Set `LLM_PROVIDER=yourname` (and any keys/model names) in `.env`.

The rest of the app only ever calls `generate()` and `embed()` through the `LLMProvider`
interface, so once these two methods work, everything else (chat, ingestion, UI) just works.
See `hyperspace.py` for a complete, working example to copy from.
"""
from __future__ import annotations

from backend.app.config import settings
from backend.app.llm.base import LLMProvider, Message


class TemplateProvider(LLMProvider):
    def __init__(self) -> None:
        # Pull whatever you need from settings / env. Add your own fields to
        # backend/app/config.py (e.g. an OPENAI_API_KEY) and read them here.
        # self.api_key = settings.your_api_key
        # self.model = settings.your_model
        ...

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 1000,
    ) -> str:
        """Call your provider's chat/completions API and return ONLY the reply text.

        `messages` is a list of {"role": "user"|"assistant", "content": str}.
        `system` is an optional system prompt string (may be None).
        Translate these into your provider's request shape, then pull the assistant's
        text out of its response and return it as a plain `str`.
        """
        raise NotImplementedError("Implement generate() for your provider")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector (list of floats) per input string, in order.

        The output length must equal len(texts). If your provider returns items out of
        order, sort them back by their index before returning.
        """
        raise NotImplementedError("Implement embed() for your provider")
