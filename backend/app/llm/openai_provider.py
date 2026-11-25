"""OpenAI provider adapter for Custom Guru.

Talks to the OpenAI API with plain `requests` (no vendor SDK dependency):
  - Chat:       POST /chat/completions  (OpenAI Chat Completions format)
  - Embeddings: POST /embeddings        (OpenAI Embeddings format)

The system prompt is passed as an OpenAI system-role message prepended to the
messages list, matching OpenAI's recommended convention.

Embedding calls are batched (_EMBED_BATCH inputs per request) to stay within
rate-limit and payload limits. Lower _EMBED_BATCH if you hit 429 errors on
large corpora.

NOTE ON EMBEDDING DIMENSIONS
-----------------------------
text-embedding-3-large returns 3072-dim vectors; text-embedding-3-small returns
1536. Keep the model consistent across a Chroma collection — mixing models
produces silent semantic garbage. If you switch models, delete data/chroma/ and
re-run ingest.
"""
from __future__ import annotations

import requests

from backend.app.config import settings
from backend.app.llm.base import LLMProvider, Message

_EMBED_BATCH = 64  # inputs per embeddings request; lower if you hit 429s


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        llm_model: str | None = None,
        embed_model: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or settings.openai_api_key
        self.llm_model = llm_model or settings.openai_llm_model
        self.embed_model = embed_model or settings.openai_embed_model
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 1000,
    ) -> str:
        """Call chat/completions and return only the reply text.

        The system prompt, when provided, is prepended as a system-role message.
        requests.RequestException propagates to the caller unchanged.
        """
        all_messages: list[Message] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.llm_model,
                "messages": all_messages,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string, in the same order.

        Large input lists are automatically split into batches of _EMBED_BATCH.
        requests.RequestException propagates to the caller unchanged.
        """
        results: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            batch = texts[i : i + _EMBED_BATCH]
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.embed_model, "input": batch},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            # OpenAI returns items in input order, but sort by index to be safe.
            data = sorted(resp.json()["data"], key=lambda item: item["index"])
            results.extend(item["embedding"] for item in data)
        return results
