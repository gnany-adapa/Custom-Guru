"""Hyperspace local-proxy adapter.

Talks to the Hyperspace proxy with plain `requests` (no vendor SDKs):
  - LLM:        POST /anthropic/v1/messages   (Anthropic Messages format, `x-api-key`)
  - Embeddings: POST /openai/v1/embeddings    (OpenAI format, `Authorization: Bearer`)
"""
from __future__ import annotations

import requests

from backend.app.config import settings
from backend.app.llm.base import LLMProvider, Message


class HyperspaceProvider(LLMProvider):
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        llm_model: str | None = None,
        embed_model: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.base_url = (base_url or settings.hyperspace_base_url).rstrip("/")
        self.api_key = api_key or settings.hyperspace_api_key
        self.llm_model = llm_model or settings.hyperspace_llm_model
        self.embed_model = embed_model or settings.hyperspace_embed_model
        self.timeout = timeout

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 1000,
    ) -> str:
        payload: dict = {
            "model": self.llm_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        resp = requests.post(
            f"{self.base_url}/anthropic/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            f"{self.base_url}/openai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.embed_model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # The API may return items out of order; sort by the index it echoes back.
        data = sorted(data, key=lambda item: item["index"])
        return [item["embedding"] for item in data]
