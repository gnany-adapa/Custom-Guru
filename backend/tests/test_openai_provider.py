"""Offline unit tests for OpenAIProvider.

All network calls are monkeypatched — no real OpenAI key or network access is
required. These tests verify that generate() and embed() correctly parse
OpenAI-shaped JSON responses, that the system prompt is prepended as a
system-role message, and that a missing API key raises ValueError at
construction time.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.app.llm.openai_provider import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers — canned OpenAI-shaped responses
# ---------------------------------------------------------------------------

def _chat_response(text: str) -> MagicMock:
    """Return a mock requests.Response shaped like OpenAI chat/completions."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return mock


def _embed_response(vectors: list[list[float]]) -> MagicMock:
    """Return a mock requests.Response shaped like OpenAI embeddings."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "model": "text-embedding-3-large",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def provider() -> OpenAIProvider:
    """OpenAIProvider initialised with a fake key so no env var is needed."""
    return OpenAIProvider(
        api_key="sk-test-fake",
        llm_model="gpt-4o",
        embed_model="text-embedding-3-large",
        base_url="https://api.openai.com/v1",
    )


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------

def test_generate_returns_reply_text(provider: OpenAIProvider) -> None:
    with patch("requests.post", return_value=_chat_response("pong")) as mock_post:
        result = provider.generate(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=10,
        )

    assert result == "pong"
    mock_post.assert_called_once()


def test_generate_prepends_system_message(provider: OpenAIProvider) -> None:
    with patch("requests.post", return_value=_chat_response("ok")) as mock_post:
        provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            system="You are a helpful assistant.",
            max_tokens=20,
        )

    _, kwargs = mock_post.call_args
    sent_messages = kwargs["json"]["messages"]
    assert sent_messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert sent_messages[1] == {"role": "user", "content": "hi"}


def test_generate_no_system_when_none(provider: OpenAIProvider) -> None:
    with patch("requests.post", return_value=_chat_response("ok")) as mock_post:
        provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            system=None,
        )

    _, kwargs = mock_post.call_args
    sent_messages = kwargs["json"]["messages"]
    assert all(m["role"] != "system" for m in sent_messages)


def test_generate_passes_model_and_max_tokens(provider: OpenAIProvider) -> None:
    with patch("requests.post", return_value=_chat_response("ok")) as mock_post:
        provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=42,
        )

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "gpt-4o"
    assert kwargs["json"]["max_tokens"] == 42


# ---------------------------------------------------------------------------
# embed() tests
# ---------------------------------------------------------------------------

def test_embed_returns_vectors(provider: OpenAIProvider) -> None:
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    with patch("requests.post", return_value=_embed_response(vectors)):
        result = provider.embed(["hello world", "foo bar"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]


def test_embed_sorts_by_index(provider: OpenAIProvider) -> None:
    """API may return items out of order; we should sort by index."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    # Return index 1 before index 0 intentionally.
    mock.json.return_value = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 1, "embedding": [0.9, 0.8]},
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
        ],
        "model": "text-embedding-3-large",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }

    with patch("requests.post", return_value=mock):
        result = provider.embed(["first", "second"])

    assert result[0] == [0.1, 0.2]
    assert result[1] == [0.9, 0.8]


def test_embed_batches_large_input(provider: OpenAIProvider) -> None:
    """Inputs exceeding _EMBED_BATCH should be split across multiple POST calls."""
    from backend.app.llm.openai_provider import _EMBED_BATCH

    n = _EMBED_BATCH + 1  # one item spills into a second batch
    texts = [f"text {i}" for i in range(n)]
    vectors = [[float(i)] for i in range(n)]

    call_count = 0

    def fake_post(url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        batch = kwargs["json"]["input"]  # type: ignore[index]
        start = call_count * _EMBED_BATCH
        resp_vectors = vectors[start : start + len(batch)]
        call_count += 1
        return _embed_response(resp_vectors)

    with patch("requests.post", side_effect=fake_post):
        result = provider.embed(texts)

    assert call_count == 2
    assert len(result) == n


# ---------------------------------------------------------------------------
# Construction error
# ---------------------------------------------------------------------------

def test_missing_api_key_raises() -> None:
    """OpenAIProvider should raise ValueError when no key is configured."""
    with patch("backend.app.llm.openai_provider.settings") as mock_settings:
        mock_settings.openai_api_key = ""
        mock_settings.openai_llm_model = "gpt-4o"
        mock_settings.openai_embed_model = "text-embedding-3-large"
        mock_settings.openai_base_url = "https://api.openai.com/v1"
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIProvider()
