"""Tests for the /chat endpoint: input validation and graceful error mapping.

All offline — `answer` is monkeypatched, so no proxy or vector store is needed.
"""
from __future__ import annotations

import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app import main as main_module
from backend.app.main import ChatRequest, app
from backend.app.rag.retriever import VectorStoreNotFoundError

client = TestClient(app)


def test_chat_request_rejects_blank_and_overlong():
    with pytest.raises(ValidationError):
        ChatRequest(question="")
    with pytest.raises(ValidationError):
        ChatRequest(question="    ")
    with pytest.raises(ValidationError):
        ChatRequest(question="x" * 3000)


def test_chat_maps_missing_vector_store_to_503(monkeypatch):
    def boom(_q):
        raise VectorStoreNotFoundError("Vector store 'hormozi' not found. Run ingest.")

    monkeypatch.setattr(main_module, "answer", boom)
    r = client.post("/chat", json={"question": "hello"})
    assert r.status_code == 503
    assert "ingest" in r.json()["detail"].lower()


def test_chat_maps_provider_failure_to_503(monkeypatch):
    def boom(_q):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(main_module, "answer", boom)
    r = client.post("/chat", json={"question": "hello"})
    assert r.status_code == 503


def test_chat_success_passes_through(monkeypatch):
    payload = {
        "answer": "A [1]",
        "covered": True,
        "citations": [{"n": 1, "source": "talk.txt", "excerpt": "…"}],
    }
    monkeypatch.setattr(main_module, "answer", lambda _q: payload)
    r = client.post("/chat", json={"question": "hello"})
    assert r.status_code == 200
    assert r.json() == payload
