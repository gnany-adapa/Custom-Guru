"""Custom Guru FastAPI application.

Exposes a health check and the grounded persona chat endpoint.
"""
from __future__ import annotations

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.app.chat.service import answer
from backend.app.config import settings
from backend.app import obs
from backend.app.rag.retriever import VectorStoreNotFoundError

app = FastAPI(title="Custom Guru", version="0.1.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


@app.get("/health")
def health() -> dict:
    """Liveness check. Reports the configured provider and persona identity."""
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "persona": settings.persona,
        "persona_name": settings.persona_name,
        "persona_initials": settings.persona_initials,
        "persona_role": settings.persona_role,
    }


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """Answer a question as the configured persona, grounded in their transcripts."""
    try:
        result = answer(req.question)
    except VectorStoreNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="LLM provider unavailable. Is the Hyperspace proxy running?",
        ) from exc
    obs.flush()  # no-op unless Langfuse is configured
    return result
