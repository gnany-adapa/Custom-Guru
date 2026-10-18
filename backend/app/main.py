"""Custom Guru FastAPI application.

v1 exposes a health check now; the /chat endpoint is added in a later milestone.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from backend.app.chat.service import answer
from backend.app.config import settings

app = FastAPI(title="Custom Guru", version="0.1.0")


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    """Liveness check. Reports which LLM provider is configured."""
    return {"status": "ok", "provider": settings.llm_provider, "persona": settings.persona}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """Answer a question as the configured persona, grounded in their transcripts."""
    return answer(req.question)
