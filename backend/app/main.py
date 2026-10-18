"""Custom Guru FastAPI application.

v1 exposes a health check now; the /chat endpoint is added in a later milestone.
"""
from __future__ import annotations

from fastapi import FastAPI

from backend.app.config import settings

app = FastAPI(title="Custom Guru", version="0.1.0")


@app.get("/health")
def health() -> dict:
    """Liveness check. Reports which LLM provider is configured."""
    return {"status": "ok", "provider": settings.llm_provider, "persona": settings.persona}
