"""Optional Langfuse tracing.

Langfuse is a *soft* dependency: if it isn't installed or no keys are configured, every
helper here degrades to a no-op so the open-source template runs fine without it. All calls
are wrapped so observability can never break the app.
"""
from __future__ import annotations

import os

from backend.app.config import settings

_enabled = False
_client = None
_lf_observe = None

if settings.langfuse_enabled:
    try:
        # The SDK reads these from the environment.
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        from langfuse import get_client, observe as _lf_observe  # type: ignore

        _client = get_client()
        _enabled = True
    except Exception:  # missing package, bad config — just disable tracing
        _enabled = False
        _client = None
        _lf_observe = None


def enabled() -> bool:
    return _enabled


def observe(*dargs, **dkwargs):
    """Trace a function with Langfuse when enabled; otherwise an identity decorator.

    Supports both `@observe` and `@observe(name="...")`.
    """

    def decorator(fn):
        if _enabled and _lf_observe is not None:
            try:
                return _lf_observe(**dkwargs)(fn)
            except Exception:
                return fn
        return fn

    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return decorator(dargs[0])
    return decorator


def update_trace(**kwargs) -> None:
    if _enabled and _client is not None:
        try:
            _client.update_current_trace(**kwargs)
        except Exception:
            pass


def flush() -> None:
    if _enabled and _client is not None:
        try:
            _client.flush()
        except Exception:
            pass
