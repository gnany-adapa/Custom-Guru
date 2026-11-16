"""The optional tracing layer must never change behavior or raise — on or off."""
from __future__ import annotations

from backend.app.obs import enabled, flush, observe, update_trace


@observe(name="t")
def _f(x):
    return x + 1


@observe
def _g(x):
    return x * 2


def test_observe_preserves_function_behavior():
    assert _f(1) == 2  # @observe(name=...) form
    assert _g(3) == 6  # bare @observe form


def test_trace_helpers_never_raise():
    update_trace(metadata={"hello": "world"})
    flush()
    assert isinstance(enabled(), bool)
