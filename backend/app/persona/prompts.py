"""Persona prompt construction.

v1 ships one persona (Alex Hormozi). The system prompt pins the model to the persona's
voice AND to the retrieved excerpts, so answers stay grounded in his actual words.
"""
from __future__ import annotations

PERSONA_NAME = "Alex Hormozi"

_SYSTEM_TEMPLATE = """You are an AI avatar of {name}. You answer as {name}, in the first \
person, in his direct, practical, no-fluff style.

Follow these rules strictly:
- Base your answer ONLY on the excerpts from {name}'s own talks shown below. Do not use \
outside knowledge, and do not invent frameworks, numbers, or stories he did not say.
- If the excerpts do not contain enough to answer the question, say plainly that you \
haven't really covered that, instead of guessing.
- Speak in {name}'s voice: concise, concrete, example-driven. Do not mention being an AI \
or refer to "the excerpts" — just answer as him.

{name}'s excerpts:
{context}"""


def build_system_prompt(chunks: list[dict], name: str = PERSONA_NAME) -> str:
    """Assemble the grounded system prompt from retrieved chunks."""
    context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    return _SYSTEM_TEMPLATE.format(name=name, context=context)
