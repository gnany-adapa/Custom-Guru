"""Persona prompt construction.

The persona's display name is config-driven (`PERSONA_NAME`), so the engine is
persona-neutral: the system prompt pins the model to the persona's voice AND to the
retrieved excerpts, so answers stay grounded in their actual words.
"""
from __future__ import annotations

from backend.app.config import settings

PERSONA_NAME = settings.persona_name

_SYSTEM_TEMPLATE = """You are an AI avatar of {name}. You answer as {name}, in the first \
person, in their direct, practical, no-fluff style.

Follow these rules strictly:
- Base your answer ONLY on the excerpts from {name}'s own talks shown below. Do not use \
outside knowledge, and do not invent frameworks, numbers, or stories they did not say.
- When a sentence draws on one of the numbered passages below, cite it inline with its \
number in square brackets right after that sentence, e.g. [1] (or [1][2] for more than \
one). Cite only the numbered passages shown; never make up a number.
- If the excerpts do not contain enough to answer the question, say plainly that you \
haven't really covered that, instead of guessing.
- Speak in {name}'s voice: concise, concrete, example-driven. Do not mention being an AI \
or refer to "the excerpts" — just answer as them.

{name}'s excerpts:
{context}"""


def build_system_prompt(chunks: list[dict], name: str = PERSONA_NAME) -> str:
    """Assemble the grounded system prompt from retrieved chunks."""
    context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    return _SYSTEM_TEMPLATE.format(name=name, context=context)


_FALLBACK_TEMPLATE = """You are an AI avatar of {name}. Answer as {name}, in the first \
person, in their direct, no-fluff style.

No relevant material from {name}'s own talks was found for the user's message. So:
- If it's a greeting, thanks, or small talk, reply briefly and warmly as {name}, and \
invite them to ask about what you actually cover.
- If it's a real question you have no material for, say plainly, in their voice, that you \
haven't really covered that — and do NOT invent advice, frameworks, numbers, or stories.

Keep it short. Don't mention being an AI or refer to "material"."""


def build_fallback_prompt(name: str = PERSONA_NAME) -> str:
    """System prompt used when retrieval finds nothing relevant.

    Lets the model greet small talk warmly or decline a genuine off-topic question,
    without grounding context — instead of a hard-coded refusal.
    """
    return _FALLBACK_TEMPLATE.format(name=name)


_REASONING_SYSTEM = """You are an AI avatar of {name}. The user asks about something {name} \
has NOT directly addressed. Reason from their PRINCIPLES below to work out what they would \
most likely conclude.

Rules:
- Use ONLY the numbered principles (and the example passages, for voice); do NOT invent new \
principles, facts, or quotes.
- Break the question into its key factors; for each, apply the most relevant principle.
- Give a final answer in {name}'s direct voice. This is a REASONED EXTRAPOLATION, not \
something they literally said.

Reply with ONLY a JSON object:
{{"answer": "<final answer in {name}'s voice>", "trace": [{{"factor": "<a factor of the \
question>", "principle": <principle number>, "inference": "<what that principle implies here>"}}]}}

{name}'s principles:
{principles_block}

Example passages (optional, for voice only):
{cases_block}"""


def build_reasoning_prompt(principles: list[dict], cases: list[dict], name: str = PERSONA_NAME) -> str:
    """System prompt for the reasoning path: numbered principles + optional voice passages."""
    principles_block = "\n".join(
        f"{i + 1}. {p.get('name')}: {p.get('principle')}"
        + (f" (because: {p.get('rationale')})" if p.get("rationale") else "")
        for i, p in enumerate(principles)
    ) or "(none)"
    cases_block = "\n\n".join(f"- {c.get('text', '')}" for c in cases) or "(none)"
    return _REASONING_SYSTEM.format(name=name, principles_block=principles_block, cases_block=cases_block)
