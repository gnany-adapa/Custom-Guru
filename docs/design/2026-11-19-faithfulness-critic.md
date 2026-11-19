# Design: Faithfulness Critic for Reasoned Answers

**Status:** proposed · **Date:** 2026-11-19 · **Issue:** #2

## Intent & why

The reasoned path (`reason_from_principles`) answers questions the persona never directly
addressed, by extrapolating from extracted principles. It already validates the **trace** —
every step must reference a real retrieved principle, and fabricated/out-of-range references
are dropped. But the free-text **answer prose** is not checked against anything: the model can
assert claims in the answer that none of the cited principles support. The eval confirms this
is the weak spot — reasoned answers score ~1–2/5 on groundedness (the LLM judge finds claims
unsupported by the cited evidence).

The faithfulness critic closes that gap: after the reasoning path drafts an answer, a second
pass checks each claim against the principles it was built from, removes or softens the
unsupported ones, exposes an auditable verdict of what it changed, and declines if nothing
substantive survives. This is the project's "grounded, auditable" thesis applied to its own
output — an answer that audits and corrects itself, in the open.

## Goals

- Raise reasoned-answer groundedness (measured: eval groundedness/transparency go up).
- Make the correction **auditable** — the user can see what the critic flagged, not just a
  cleaned result.
- Never fabricate: the critic only removes/softens; it never adds claims.
- Be **toggleable** so the eval can measure critic-off vs critic-on (a concrete before/after
  result for the report/demo).
- Degrade safely: a critic failure must never break the answer.

## Non-goals

- Not applied to the **grounded** path — grounded answers already have verbatim inline-citation
  verification (`_apply_citations`).
- Not a multi-agent debate or from-scratch regeneration — it revises the existing draft.
- Not verbatim quote-matching — a reasoned answer is extrapolation, not quotation, so there is
  nothing to string-match; the check is necessarily LLM-judge-style.

## Anti-goals (failure even if it "works")

- Making answers *sound* more confident/polished without grounding them — the opposite of the
  thesis.
- Over-declining: discarding salvageable answers because one minor aside was unsupported.
- A silent rewrite that hides what changed — defeats auditability.

## Constraints

- Provider-agnostic (`LLMProvider.generate` only); works on the Hyperspace proxy.
- +1 LLM call per reasoned answer (reasoned path only; grounded/fallback unaffected).
- Must follow the codebase's defensive-JSON pattern: unparseable critic output → safe no-op
  (keep the original answer, mark unchecked) rather than an error.

## Approaches considered

- **A. Verify + annotate only.** Score faithfulness and flag claims; answer unchanged.
  Most auditable, least destructive, but leaves a shaky answer on screen and does **not** move
  the eval groundedness number. Rejected as the sole behavior.
- **B. Verify + revise (+ decline if unsalvageable).** Strip/soften unsupported claims, return
  the cleaned answer; decline when nothing survives. Matches the codebase's existing
  "drop what isn't supported, keep what is" discipline and directly lifts the metric.
- **C. Verify + gate only.** Below a faithfulness bar, replace the whole answer with a decline.
  Strongest guarantee but throws away salvageable answers (over-declines). Rejected as the sole
  behavior.
- **Chosen: B, with A's visible verdict layered on.** Revise for a trustworthy final answer
  (moves the metric), AND surface what was flagged/removed (keeps it auditable), AND fall back
  to a decline only when the answer collapses. Best serves both the eval and the thesis.

## The how (decided)

**New module `backend/app/chat/critic.py`**
- `critique(question, answer, principles, provider, *, min_supported=1) -> dict`
- One LLM call. Prompt gives the question, the draft answer, and the numbered principles
  (name + statement). The critic returns JSON:
  `{"claims": [{"claim": str, "supported": bool, "principle": int|null}], "revised_answer": str}`.
  - `revised_answer`: the answer with unsupported claims removed/softened; keeps only what the
    principles support.
  - A claim is "supported" only if an actual listed principle backs it.
- Returns a normalized dict:
  `{"checked": bool, "faithful": bool, "flagged": [unsupported claim strings], "answer": revised_or_original}`.
- **Defensive:** unparseable/empty output → `{"checked": False, "faithful": None, "flagged": [], "answer": <original>}`
  (safe no-op; never raises).
- **Gate:** if every substantive claim is unsupported (or `revised_answer` is empty / < min_supported
  supported claims), signal "unsalvageable" so the caller declines.

**Prompt** lives in `prompts.py` (`build_critic_prompt`), consistent with the other builders;
persona-neutral; forces JSON-only output like the existing reasoning/extraction prompts.

**Service wiring (`service.py`, reasoned branch only):**
- After `reason_from_principles`, if `settings.critic_enabled`, call `critique(...)`, passing
  the **full retrieved `principles` list** (in scope here), NOT just the trace `citations` —
  the draft was reasoned from the full set, so a valid claim may rest on a retrieved principle
  that never entered the trace; feeding only citations would false-flag it (over-declining).
- On success: set the answer to the revised text and attach a `critique` key to the result:
  `{"checked": True, "faithful": bool, "flagged": [...]}`.
- On "unsalvageable" (gate trips): **keep `mode="reasoned"`** — the router chose correctly; the
  critic's refusal is a separate axis. Replace the answer with a fixed, honest decline string
  (no second LLM call — do NOT reuse `build_fallback_prompt`, which means another `generate`
  and "no material found" wording), set `citations=[]`, `trace=[]`, and `critique.faithful=False`.
- On critic failure/no-op: keep the original reasoned answer, `critique.checked = False`.
- The returned dict gains one key, `critique`; grounded/fallback set `critique` to a
  checked=False default so the shape is uniform. (Document in the `answer()` docstring.)
- **`mode` and faithfulness are orthogonal:** `mode` records the routing path (grounded /
  reasoned / fallback); `critique` records whether the reasoned prose is faithful. A critic
  decline is a reasoned-path outcome, never a fallback — so `eval` behavior-routing is
  unaffected by the critic.

**Config:** `critic_enabled: bool` (env `CRITIC_ENABLED`, default true). Surfaced in
`.env.example`. The eval toggles it to A/B.

**UI (`streamlit_app.py`, reasoned messages):** a small "✓ Faithfulness-checked" badge; when
`flagged` is non-empty, an expander "What the critic flagged (N)" listing the removed/softened
claims. Auditable and demo-able. Degrades silently when `checked` is False.

**Eval (`eval/runner.py`):** add a critic-on-vs-off comparison — run the question set twice
(critic disabled, then enabled) and report the groundedness/transparency delta. This is the
headline before/after artifact. **To keep the delta honest:** also report per-run counts of
answers that were *unchanged / revised / declined*, because the judge scores a clean decline
5/5 on groundedness — so a naive delta could be inflated by declines rather than genuine
revision. Routing `behavior_pass_rate` is unaffected (a critic decline stays `mode="reasoned"`,
so novel questions remain routing successes). (Implementation note: thread a `critic` flag
through `answer()` / the runner; no judge-rubric change required, though a `faithfulness`
dimension is an optional stretch.)

**Tests (`backend/tests/test_critic.py` + a service test):**
- faithful answer → passes through unchanged, `faithful=True`, no flags;
- answer with an unsupported claim → claim flagged, `revised_answer` drops it;
- unsalvageable answer → gate signals decline;
- malformed critic JSON → safe no-op (original answer, `checked=False`);
- service: a reasoned result carries a `critique` key; gate routes to fallback.
All with a fake provider (no proxy), mirroring `test_chat_service_unit.py`.

## What's left to the builder

- Exact critic prompt wording (follow the JSON-only, no-prose style of the existing prompts).
- The precise `revised_answer`-empty / all-unsupported threshold for the decline gate
  (default: decline only if zero substantive claims are supported — conservative, avoids
  over-declining).
- Whether the eval A/B lives in the `runner.py` `__main__` or a small separate entry point.
