# Faithfulness Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a critic that checks each reasoned answer against the principles it was built from, revises out unsupported claims, exposes an auditable verdict, and declines when nothing survives.

**Architecture:** A new `critic.py` runs one LLM-judge pass over the reasoned draft + the full retrieved principles, returning supported/unsupported claims and a revised answer. The chat service calls it on the reasoned path only, attaches a uniform `critique` field to every result, and — when no claim is supported — emits a fixed honest decline that stays `mode="reasoned"`. The critic is toggleable so the eval can measure groundedness critic-off vs critic-on.

**Tech Stack:** Python, FastAPI, Streamlit, pytest, Chroma, provider-agnostic `LLMProvider`.

**Spec:** `docs/design/2026-11-19-faithfulness-critic.md`

## Global Constraints

- Provider-agnostic: only `LLMProvider.generate(messages, system=, max_tokens=)`.
- At most **+1 LLM call** per reasoned answer. The decline path must NOT call `generate` again (no reuse of `build_fallback_prompt`); use a fixed string.
- Defensive JSON: unparseable/empty critic output → safe no-op (keep original answer, `checked=False`); never raise.
- Persona-neutral: no hardcoded names; use `settings.persona_name`.
- Critic applies to the **reasoned path only**; grounded/fallback get a uniform `checked=False` `critique` default.
- Conventional Commits; tests run with `.venv/bin/python -m pytest`.

## Review Focus

- **Malformed/empty critic JSON** → safe no-op (original answer returned, `critique.checked=False`), never an exception. *(Task 1 tests.)*
- **All claims unsupported / empty `revised_answer`** → gate declines with the fixed string, `mode` stays `"reasoned"`, `critique.declined=True`. *(Task 2 tests.)*
- **`CRITIC_ENABLED=false`** → reasoned answer unchanged, but the uniform `critique` default is still present. *(Task 2 tests.)*
- **Grounded & fallback results** still carry the uniform `critique` default (shape consistency for `/chat` + UI). *(Task 2 tests.)*
- **A claim resting on a retrieved principle not in the trace** must not be false-flagged — the critic receives the full `principles` list, not just `citations`. *(Task 2 wiring; enforced by passing the full list.)*

---

### Task 1: Critic core (config, prompt, module)

**Files:**
- Modify: `backend/app/config.py` (add `critic_enabled`)
- Modify: `backend/app/persona/prompts.py` (add `build_critic_prompt`)
- Modify: `.env.example` (add `CRITIC_ENABLED`)
- Create: `backend/app/chat/critic.py`
- Test: `backend/tests/test_critic.py`

**Interfaces:**
- Consumes: `settings` (config); `LLMProvider`.
- Produces:
  - `settings.critic_enabled: bool` (env `CRITIC_ENABLED`, default `"true"`; parse truthily like `"1/true/yes"`).
  - `build_critic_prompt(question: str, answer: str, principles: list[dict], name: str | None = None) -> str` — numbers the principles (name + statement), forces a JSON-only reply `{"claims": [{"claim": str, "supported": bool, "principle": int|null}], "revised_answer": str}`; a claim is `supported` only if a listed principle backs it; `revised_answer` keeps only supported claims; `name` defaults to `settings.persona_name`.
  - `critique(question: str, answer: str, principles: list[dict], provider: LLMProvider, *, min_supported: int = 1) -> dict` returning `{"checked": bool, "faithful": bool | None, "flagged": list[str], "answer": str, "salvageable": bool}`.

- [ ] **Step 1: Write failing tests** in `backend/tests/test_critic.py` with a `_FakeProvider` (reply string injected; `embed` raises), mirroring `test_chat_service_unit.py`:

```python
def test_faithful_answer_passes_through():
    # provider returns all-supported claims + unchanged revised_answer
    out = critique("q", "Raise price.", PRINCIPLES, fp(all_supported))
    assert out["checked"] and out["faithful"] and out["flagged"] == []
    assert out["salvageable"] and out["answer"] == "Raise price."

def test_unsupported_claim_is_flagged_and_revised():
    out = critique("q", "Raise price and hire a CFO.", PRINCIPLES, fp(one_unsupported))
    assert out["faithful"] is False
    assert "hire a CFO" in " ".join(out["flagged"])
    assert out["answer"] == "Raise price."          # revised_answer from the critic
    assert out["salvageable"] is True

def test_all_unsupported_is_unsalvageable():
    out = critique("q", "...", PRINCIPLES, fp(all_unsupported_empty_revised))
    assert out["salvageable"] is False
    assert out["faithful"] is False

def test_malformed_json_is_safe_noop():
    out = critique("q", "original", PRINCIPLES, fp("not json"))
    assert out["checked"] is False and out["faithful"] is None
    assert out["answer"] == "original" and out["salvageable"] is True
```

- [ ] **Step 2: Run tests, verify they fail** — `.venv/bin/python -m pytest backend/tests/test_critic.py -v` → FAIL (import/`critique` undefined).
- [ ] **Step 3: Add `critic_enabled`** to `config.py` `Settings` (env `CRITIC_ENABLED`, default true) and `CRITIC_ENABLED=true` to `.env.example` with a one-line comment.
- [ ] **Step 4: Add `build_critic_prompt`** to `prompts.py` in the style of `build_reasoning_prompt` (numbered principles, JSON-only, persona-neutral via `settings.persona_name`).
- [ ] **Step 5: Implement `critique(...)`** in `critic.py`: reuse the defensive `_parse_json_object` pattern; compute `flagged` = claims with `supported` falsey; `supported_count` = supported claims; `revised` = `revised_answer` stripped or original; `salvageable = supported_count >= min_supported and bool(revised)`; `faithful = (len(flagged) == 0)`; `answer = revised if salvageable else answer`; unparseable → `{"checked": False, "faithful": None, "flagged": [], "answer": answer, "salvageable": True}`.
- [ ] **Step 6: Run tests, verify pass.**
- [ ] **Step 7: Commit** — `feat: add faithfulness critic core (verify/revise reasoned answers)`.

---

### Task 2: Wire the critic into the chat service

**Files:**
- Modify: `backend/app/chat/service.py`
- Test: `backend/tests/test_chat_service_unit.py`

**Interfaces:**
- Consumes: `critique` (Task 1); `settings.critic_enabled`; existing `reason_from_principles`, `retrieve_principles`.
- Produces: `answer(...)` gains `critic: bool | None = None` (resolved to `settings.critic_enabled` when `None`). Every return dict gains `critique: {"checked": bool, "faithful": bool | None, "flagged": list[str], "declined": bool}`. Module constant `_CRITIC_DECLINE` (uses `settings.persona_name`).

- [ ] **Step 1: Write failing tests** in `test_chat_service_unit.py` (stub `retrieve`/`retrieve_principles`, fake provider whose reply flips between the reasoning JSON and the critic JSON as needed — or stub `service.critique` directly for determinism):

```python
def test_reasoned_result_carries_critique(monkeypatch):
    # critic finds all supported → faithful, not declined, answer revised text
    assert result["mode"] == "reasoned"
    assert result["critique"] == {"checked": True, "faithful": True, "flagged": [], "declined": False}

def test_unsupported_claim_is_revised_in_reasoned_answer(monkeypatch):
    assert result["critique"]["flagged"]            # non-empty
    assert result["critique"]["declined"] is False
    assert result["answer"] == REVISED_TEXT

def test_unsalvageable_reasoned_answer_declines_but_stays_reasoned(monkeypatch):
    assert result["mode"] == "reasoned"             # NOT fallback
    assert result["critique"]["declined"] is True
    assert result["critique"]["faithful"] is False
    assert result["answer"] == service._CRITIC_DECLINE
    assert result["citations"] == [] and result["trace"] == []

def test_critic_disabled_leaves_answer_unchanged(monkeypatch):
    result = answer("q", provider=fp, threshold=0.75, critic=False)
    assert result["critique"] == {"checked": False, "faithful": None, "flagged": [], "declined": False}

def test_grounded_and_fallback_carry_uniform_critique_default(monkeypatch):
    assert grounded["critique"]["checked"] is False
    assert fallback["critique"]["checked"] is False
```

- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement wiring** in `service.py`:
  - add `critic: bool | None = None` param; `critic = settings.critic_enabled if critic is None else critic`.
  - define `_DEFAULT_CRITIQUE = {"checked": False, "faithful": None, "flagged": [], "declined": False}` and `_CRITIC_DECLINE` (honest, persona-named).
  - grounded + fallback returns: add `"critique": dict(_DEFAULT_CRITIQUE)`.
  - reasoned branch: after `reason_from_principles(...)`, if `critic`, call `critique(question, result["answer"], principles, provider)` (full `principles`, not citations). Map: no-op → default critique; `salvageable` False → replace with `{answer:_CRITIC_DECLINE, covered:False, mode:"reasoned", trace:[], citations:[], critique:{checked:True, faithful:False, flagged:c["flagged"], declined:True}}`; else → `result["answer"]=c["answer"]`, `result["critique"]={checked:True, faithful:c["faithful"], flagged:c["flagged"], declined:False}`. If `critic` off → `result["critique"]=dict(_DEFAULT_CRITIQUE)`.
  - update the `answer()` docstring `Returns` to mention `critique`.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** — `feat: run the faithfulness critic on reasoned answers`.

---

### Task 3: Eval A/B (critic off vs on)

**Files:**
- Modify: `backend/app/eval/runner.py`
- Test: `backend/tests/test_eval.py`

**Interfaces:**
- Consumes: `answer` with the `critic` flag (Task 2); existing `run_eval`, `aggregate`.
- Produces: `critique_counts(results: list[dict]) -> dict` → `{"unchanged": int, "revised": int, "declined": int}` classifying reasoned results by their `critique`; `compare_critic(...)` running the set twice and returning both aggregates + counts.

- [ ] **Step 1: Write failing offline test** in `test_eval.py`:

```python
def test_critique_counts_classifies_reasoned_results():
    results = [
        {"mode": "reasoned", "critique": {"checked": False, "flagged": [], "declined": False}},
        {"mode": "reasoned", "critique": {"checked": True, "flagged": ["x"], "declined": False}},
        {"mode": "reasoned", "critique": {"checked": True, "flagged": ["y"], "declined": True}},
        {"mode": "grounded", "critique": {"checked": False, "flagged": [], "declined": False}},
    ]
    assert critique_counts(results) == {"unchanged": 1, "revised": 1, "declined": 1}
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `critique_counts` (unchanged = reasoned & not flagged & not declined; revised = flagged & not declined; declined = declined) and `compare_critic(questions=None, provider=None)` that calls `run_eval` with `answer_fn=lambda q: answer(q, critic=False)` then `critic=True`, returning `{"off": {...}, "on": {...}, "counts": critique_counts(on_results)}`. Extend `run_eval` results to retain `critique`. Update `__main__` to print the off/on groundedness + transparency + the unchanged/revised/declined counts.
- [ ] **Step 4: Run, verify pass** (offline unit); then a live `.venv/bin/python -m backend.app.eval.runner` sanity run is a manual check, not part of the test.
- [ ] **Step 5: Commit** — `test: eval A/B for the critic (groundedness lift + revise/decline counts)`.

---

### Task 4: Surface the verdict in the UI

**Files:**
- Modify: `frontend/streamlit_app.py`

**Interfaces:**
- Consumes: `/chat` response now carries `critique`.
- Produces: reasoned assistant messages show a "✓ Faithfulness-checked" badge when `critique.checked`, and an expander "What the critic flagged (N)" listing `flagged` when non-empty; degrade silently when `checked` is false.

- [ ] **Step 1:** `ask_backend()` returns `critique = data.get("critique", {})`; store it in session and thread through the replay loop + `render_message(..., critique=None)`.
- [ ] **Step 2:** In `render_message`, for reasoned messages with `critique.get("checked")`, render a small badge (reuse `.cg-extrap`-style CSS or a new `.cg-faithful` class); if `critique.get("flagged")`, add an expander listing them.
- [ ] **Step 3: Manual verify** — run backend + Streamlit, ask a novel question, confirm the badge renders and (when the critic flags something) the expander lists it. UI is not unit-tested (Streamlit view).
- [ ] **Step 4: Commit** — `feat: show the faithfulness verdict in the chat UI`.

---

## Notes for the executor

- Keep the critic prompt JSON-only and short; the hardest part is making `supported` strict (a claim counts only if a *listed* principle backs it) without over-flagging ordinary connective prose.
- Do not thread `critic` into `main.py` — `/chat` returns the whole `answer()` dict, so `critique` flows through automatically; just confirm with an existing API test that the key appears.
- After Task 4, re-run the full suite and do one live eval A/B for the PR description's before/after numbers.
