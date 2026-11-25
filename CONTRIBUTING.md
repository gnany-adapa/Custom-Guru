# Contributing to Custom Guru

Thanks for your interest in contributing! This project is an open-source engine for
building **grounded, auditable** AI personas — every answer is either tied to a verified
source quote or clearly labelled as reasoned extrapolation. Contributions that strengthen
that groundedness/transparency are especially welcome.

## Ways to contribute
- **Bug reports & fixes** — open an issue with repro steps, or a PR with a failing test.
- **New LLM provider adapters** — see `backend/app/llm/provider_template.py`.
- **Retrieval / extraction quality** — chunking, the principle-extraction pipeline, dedup.
- **Docs & examples** — especially a "bring your own corpus" walkthrough.
- **Tests** — the suite is fast and offline-first; more coverage is always welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # see the file — Hyperspace is the default; OpenAI block is below it
```

**Running tests:**

Most tests are offline and need no LLM access. A handful (`@pytest.mark.live`) hit the
live embeddings/chat endpoint and require the proxy or an API key to be configured.

```bash
# All contributors — works with zero LLM access:
python -m pytest -q -m "not live"

# With a configured provider (Hyperspace, OpenAI, etc.):
python -m pytest -q
```

The offline suite covers: input validation, citation extraction, the reasoning engine,
the faithfulness critic, the eval harness, the ingest helpers (chunking/cleaning), and
the API layer. The live tests additionally exercise the actual embeddings and end-to-end
retrieval against a real model.

**PRs must pass the offline suite** (`pytest -m "not live"`) — this requires no LLM and
runs in CI on every pull request.

**No LLM provider yet?** You can still read the code, run all offline tests, open issues,
improve docs, and even implement a new provider from `provider_template.py` — none of that
requires the proxy.

## Workflow
1. **Open an issue first** for anything non-trivial, so we can agree on the approach.
2. Work on a **feature branch** (`feat/...`, `fix/...`), not `main`.
3. Keep changes focused; write or update **tests** (we practice test-first where practical).
4. Open a **pull request** describing the problem, the change, and how you verified it.
5. A maintainer will review. PRs must pass the offline test suite (`pytest -m "not live"`) and keep the diff readable.

## Conventions
- **Commit messages:** Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- **Never commit secrets or data** — `.env`, transcripts under `data/`, and local vector
  stores are gitignored. Keep it that way.
- **Groundedness is a contract:** any claim the system surfaces must trace to a verified
  source quote, or be explicitly marked as extrapolation. Don't weaken that.

## Code of Conduct
By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
