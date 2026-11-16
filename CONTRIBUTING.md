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
pip install -r requirements.txt
cp .env.example .env            # fill in your LLM provider keys
python -m pytest -q             # run the tests
```

## Workflow
1. **Open an issue first** for anything non-trivial, so we can agree on the approach.
2. Work on a **feature branch** (`feat/...`, `fix/...`), not `main`.
3. Keep changes focused; write or update **tests** (we practice test-first where practical).
4. Open a **pull request** describing the problem, the change, and how you verified it.
5. A maintainer will review. PRs should pass `pytest` and keep the diff readable.

## Conventions
- **Commit messages:** Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- **Never commit secrets or data** — `.env`, transcripts under `data/`, and local vector
  stores are gitignored. Keep it that way.
- **Groundedness is a contract:** any claim the system surfaces must trace to a verified
  source quote, or be explicitly marked as extrapolation. Don't weaken that.

## Code of Conduct
By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
