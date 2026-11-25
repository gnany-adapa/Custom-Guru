# Security Policy

## Supported versions

Custom Guru is in active development. Only the latest commit on `main` receives security fixes.

| Version | Supported |
|---|---|
| `main` (latest) | Yes |
| Older commits | No — please update to `main` |

## Reporting a vulnerability

**Do not open a public GitHub issue for a security vulnerability.**

There are two ways to report privately:

1. **GitHub private vulnerability reporting (preferred)** — use the
   [Report a vulnerability](../../security/advisories/new) button on the Security tab of this
   repo. GitHub keeps the report confidential and lets us coordinate a fix before disclosure.

2. **Email** — if you prefer not to use GitHub's flow, email the maintainer directly at
   `security@[placeholder — maintainer: fill in before publishing]`.

Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a minimal proof-of-concept
- Any suggested fix if you have one

We aim to acknowledge reports within **7 days** and to publish a fix within **30 days** for
confirmed vulnerabilities. We will credit you in the fix commit unless you prefer to remain
anonymous.

## Scope

### In scope

- Logic that reads or forwards the LLM provider API key (`.env` parsing, provider credential
  flow in `backend/app/llm/`)
- Prompt-injection paths that could cause the system to leak the API key, the system prompt, or
  corpus content to an end-user in an unintended way
- Retrieval or citation logic that could be manipulated to surface false attribution or fabricated
  quotes as verified

### Out of scope

- Vulnerabilities in third-party dependencies — please report those upstream
- The content of any persona corpus you configure locally: `data/` is gitignored and never
  committed; that data is entirely under your control
- Issues that only affect a locally-run instance with no external users

## Notes on data handling

Custom Guru is a **local-first tool**; it ships with no hosted service and no server you do not
run yourself. The only sensitive item it handles at runtime is the **LLM provider API key** (read
from `.env`, which is gitignored). Persona corpus transcripts under `data/` are also gitignored
and are never committed. The repo itself ships with no credentials and no persona data.
