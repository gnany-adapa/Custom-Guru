# Custom Guru 🧠

**Chat with an AI avatar of a thought leader that answers from their *actual* recorded words — not generic impersonation.**

Custom Guru is an open-source template. You bring your own transcripts (talks,
interviews, writings) for any persona you like; the app grounds every answer in that
source material and replies in the persona's voice. Ships with **zero persona data** — it's
the engine, not someone's scraped life.

> **v1 status:** "Grounded Hormozi" — single persona, chat interface, answers only from the
> transcripts you provide. Reasoning traces, citations, multiple personas, and a
> multi-agent "board of directors" are planned for later versions.

---

## Architecture

```
Streamlit UI  ──HTTP──▶  FastAPI backend
                              │
                 ┌────────────┼─────────────┐
                 ▼            ▼              ▼
            Retriever    Persona prompt   LLMProvider (adapter)
             (Chroma)                      └─ Hyperspace (default)
                                           └─ OpenAI / Claude / … (pluggable)
```

- **Provider-agnostic LLM layer** — the app talks to one `LLMProvider` interface, never a
  vendor SDK directly. Swap providers with an env var.
- **Grounded-only** — retrieves relevant transcript chunks and answers from them; if
  nothing relevant is found, the avatar says it hasn't covered that topic.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in your keys
# drop 3-4 .txt transcripts into data/personas/hormozi/transcripts/
python -m backend.app.rag.ingest        # build the vector store
uvicorn backend.app.main:app --reload   # start the API
streamlit run frontend/streamlit_app.py # start the UI
```

## Bring your own transcripts

Place plain-text files under `data/personas/<name>/transcripts/`. This folder is
gitignored — your data never gets committed.

## Using a different LLM provider

The app never calls a vendor SDK directly. It depends on one small interface,
`LLMProvider` ([`backend/app/llm/base.py`](backend/app/llm/base.py)), with just two methods:

```python
def generate(self, messages, system=None, max_tokens=1000) -> str   # chat completion
def embed(self, texts) -> list[list[float]]                         # embeddings
```

v1 defaults to a local **Hyperspace** proxy (`LLM_PROVIDER=hyperspace`) so there's no cloud
cost while developing. To plug in a different provider — OpenAI, Anthropic, Gemini, a local
model, anything — you write one adapter; no other code changes.

**Steps to add a provider (e.g. `openai`):**

1. Copy [`backend/app/llm/provider_template.py`](backend/app/llm/provider_template.py) to a
   new file, e.g. `backend/app/llm/openai_provider.py`.
2. Implement `generate()` and `embed()` using your provider's API or SDK. Translate *to and
   from* the neutral types in the signatures — the rest of the app only sees those.
3. Register the class in [`backend/app/llm/factory.py`](backend/app/llm/factory.py):
   ```python
   _PROVIDERS = {
       "hyperspace": HyperspaceProvider,
       "openai": OpenAIProvider,   # <- your new adapter
   }
   ```
4. Set `LLM_PROVIDER=openai` (and your keys) in `.env`. Done.

[`hyperspace.py`](backend/app/llm/hyperspace.py) is a complete worked example to copy from.

> Note: `LLMProvider` currently bundles chat **and** embeddings into one provider. If you
> want chat from one vendor and embeddings from another, that split is planned for a later
> version — for now a single adapter handles both.

## License

TBD (will be added before first public release).
