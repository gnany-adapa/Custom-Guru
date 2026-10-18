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

## Default provider: Hyperspace

v1 defaults to a local Hyperspace proxy so there's no cloud cost while developing. To use a
different provider, implement a new adapter under `backend/app/llm/` and set
`LLM_PROVIDER` in `.env`.

## License

TBD (will be added before first public release).
