---
name: Bug report
about: Something is broken or behaving unexpectedly
labels: bug
---

## What happened?

<!-- Brief description of the unexpected behaviour. One sentence is fine. -->

## Steps to reproduce

1. 
2. 
3. 

## Expected behaviour

<!-- What should have happened? -->

## Actual behaviour

<!-- What actually happened? Paste any error traceback, response JSON, or screenshot below. -->

```
(paste here)
```

## Environment

| Field | Value |
|---|---|
| Python version (`python --version`) | |
| LLM provider (`LLM_PROVIDER` in `.env`) | |
| Provider model (`LLM_MODEL` / `EMBED_MODEL`) | |
| Corpus size (approximate number of transcript files) | |
| OS | |

## Which component does this affect?

<!-- Check all that apply -->

- [ ] Ingest / principle extraction (`backend/app/persona/`)
- [ ] Retrieval / RAG (`backend/app/rag/`)
- [ ] Chat / reasoning (`backend/app/chat/`)
- [ ] Faithfulness critic (`backend/app/chat/critic.py`)
- [ ] LLM provider adapter (`backend/app/llm/`)
- [ ] API layer (`backend/app/main.py`)
- [ ] Streamlit UI (`frontend/`)
- [ ] Other / unsure

## Notes

<!-- Did this used to work? Any guesses about the cause? Anything else relevant? -->
