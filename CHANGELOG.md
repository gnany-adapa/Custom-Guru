# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Grounded Q&A with verified inline citations** — every quoted span is checked to
  literally exist in the source corpus; fabricated or paraphrased citations are dropped,
  not shown (`_quote_supported`, `_apply_citations`).
- **Principle-extraction pipeline** — offline map → consolidate → compress pipeline
  distils a corpus into a small, inspectable set of durable principles (`principles.json`),
  each backed by verified quotes.
- **Principle-based reasoning engine** — questions the person never addressed are answered
  by applying extracted principles through a visible factor → principle → inference trace,
  explicitly labelled as extrapolation and never passed off as a real quote.
- **Faithfulness critic** — checks each reasoned answer against the principles it cites and
  surfaces unsupported claims; verdict shown in the Streamlit UI.
- **Persona-neutral default configuration** — the repo ships with no persona data; corpus
  is gitignored and must be supplied by the user (`data/personas/<name>/transcripts/`).
- **Provider-agnostic LLM adapter layer** — `LLMProvider` interface with a Hyperspace
  local-proxy implementation; `provider_template.py` and an OpenAI adapter in `draft/`
  for external contributors.
- **Evaluation harness** — offline eval for groundedness, voice fidelity, transparency,
  and routing accuracy; A/B support for comparing critic-on vs. critic-off.
- **Streamlit demo UI** — chat interface with citation display and faithfulness verdict.
- **FastAPI backend** — `/chat` endpoint with Pydantic input validation and optional
  Langfuse tracing.
