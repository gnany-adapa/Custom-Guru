"""Custom Guru — Streamlit chat UI.

A thin client over the FastAPI backend: it sends questions to POST /chat and renders the
grounded answer plus its sources. All the retrieval/LLM logic lives in the backend, so this
file can be swapped for a React frontend later without touching the brains.

Run (with the backend already up):
    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Custom Guru", page_icon="🧠")
st.title("🧠 Custom Guru")
st.caption("Chat with Alex Hormozi — every answer is grounded in his own talks.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

if prompt := st.chat_input("Ask Alex anything he's talked about…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat", json={"question": prompt}, timeout=120
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", []) if data.get("covered") else []
            except requests.RequestException as exc:
                answer = (
                    f"⚠️ Couldn't reach the backend at `{BACKEND_URL}`. "
                    f"Is it running (`uvicorn backend.app.main:app`)?\n\n`{exc}`"
                )
                sources = []
        st.markdown(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
