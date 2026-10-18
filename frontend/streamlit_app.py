"""Custom Guru — Streamlit chat UI.

A thin client over the FastAPI backend: it sends questions to POST /chat and renders the
grounded answer plus its sources. All retrieval/LLM logic lives in the backend, so this
file can be swapped for a React frontend later without touching the brains.

Run (with the backend already up):
    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PERSONA_NAME = "Alex Hormozi"
PERSONA_INITIALS = "AH"
GREETING = "What do you want to figure out?"
SUBTITLE = (
    "Ask Alex anything he's talked about. You'll get his actual thinking — "
    "grounded in his own talks, never made up."
)
STARTERS = [
    "How should I price my offer?",
    "How do I get my time back?",
    "How do I get more clients?",
]

st.set_page_config(
    page_title="Custom Guru",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- styles
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg: #ffffff;
  --panel: #f7f6f4;
  --ink: #1c1b18;
  --muted: #78746b;
  --line: #e6e3dd;
  --gold: #b07d2b;
  --gold-weak: #f3ebd9;
}

html, body, [class*="st-"], .stApp, button, input, textarea {
  font-family: 'Schibsted Grotesk', system-ui, sans-serif !important;
}

/* Keep Material icon ligatures working — the UI font must not clobber icon fonts. */
[data-testid="stIconMaterial"],
span.material-symbols-rounded,
span.material-symbols-outlined,
.material-icons {
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
    'Material Icons', 'Material Icons Round' !important;
}

.stApp { background: var(--bg); }

/* Center + constrain the main chat column */
[data-testid="stMainBlockContainer"] {
  max-width: 760px;
  padding-top: 2.5rem;
  padding-bottom: 7rem;
}

/* Hide only the Deploy button and the main menu for a cleaner canvas — NOT the
   whole toolbar, which also holds the "Expand sidebar" control used to reopen a
   collapsed sidebar. Keeping the toolbar means collapse/expand works normally. */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stMainMenuButton"] { display: none; }

/* ---------------- Sidebar ---------------- */
[data-testid="stSidebar"] {
  background: var(--panel);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div { padding-top: 1.1rem; }

.cg-brand {
  display: flex; align-items: center; gap: .6rem;
  padding: 0 .25rem .9rem; margin-bottom: .4rem;
}
.cg-brand-mark {
  width: 30px; height: 30px; border-radius: 9px;
  background: var(--ink); color: #fff;
  display: grid; place-items: center;
  font-weight: 700; font-size: .78rem; letter-spacing: .02em;
}
.cg-brand-name { font-weight: 600; font-size: 1.08rem; color: var(--ink); }

.cg-persona {
  display: flex; align-items: center; gap: .7rem;
  padding: .7rem .75rem; margin: .4rem .1rem;
  background: #fff; border: 1px solid var(--line); border-radius: 14px;
}
.cg-persona-avatar {
  width: 38px; height: 38px; border-radius: 50%;
  background: var(--gold); color: #fff;
  display: grid; place-items: center;
  font-weight: 700; font-size: .85rem; flex: none;
}
.cg-persona-name { font-weight: 600; font-size: .95rem; color: var(--ink); line-height: 1.2; }
.cg-persona-role { font-size: .78rem; color: var(--muted); margin-top: .1rem; }

.cg-section { font-size: .72rem; color: var(--muted); letter-spacing: .04em;
  padding: .2rem .35rem; margin-top: .5rem; }

.cg-status { display: flex; align-items: center; gap: .45rem;
  font-size: .8rem; color: var(--muted); padding: .35rem; }
.cg-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.cg-dot.ok { background: #2e9e5b; }
.cg-dot.off { background: #c0493b; }

/* ---------------- Hero (empty state) ---------------- */
.cg-hero { text-align: center; margin: 9vh auto 2.2rem; }
.cg-hero-mark {
  width: 56px; height: 56px; border-radius: 50%;
  background: var(--gold); color: #fff; margin: 0 auto 1.2rem;
  display: grid; place-items: center; font-weight: 700; font-size: 1.2rem;
}
.cg-hero-title { font-size: 1.95rem; font-weight: 600; color: var(--ink);
  letter-spacing: -.02em; margin: 0; }
.cg-hero-sub { font-size: 1rem; color: var(--muted); max-width: 30rem;
  margin: .8rem auto 0; line-height: 1.5; }

/* ---------------- Starter chips (buttons) ---------------- */
[data-testid="stMainBlockContainer"] .stButton button {
  width: 100%; text-align: left; white-space: normal;
  background: #fff; color: var(--ink);
  border: 1px solid var(--line); border-radius: 13px;
  padding: .75rem .9rem; font-size: .9rem; font-weight: 400;
  box-shadow: none; transition: border-color .12s ease, background .12s ease;
}
[data-testid="stMainBlockContainer"] .stButton button:hover {
  border-color: var(--gold); background: var(--gold-weak);
}
[data-testid="stMainBlockContainer"] .stButton button p {
  white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.35;
}

/* Sidebar "New conversation" button */
[data-testid="stSidebar"] .stButton button {
  width: 100%; background: var(--ink); color: #fff;
  border: none; border-radius: 11px; padding: .6rem .8rem;
  font-weight: 500; font-size: .9rem;
}
[data-testid="stSidebar"] .stButton button:hover { background: #000; color: #fff; }

/* ---------------- Chat messages ---------------- */
[data-testid="stChatMessage"] { background: transparent; padding: .35rem 0; }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--gold); }
[data-testid="stChatMessageAvatarAssistant"] svg { fill: #fff; color: #fff; }
[data-testid="stChatMessageAvatarUser"] { background: var(--ink); }
[data-testid="stChatMessageAvatarUser"] svg { fill: #fff; color: #fff; }

.cg-sources { font-size: .78rem; color: var(--muted); margin-top: .5rem;
  padding-top: .5rem; border-top: 1px solid var(--line); }

/* ---------------- Input pill ---------------- */
[data-testid="stBottomBlockContainer"] { max-width: 760px; margin: 0 auto; }
[data-testid="stChatInput"] {
  border: 1px solid var(--line); border-radius: 26px;
  background: var(--panel); box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--gold); background: #fff; }
[data-testid="stChatInput"] textarea { font-size: .98rem; }
[data-testid="stChatInputSubmitButton"] { border-radius: 50%; }
[data-testid="stChatInputSubmitButton"]:disabled { color: var(--muted); }
[data-testid="stChatInputSubmitButton"]:not(:disabled) { background: var(--gold); }
[data-testid="stChatInputSubmitButton"]:not(:disabled) svg,
[data-testid="stChatInputSubmitButton"]:not(:disabled) [data-testid="stIconMaterial"] {
  color: #fff; fill: #fff;
}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- backend
def get_health() -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def ask_backend(question: str) -> dict:
    try:
        r = requests.post(f"{BACKEND_URL}/chat", json={"question": question}, timeout=120)
        r.raise_for_status()
        data = r.json()
        return {
            "answer": data["answer"],
            "sources": data.get("sources", []) if data.get("covered") else [],
        }
    except requests.RequestException as exc:
        return {
            "answer": (
                f"Can't reach the backend at `{BACKEND_URL}`. "
                f"Start it with `uvicorn backend.app.main:app`, then try again.\n\n`{exc}`"
            ),
            "sources": [],
        }


# --------------------------------------------------------------------------- state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

health = get_health()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(
        '<div class="cg-brand"><span class="cg-brand-mark">CG</span>'
        '<span class="cg-brand-name">Custom Guru</span></div>',
        unsafe_allow_html=True,
    )
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    st.markdown('<div class="cg-section">Talking to</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="cg-persona"><div class="cg-persona-avatar">{PERSONA_INITIALS}</div>'
        f'<div><div class="cg-persona-name">{PERSONA_NAME}</div>'
        f'<div class="cg-persona-role">Grounded in his own talks</div></div></div>',
        unsafe_allow_html=True,
    )

    if health:
        dot, label = "ok", f"Connected to {health.get('provider', '')}"
    else:
        dot, label = "off", "Backend offline"
    st.markdown(
        f'<div class="cg-status"><span class="cg-dot {dot}"></span>{label}</div>',
        unsafe_allow_html=True,
    )


def render_message(role: str, content: str, sources: list[str] | None = None) -> None:
    avatar = "🧠" if role == "assistant" else "🧑"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if sources:
            st.markdown(
                f'<div class="cg-sources">From {PERSONA_NAME}’s talks: '
                f'{", ".join(sources)}</div>',
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- main
prompt = st.chat_input(f"Ask {PERSONA_NAME} anything he's covered…")
if st.session_state.pending and not prompt:
    prompt = st.session_state.pending
st.session_state.pending = None

show_greeting = not st.session_state.messages and not prompt
if show_greeting:
    st.markdown(
        f'<div class="cg-hero"><div class="cg-hero-mark">{PERSONA_INITIALS}</div>'
        f'<h1 class="cg-hero-title">{GREETING}</h1>'
        f'<p class="cg-hero-sub">{SUBTITLE}</p></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(STARTERS))
    for col, starter in zip(cols, STARTERS):
        if col.button(starter, key=f"starter-{starter}"):
            st.session_state.pending = starter
            st.rerun()

# Replay the conversation so far.
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("sources"))

# Handle a new question (from the input box or a starter chip).
if prompt:
    render_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking…"):
        result = ask_backend(prompt)
    render_message("assistant", result["answer"], result["sources"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
    if show_greeting:
        st.rerun()
