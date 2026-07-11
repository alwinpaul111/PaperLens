"""
PaperLens - Streamlit frontend.

Two modes, controlled by RUN_MODE env var:
  - "api"    (default): talks to the FastAPI backend over HTTP (use this
              when you deploy FastAPI + Streamlit as separate services,
              e.g. FastAPI on Render + Streamlit on Streamlit Cloud)
  - "direct": imports the RAG pipeline directly in-process (simplest for
              a single-container Streamlit Cloud deployment, same pattern
              you used for the hate speech detector)
"""
import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

RUN_MODE = os.getenv("RUN_MODE", "direct")
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PaperLens", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink: #10141C;
    --ink-panel: #161B26;
    --paper: #E8E1CF;
    --sage: #6E9C87;
    --sage-bright: #85B8A0;
    --gold: #C9A24B;
    --text-soft: #B8BEC9;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main title */
h1 {
    font-family: 'Source Serif 4', serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    color: var(--paper) !important;
}

h1::after {
    content: "";
    display: block;
    width: 48px;
    height: 3px;
    background: var(--gold);
    margin-top: 0.6rem;
    border-radius: 2px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--ink-panel);
    border-right: 1px solid rgba(232, 225, 207, 0.08);
}

section[data-testid="stSidebar"] h1 {
    font-size: 1.6rem !important;
}

section[data-testid="stSidebar"] h1::after {
    width: 32px;
}

section[data-testid="stSidebar"] h3 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--sage-bright) !important;
    font-weight: 500 !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    border-radius: 6px;
    border: 1px solid rgba(110, 156, 135, 0.4);
    background: transparent;
    color: var(--sage-bright);
    transition: all 0.15s ease;
}

.stButton > button:hover {
    background: rgba(110, 156, 135, 0.12);
    border-color: var(--sage-bright);
    color: var(--paper);
}

/* Primary process button gets filled treatment via nth default - keep secondary look for all, consistent and calm */

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 8px;
    border: 1px solid rgba(232, 225, 207, 0.06);
}

/* Source citation expander - styled like an academic footnote block */
[data-testid="stExpander"] {
    border: 1px solid rgba(201, 162, 75, 0.25) !important;
    border-radius: 6px !important;
    background: rgba(201, 162, 75, 0.04);
}

[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--gold) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border-radius: 6px;
}

/* Caption / footer text */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--text-soft) !important;
}

/* Divider - thinner, more intentional */
hr {
    border-color: rgba(232, 225, 207, 0.08) !important;
}
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-session"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, citations)

# ---------- Direct mode setup ----------
if RUN_MODE == "direct":
    from app.ingest import ingest_pdfs
    from app.rag_pipeline import answer_question, ConversationMemory
    from app.vector_store import index_exists
    from app.config import UPLOAD_DIR

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()


def index_ready() -> bool:
    if RUN_MODE == "direct":
        return index_exists()
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json().get("index_ready", False)
    except requests.RequestException:
        return False


def do_upload(uploaded_files):
    if RUN_MODE == "direct":
        saved_paths = []
        for f in uploaded_files:
            dest = Path(UPLOAD_DIR) / f.name
            with open(dest, "wb") as out:
                out.write(f.read())
            saved_paths.append(str(dest))
        return ingest_pdfs(saved_paths)
    else:
        files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
        r = requests.post(f"{API_URL}/upload", files=files, timeout=120)
        r.raise_for_status()
        return r.json()


def do_ask(question: str):
    if RUN_MODE == "direct":
        response = answer_question(question, memory=st.session_state.memory)
        return response.answer, [c.__dict__ for c in response.citations]
    else:
        r = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "session_id": st.session_state.session_id},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return data["answer"], data["citations"]


# ---------- Sidebar: upload ----------
with st.sidebar:
    st.title("PaperLens")
    st.caption("Chat with your research papers, grounded with citations.")
    st.divider()

    st.subheader("1. Upload papers")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDFs", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Extracting text, chunking, and embedding..."):
            try:
                result = do_upload(uploaded_files)
                st.session_state.chat_history = []
                if RUN_MODE == "direct":
                    st.session_state.memory.clear()
                else:
                    try:
                        requests.post(f"{API_URL}/reset", params={"session_id": st.session_state.session_id})
                    except requests.RequestException:
                        pass
                st.success(
                    f"Indexed {result['documents_processed']} document(s), "
                    f"{result['pages_processed']} pages, {result['chunks_created']} chunks. "
                    f"Previous documents and conversation were replaced."
                )
            except Exception as e:
                st.error(f"Failed to process PDFs: {e}")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        if RUN_MODE == "direct":
            st.session_state.memory.clear()
        else:
            try:
                requests.post(f"{API_URL}/reset", params={"session_id": st.session_state.session_id})
            except requests.RequestException:
                pass
        st.rerun()

    st.divider()
    st.caption("Built by Alwin Paul · DistilBERT-style project pattern")

# ---------- Main chat area ----------
st.header("Ask a question about your papers")

if not index_ready():
    st.info("Upload at least one PDF to get started.")

def render_citations(citations):
    with st.expander(f"{len(citations)} source(s)"):
        for c in citations:
            st.markdown(
                f"""<div style="display:flex; align-items:baseline; gap:0.6rem; margin-bottom:0.3rem;">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#10141C;
                                 background:#C9A24B; border-radius:50%; width:1.6rem; height:1.6rem;
                                 display:inline-flex; align-items:center; justify-content:center;
                                 flex-shrink:0; font-weight:600;">
                        {c['page_number']}
                    </span>
                    <span style="font-family:'Source Serif 4',serif; font-weight:600; color:#E8E1CF;">
                        {c['doc_name']}
                    </span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#6E9C87;">
                        relevance {c['relevance_score']}
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.caption(c["snippet"])


for question, answer, citations in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)
        if citations:
            render_citations(citations)

question = st.chat_input("Ask something about the uploaded papers...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and generating answer..."):
            try:
                answer, citations = do_ask(question)
            except Exception as e:
                answer, citations = f"Error: {e}", []
        st.write(answer)
        if citations:
            render_citations(citations)
    st.session_state.chat_history.append((question, answer, citations))
