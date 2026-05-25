import os
import shutil
from pathlib import Path
import streamlit as st

# Load .env manually so it works regardless of working directory
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8-sig").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from ingest import load_and_index, load_existing
from chain import build_chain, messages_to_langchain

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("RAG Chatbot")
st.caption("Index websites, then ask questions about their content.")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Sources")

    url_input = st.text_area(
        "URLs to index (one per line)",
        placeholder="https://example.com\nhttps://docs.example.com/guide",
        height=150,
    )

    if st.button("Ingest URLs", type="primary", use_container_width=True):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if not urls:
            st.warning("Enter at least one URL.")
        else:
            with st.spinner(f"Loading and indexing {len(urls)} URL(s)…"):
                try:
                    st.session_state.vectorstore = load_and_index(urls)
                    st.session_state.messages = []
                    st.success(f"Indexed {len(urls)} URL(s).")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    if st.button("Clear Database", use_container_width=True):
        chroma_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("messages", None)
        st.rerun()

    if "vectorstore" in st.session_state:
        st.success("Vector store loaded.")
    else:
        st.info("No data indexed yet.")


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Auto-load an existing vector store on first run
if "vectorstore" not in st.session_state:
    vs = load_existing()
    if vs:
        st.session_state.vectorstore = vs


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask something about the indexed content…"):
    if "vectorstore" not in st.session_state:
        st.error("Please ingest at least one URL using the sidebar first.")
        st.stop()

    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set. Add it to a .env file in the rag_chatbot folder.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            chain = build_chain(st.session_state.vectorstore)
            # Exclude the just-added user message from history
            history = messages_to_langchain(st.session_state.messages[:-1])
            result = chain.invoke({"input": prompt, "chat_history": history})
            answer = result["answer"]

        st.write(answer)

        with st.expander("Sources"):
            seen = set()
            for doc in result.get("context", []):
                src = doc.metadata.get("source", "Unknown")
                if src not in seen:
                    seen.add(src)
                    st.markdown(f"**{src}**")
                st.caption(doc.page_content[:300] + "…")

    st.session_state.messages.append({"role": "assistant", "content": answer})
