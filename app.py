"""Streamlit UI for the PDF question-answering app.

All retrieval and chain logic lives in rag_core.py so it can be tested without
starting Streamlit.
"""

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.globals import set_verbose
from langchain_core.messages import AIMessage, HumanMessage

import rag_core as rag

load_dotenv()
set_verbose(os.getenv("RAG_VERBOSE") == "1")

CACHE_DIR = Path(os.getenv("RAG_CACHE_DIR", Path(__file__).parent / ".faiss_cache"))

st.set_page_config(
    page_title="Chat with Doc",
    page_icon="📄",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load the embedding model once per process, not once per session."""
    return rag.get_embeddings()


@st.cache_resource(show_spinner=False)
def get_vectorstore(doc_id: str, _file_bytes: bytes):
    """Reuse the index for a given file, from disk if possible.

    `_file_bytes` is deliberately excluded from the cache key: hashing multiple
    megabytes on every rerun is wasted work when `doc_id` is already a hash of
    the file contents.
    """
    cache_path = CACHE_DIR / doc_id
    if cache_path.exists():
        return rag.load_vectorstore(str(cache_path), get_embeddings())

    chunks = rag.split_documents(rag.load_pdf(_file_bytes))
    store = rag.build_vectorstore(chunks, get_embeddings())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rag.save_vectorstore(store, str(cache_path))
    return store


def stream_answer(chain, question: str, history: list):
    """Yield answer tokens and stash the retrieved chunks for the sources panel."""
    st.session_state.sources = []
    for chunk in chain.stream({"input": question, "chat_history": history}):
        if "context" in chunk:
            st.session_state.sources = chunk["context"]
        if "answer" in chunk:
            yield chunk["answer"]


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} chunks)"):
        for doc in sources:
            preview = doc.page_content.strip()
            if len(preview) > 600:
                preview = preview[:600] + "…"
            st.markdown(f"**Page {doc.metadata.get('page', '?')}**")
            st.caption(preview)


def main() -> None:
    st.title("Question Answering Tool")
    st.caption("Upload a PDF, then ask questions about it.")

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Add it to .env or Streamlit secrets.")
        return

    with st.sidebar:
        k = st.slider("Chunks to retrieve", 1, 10, rag.DEFAULT_K)
        model = st.text_input(
            "Groq model", os.getenv("GROQ_MODEL", rag.DEFAULT_MODEL)
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources = []
            st.rerun()
        if st.session_state.get("doc_id"):
            st.caption(f"Index: {st.session_state.doc_id}")

    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    if uploaded_file is None:
        st.info("Upload a PDF to get started.")
        return

    file_bytes = uploaded_file.getvalue()
    doc_id = hashlib.sha256(file_bytes).hexdigest()[:12]

    # Rebuild whenever the file changes, so a second upload is never silently
    # answered from the first document's index.
    if st.session_state.get("doc_id") != doc_id:
        try:
            with st.spinner(
                "Indexing document… (the first run also downloads the embedding model)"
            ):
                st.session_state.vectorstore = get_vectorstore(doc_id, file_bytes)
        except Exception as exc:
            st.error(f"Could not index this PDF: {exc}")
            return
        st.session_state.doc_id = doc_id
        st.session_state.messages = []
        st.session_state.sources = []

    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    if prompt := st.chat_input("Ask a question about the document…"):
        st.session_state.messages.append(HumanMessage(prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        chain = rag.build_chain(st.session_state.vectorstore, model=model, k=k)
        history = st.session_state.messages[:-1]

        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(stream_answer(chain, prompt, history))
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return
            render_sources(st.session_state.get("sources"))

        st.session_state.messages.append(AIMessage(answer))


main()
