"""Core RAG pipeline: PDF loading, chunking, indexing, and chain construction.

Kept separate from the Streamlit UI so it can be imported and tested without
starting the app.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Sequence

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:  # older envs that only ship the community package
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DEFAULT_K = 4
NOT_FOUND_MESSAGE = "I couldn't find that in the document."


class DocumentError(RuntimeError):
    """Raised when a PDF cannot be turned into searchable text."""


# --------------------------------------------------------------------------- #
# Loading and indexing
# --------------------------------------------------------------------------- #

def load_pdf(file_bytes: bytes) -> List[Document]:
    """Parse a PDF from raw bytes, cleaning up the temp file afterwards."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    try:
        documents = PyPDFLoader(path).load()
    finally:
        os.unlink(path)

    documents = [doc for doc in documents if doc.page_content.strip()]
    if not documents:
        raise DocumentError(
            "No extractable text found. This PDF is most likely a scan, which "
            "needs OCR before it can be indexed."
        )
    return documents


def split_documents(documents: Sequence[Document]) -> List[Document]:
    """Split on the largest natural boundary that fits, not on raw character count."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise DocumentError("The document produced no usable text chunks.")
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """Normalised embeddings so FAISS's L2 distance ranks the same as cosine."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    chunks: Sequence[Document],
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    return FAISS.from_documents(list(chunks), embeddings or get_embeddings())


def save_vectorstore(store: FAISS, path: str) -> None:
    store.save_local(path)


def load_vectorstore(
    path: str,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    # Only ever load indexes this app wrote itself; the pickle inside a FAISS
    # index can execute arbitrary code, hence the explicit opt-in.
    return FAISS.load_local(
        path,
        embeddings or get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# --------------------------------------------------------------------------- #
# Prompts and chain
# --------------------------------------------------------------------------- #

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, rewrite the "
            "question as a standalone question that can be understood without "
            "the history. Do not answer it. Return only the rewritten question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a single PDF using only the retrieved "
            "context below. Each context block begins with a page marker, for "
            "example [page 4].\n\n"
            "Rules:\n"
            "- Use only the context. Never rely on outside knowledge.\n"
            f"- If the answer is not in the context, reply exactly: {NOT_FOUND_MESSAGE!r}\n"
            "- When you state a fact, cite its page marker like (page 4).\n\n"
            "Context:\n{context}",
        ),
        ("human", "{input}"),
    ]
)

# Renders each retrieved chunk with its 1-based page number so the model can cite it.
DOCUMENT_PROMPT = PromptTemplate.from_template("[page {page}]\n{page_content}")


def annotate_pages(documents: Sequence[Document]) -> List[Document]:
    """Return copies with 1-based page numbers (PDF metadata is 0-based).

    Copies, not in place: the cached vector store hands out the same Document
    objects on every query, so mutating them would drift the numbering.
    """
    return [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "page": int(doc.metadata.get("page", 0)) + 1},
        )
        for doc in documents
    ]


def build_chain(
    vectorstore: FAISS,
    model: str = DEFAULT_MODEL,
    k: int = DEFAULT_K,
    temperature: float = 0.0,
):
    """History-aware retriever + grounded stuffing chain, returning answer + context."""
    llm = ChatGroq(model=model, temperature=temperature)

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONDENSE_PROMPT
    )
    retriever_with_pages = history_aware_retriever | RunnableLambda(annotate_pages)

    answer_chain = create_stuff_documents_chain(
        llm, QA_PROMPT, document_prompt=DOCUMENT_PROMPT
    )

    return create_retrieval_chain(retriever_with_pages, answer_chain)
