# 🤖 QnA Tool – Ask Questions from Your PDFs

**QnA Tool** is a document question-answering web app that lets you upload a PDF and ask natural language questions about it. Retrieval is grounded: every answer is built only from the retrieved passages and cites the page it came from. Built with **Streamlit**, **LangChain**, **FAISS**, and an open-weight LLM served by **Groq**.

---

## 🚀 Features

- 📄 Upload any text-based PDF
- ❓ Ask questions, including follow-ups that refer back to earlier turns
- 📌 Page-level citations, with the retrieved chunks shown under each answer
- 🚫 Refuses to answer from outside knowledge — says so when the document doesn't cover it
- ⚡ Token streaming, so answers appear as they're generated
- ♻️ Index is cached per document, so re-opening the same PDF skips re-embedding
- 🔍 Normalised embeddings in a FAISS index

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Orchestration**: LangChain (`create_history_aware_retriever` + `create_retrieval_chain`)
- **Language Model**: `openai/gpt-oss-120b` via Groq API
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, normalised
- **Vector Store**: FAISS (persisted to disk per document)
- **PDF Loader**: PyPDFLoader (LangChain Community)

---

## 📁 Project Structure

```
app.py        Streamlit UI, session state, streaming and source rendering
rag_core.py   Loading, chunking, indexing, prompts and chain construction
```

The pipeline is kept out of the UI module so it can be imported and tested without starting Streamlit.

---

## 📸 Screenshot

<!-- Replace this with your actual image link -->
![QnA Tool UI](https://drive.google.com/uc?export=view&id=1ajrs6uOqzKr1ILN7imcd3f0IKvyLzb7d)

---

## 🌐 Live Demo

👉 [Try it on Streamlit Cloud](https://qnatool.streamlit.app/)

---

## 🧭 How It Works

1. Upload a PDF; it is hashed so the same file is never re-indexed
2. Text is extracted with PyPDFLoader (scans without OCR are rejected with a clear error)
3. Chunks are split with `RecursiveCharacterTextSplitter` (800 chars, 150 overlap)
4. Chunks are embedded with MiniLM and stored in a FAISS index, saved to disk
5. Your question is rewritten into a standalone query using the chat history
6. The top-k chunks are retrieved and rendered with `[page N]` markers
7. The LLM answers using only that context, citing pages, and streams the answer back

---

## 🔍 Use Cases

- Quickly understand large academic papers or reports
- Question answering over legal or policy documents
- Automated helpdesk/document assistant
- Study tool for students

---

## ▶️ Run Locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` with your Groq key (free at console.groq.com):

```
GROQ_API_KEY=your_key_here
```

Then:

```bash
streamlit run app.py
```

### Optional environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq retires models; override here if the default 404s |
| `RAG_CACHE_DIR` | `.faiss_cache/` | Where indexes are persisted |
| `RAG_VERBOSE` | unset | Set to `1` for LangChain debug logging |

---

## 🔧 Future Improvements

- Support for more file types (DOCX, TXT)
- Multi-file document support
- OCR fallback for scanned PDFs
- Answer-quality evaluation harness (groundedness, citation accuracy)

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Ajinkya Patil**
- GitHub: [@AjinkyaPatil13](https://github.com/AjinkyaPatil13)
- LinkedIn: [Ajinkya Patil](https://linkedin.com/in/ajinkya-patil-11a0a32a6/)
