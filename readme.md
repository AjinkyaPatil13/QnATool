# 🤖 QnA Tool – Ask Questions from Your PDFs

**QnA Tool** is a document question-answering web app that allows users to upload PDF files and ask natural language questions based on the content. It is built using **Streamlit**, **LangChain**, **LLaMA 3 (via Groq API)**, **PyPDFLoader**, and **FAISS**, and is deployed on **Streamlit Cloud**. This tool is perfect for students, researchers, and professionals who want to extract insights from documents without reading them word by word.

---

## 🚀 Features

- 📄 Upload any PDF file  
- ❓ Ask questions and get context-aware answers from the document  
- ⚡ Ultra-fast inference using Groq API  
- 🧠 Powered by LLaMA 3 for accurate answers  
- 🔍 Embedding-based retrieval using FAISS  
- 💬 Clean and interactive Streamlit UI

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit  
- **LLM Orchestration**: LangChain  
- **Language Model**: LLaMA 3 via Groq API  
- **Embeddings & Retrieval**: FAISS  
- **PDF Loader**: PyPDFLoader (LangChain Community)

---

## 📸 Screenshot

<!-- Replace this with your actual image link -->
![QnA Tool UI](https://drive.google.com/uc?export=view&id=1ajrs6uOqzKr1ILN7imcd3f0IKvyLzb7d)


---

## 🌐 Live Demo

👉 [Try it on Streamlit Cloud](https://qnatool.streamlit.app/)

---

## 🧭 How It Works

1. Upload a PDF document  
2. The file is loaded and chunked using PyPDFLoader  
3. Embeddings are generated and stored in FAISS  
4. User inputs a question  
5. LangChain retrieves relevant chunks from the vector store  
6. Groq's LLaMA 3 model answers the question using retrieved context

---

## 🔍 Use Cases

- Quickly understand large academic papers or reports  
- Question answering over legal or policy documents  
- Automated helpdesk/document assistant  
- Study tool for students

---

## 🔧 Future Improvements

- Support for more file types (DOCX, TXT)  
- Source highlighting with citations  
- Chat history and session memory  
- Model selection (Groq, OpenAI, HuggingFace, etc.)  
- Multi-file document support

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Ajinkya Patil**  
- GitHub: [@AjinkyaPatil13](https://github.com/AjinkyaPatil13)  
- LinkedIn: [Ajinkya Patil](https://linkedin.com/in/ajinkya-patil-11a0a32a6/)
