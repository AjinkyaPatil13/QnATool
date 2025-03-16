import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader  # Changed loader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    loader = PyPDFLoader(file_path)  # Using PDF loader without poppler dependency
    documents = loader.load()
    return documents

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Fixed newline character
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(doc_chunks, embeddings)

def create_chain(vectorstore):
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        ),
        chain_type="map_reduce",
        verbose=True
    )

# Streamlit UI setup
st.set_page_config(
    page_title="Chat with Doc",
    page_icon="📄",
    layout="centered"
)
st.title("Question Answering Tool")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload handling
uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
if uploaded_file:
    file_path = os.path.join(working_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize components only once
    if "conversation_chain" not in st.session_state:
        documents = load_document(file_path)
        vectorstore = setup_vectorstore(documents)
        st.session_state.conversation_chain = create_chain(vectorstore)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input handling
if prompt := st.chat_input("Ask your question..."):
    # Add user message to history and display
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": prompt})
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})