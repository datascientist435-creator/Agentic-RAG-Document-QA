import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Agentic RAG Document Q&A", layout="wide")

# Persistent State Management
if "history" not in st.session_state:
    st.session_state.history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False

# Sidebar for Configuration
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    st.markdown("Supply an API key to run inference using an LLM model.")
    api_provider = st.selectbox("Select Inference API", ["Groq API (Fast & Free)", "OpenAI API"])
    
    # Identify which API is selected
    if "Groq" in api_provider:
        provider_name = "Groq"
        model_selection = "llama-3.3-70b-versatile"
    else:
        provider_name = "OpenAI"
        model_selection = "gpt-3.5-turbo"

    api_key = st.text_input(f"Enter your {provider_name} API Key", type="password")
    st.info("Your API key is only stored securely in this active session and will not be saved permanently.", icon="🔐")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload a PDF document to start chunking", type=["pdf"])

st.title("📄 Agentic RAG Document Q&A")
st.markdown("Upload any massive PDF document and start chatting with it. Powered by **LangChain**, **FAISS**, and **HuggingFace** Embeddings.")

# Vectorization Pipeline
@st.cache_resource
def process_document(file_path):
    # 1. Ingestion
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # 3. Embeddings Mapping (using absolutely free local huggingface sentence transformers)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Storage & Indexing in Vector DB
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

if uploaded_file and api_key and not st.session_state.vector_store_initialized:
    # Save uploaded bytes to a persistent temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("Processing document and generating mathematical vector embeddings..."):
        try:
            # Generate FAISS DB
            vector_store = process_document(tmp_file_path)
            
            # Setup LLM Generation pipeline
            if provider_name == "Groq":
                llm = ChatGroq(groq_api_key=api_key, model_name=model_selection, temperature=0)
            else:
                llm = ChatOpenAI(openai_api_key=api_key, model_name=model_selection, temperature=0)
                
            # Connect Conversational Memory and Retrieval Chain
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}), # fetch top 3 most relevant chunks
                memory=memory
            )
            st.session_state.vector_store_initialized = True
            st.success("Document analyzed, chunked, embedded, and Vector Database is primed!")
        except Exception as e:
            st.error(f"Error initializing the RAG pipeline or API connection: {str(e)}")
            st.stop()

# Gatekeepers
if not uploaded_file:
    st.warning("Please upload a PDF document and supply your API key in the sidebar.")
    st.stop()
elif not api_key:
    st.warning("Please enter your API Key in the sidebar to authenticate the LLM.")
    st.stop()
elif not st.session_state.vector_store_initialized:
    st.warning("Waiting for initialization...")
    st.stop()

# --- Main Chat Interface ---

# Render history messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Chat Input
user_query = st.chat_input("Ask any question regarding your uploaded document...")

if user_query:
    # Append to state and render immediately
    st.session_state.history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Perform Vector Search & Generation
    if st.session_state.qa_chain:
        with st.spinner("Agent is retrieving semantically relevant context chunks and generating an answer..."):
            try:
                # Let LangChain handle the retrieval calculation and LLM prompt
                result = st.session_state.qa_chain({"question": user_query})
                response = result['answer']
                
                # Append assistant response to UI
                st.session_state.history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
            except Exception as e:
                st.error(f"Error executing retrieval chain calculation: {str(e)}")
