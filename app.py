import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Agentic RAG Document Q&A", layout="wide")

# Persistent State Management
if "history" not in st.session_state:
    st.session_state.history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False

# Sidebar for Configuration
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    st.markdown("Supply an API key to run inference using an LLM model.")
    api_provider = st.selectbox("Select Inference API", ["Groq API (Fast & Free)", "OpenAI API"])

    if "Groq" in api_provider:
        provider_name = "Groq"
        model_selection = "llama-3.3-70b-versatile"
    else:
        provider_name = "OpenAI"
        model_selection = "gpt-3.5-turbo"

    api_key = st.text_input(f"Enter your {provider_name} API Key", type="password")
    st.info("Your API key is only stored in this active session.", icon="🔐")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

st.title("📄 Agentic RAG Document Q&A")
st.markdown("Upload any PDF document and chat with it. Powered by **LangChain**, **FAISS**, and **HuggingFace** Embeddings.")

# RAG Prompt Template
RAG_TEMPLATE = """You are a helpful AI assistant. Answer the user's question based ONLY on the following context extracted from their document. If the context doesn't contain the answer, say "I couldn't find that information in the document."

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# Vectorization Pipeline
@st.cache_resource
def process_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(history):
    formatted = ""
    for msg in history[-6:]:  # Last 3 exchanges for context
        formatted += f"{msg['role'].upper()}: {msg['content']}\n"
    return formatted if formatted else "No previous conversation."

if uploaded_file and api_key and not st.session_state.vector_store_initialized:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("Processing document and generating vector embeddings..."):
        try:
            vector_store = process_document(tmp_file_path)
            st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            if provider_name == "Groq":
                st.session_state.llm = ChatGroq(groq_api_key=api_key, model_name=model_selection, temperature=0)
            else:
                st.session_state.llm = ChatOpenAI(openai_api_key=api_key, model_name=model_selection, temperature=0)

            st.session_state.vector_store_initialized = True
            st.success("Document analyzed and Vector Database is primed!")
        except Exception as e:
            st.error(f"Error initializing pipeline: {str(e)}")
            st.stop()

# Gatekeepers
if not uploaded_file:
    st.warning("Please upload a PDF document and supply your API key in the sidebar.")
    st.stop()
elif not api_key:
    st.warning("Please enter your API Key in the sidebar.")
    st.stop()
elif not st.session_state.vector_store_initialized:
    st.warning("Waiting for initialization...")
    st.stop()

# Render history messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Chat Input
user_query = st.chat_input("Ask any question regarding your uploaded document...")

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    if st.session_state.retriever and st.session_state.llm:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                # Retrieve relevant docs
                docs = st.session_state.retriever.invoke(user_query)
                context = format_docs(docs)
                chat_history = format_chat_history(st.session_state.history)

                # Build and invoke the chain using LCEL
                chain = rag_prompt | st.session_state.llm | StrOutputParser()
                response = chain.invoke({
                    "context": context,
                    "chat_history": chat_history,
                    "question": user_query
                })

                st.session_state.history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
