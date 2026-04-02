# 📄 Agentic RAG Document Q&A 🤖

An interactive AI-powered web application that lets users **upload any PDF document and chat with it** in real-time using Retrieval-Augmented Generation (RAG). Built with **Python**, **Streamlit**, **LangChain**, and **FAISS**.

> 🌐 **Live Demo:** [https://(https://datascientist435-creator-agentic-rag-document-qa-app.streamlit.app)
datascientist435-creator-agentic-rag-document-qa-app.streamlit.app]
---

## 🌟 Key Features

| Feature | Description |
|:--------|:------------|
| **PDF Upload & Parsing** | Upload any PDF — the app extracts and processes all text automatically |
| **Semantic Chunking** | Large documents are split into intelligent chunks for precise retrieval |
| **Vector Embeddings** | Text is converted into mathematical vectors using HuggingFace's `all-MiniLM-L6-v2` model |
| **In-Memory Vector DB** | Meta's FAISS library performs lightning-fast similarity searches |
| **Pluggable LLMs** | Switch between **Groq (Llama 3.3 70B — Free)** and **OpenAI (GPT-3.5 Turbo)** |
| **Chat Memory** | The AI remembers your previous questions for contextual follow-up answers |
| **Session Reset** | One-click reset to change API keys, switch providers, or upload a new document |

---

## 🛠️ Technical Architecture

```
User uploads PDF
       ↓
┌──────────────────────┐
│  1. INGESTION        │  PyPDFLoader extracts raw text from PDF bytes
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  2. CHUNKING         │  RecursiveCharacterTextSplitter breaks text into
│                      │  1000-char chunks with 200-char overlap
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  3. EMBEDDING        │  HuggingFace sentence-transformers (all-MiniLM-L6-v2)
│                      │  converts each chunk into a 384-dimensional vector
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  4. VECTOR STORAGE   │  FAISS indexes all vectors in-memory for
│                      │  ultra-fast similarity search
└──────────┬───────────┘
           ↓
   User asks a question
           ↓
┌──────────────────────┐
│  5. RETRIEVAL        │  FAISS retrieves the top K=3 most semantically
│                      │  relevant chunks using dot-product similarity
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  6. GENERATION       │  Retrieved chunks + chat history are sent to
│                      │  Groq/OpenAI LLM via LangChain LCEL pipeline
│                      │  to generate a contextual, cited answer
└──────────────────────┘
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10+ installed
- A free **Groq API Key** ([Get one here](https://console.groq.com/keys)) or an **OpenAI API Key**

### 1. Clone the Repository
```bash
git clone https://github.com/datascientist435-creator/Agentic-RAG-Document-QA.git
cd Agentic-RAG-Document-QA
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🔑 Getting a Free API Key

### Option 1: Groq API (Recommended — Fastest & Completely Free)

| Step | Action |
|:-----|:-------|
| 1 | Go to [console.groq.com](https://console.groq.com) |
| 2 | Sign up with Google (30 seconds, no credit card) |
| 3 | Click **"API Keys"** in the left sidebar |
| 4 | Click **"Create API Key"** → give it any name |
| 5 | Copy the key and paste it into the app's sidebar |

**Groq Free Tier Limits:**

| Feature | Limit |
|:--------|:------|
| API Keys | ♾️ Unlimited — create/delete as many as you want |
| Requests per minute | 30 requests/min |
| Requests per day | 14,400 requests/day |
| Cost | **$0 — completely free** |

### Option 2: OpenAI API
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up → You get **$5 free credits** for new accounts
3. Go to API Keys → Create a new key → Paste into the app

---

## 📦 Tech Stack

| Technology | Purpose |
|:-----------|:--------|
| **Python** | Core programming language |
| **Streamlit** | Web UI framework (frontend + backend in one file) |
| **LangChain** | LLM orchestration framework |
| **LCEL (LangChain Expression Language)** | Modern chain composition using pipe operators |
| **FAISS** | In-memory vector database by Meta AI |
| **HuggingFace Sentence Transformers** | Free, local embedding model (`all-MiniLM-L6-v2`) |
| **Groq API** | Ultra-fast free LLM inference (Llama 3.3 70B) |
| **OpenAI API** | Industry-standard LLM (GPT-3.5 Turbo) |
| **PyPDF** | PDF text extraction library |

---

## 🌐 Deployment (Streamlit Community Cloud)

This app is deployed for free on **Streamlit Community Cloud**:

1. Push your code to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub
3. Click **"New app"** → Select your repo → Set main file to `app.py`
4. Click **"Deploy!"** → Your app is live in ~3 minutes

---

## 📁 Project Structure

```
Agentic-RAG-Document-QA/
├── app.py                 # Main Streamlit application with full RAG pipeline
├── requirements.txt       # Python dependencies for local & cloud deployment
└── README.md              # This file — project documentation
```

---

## 👨‍💻 Author

**Nantha Kumaar Venkatachalam**  
AI Developer & MLOps Engineer  
🌐 [Portfolio](https://datascientist435-creator.github.io) | 💻 [GitHub](https://github.com/datascientist435-creator)
