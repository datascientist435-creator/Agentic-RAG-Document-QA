# Agentic RAG Document Q&A 🤖
An interactive web application built with Python, **Streamlit** and **LangChain**. This app allows users to upload a massive PDF document and converse directly with it via a Generative AI pipeline. It uses advanced Retrieval-Augmented Generation (RAG) architecture running exclusively on a local FAISS vector database alongside free open-source embeddings from **HuggingFace**.

## 🌟 Key Engineering Features
- **Semantic Chunk Vectorization**: Analyzes and maps large texts down to geometric embedding vectors locally inside your machine.
- **In-Memory Vector Server**: Actively parses vectors via Meta's blazing-fast `FAISS` library using exact similarity dot-product calculations.
- **Pluggable LLMs**: Toggle between blazingly fast free inference via the **Groq API (Llama 3)** or the industry-standard **OpenAI API**.
- **Conversational Memory Processing**: The Generative AI agent remembers previous questions mathematically and concatenates recent context to answer complex follow-ups.

## 🛠️ The Technical Architecture Pipeline
1. **Document Ingestion**: Parses raw PDF bytes into strings utilizing `PyPDFLoader`.
2. **Text Chunking**: Breaks large, complex documents into relevant, highly manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embeddings Space & Storage**: Calculates embeddings utilizing the optimized `all-MiniLM-L6-v2` `sentence-transformers` model from HuggingFace to embed semantic knowledge into a FAISS index object.
4. **Retrieval**: Upon receiving an end-user query, similarity metric searches execute against the database returning only the top `K=3` most relevant text chunks (nodes).
5. **Generation**: The context strings and chat history arrays are parsed directly into the Conversational LLM chain loop to synthesize an accurate, directly-cited response for the user.

## 🚀 How to Run Locally

### 1. Clone the Source Repository
```bash
git clone https://github.com/your-username/Agentic-RAG.git
cd Agentic-RAG
```

### 2. Install Pipeline Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Inference Application
```bash
streamlit run app.py
```

## 🌐 Live Web Demo Note
The application can be deployed instantly for completely free using **Streamlit Community Cloud** with near-zero backend configuration, perfect for portfolio demonstrations.
