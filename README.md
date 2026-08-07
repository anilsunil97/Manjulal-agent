# 🤖 Manjulal Agent — Petpooja Support Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about **Petpooja dashboard functionality and pricing**, powered by **Groq**, **LangChain**, and **Streamlit**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Features

- Answers questions **only from your Petpooja documents** — no hallucinations
- Supports **PDF, DOCX, and Excel** files (including table extraction from docx/pdf)
- **Persistent FAISS index** — chunks are built once and reloaded on restart, no re-embedding unless docs change
- **MMR retrieval** for precise, diverse answers
- Drag-and-drop **file upload** in the sidebar to add new documents
- Fallback message: *"We are updating more information."* when answer is not in docs
- CI/CD pipeline via GitHub Actions

## Setup (Local)

### 1. Clone the repo
```bash
git clone https://github.com/anilsunil97/Manjulal-agent.git
cd Manjulal-agent
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 3. Add your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Add documents
Place your PDF, DOCX, or Excel files inside the `petpooja_docs/` folder.

### 5. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501, click **Build Index**, and start asking questions.

---

## Deployment on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app** → select this repo (`Manjulal-agent`)
3. Set **Main file path** to `app.py`
4. Under **Advanced settings**, set Python version to `3.11`
5. Click **Secrets** and add:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
6. Click **Deploy** — Streamlit Cloud will auto-deploy on every push to `master`

> **Note:** The FAISS index is rebuilt on first run in Streamlit Cloud since `faiss_index/` is in `.gitignore`. Click **Build Index** after deployment.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`, local, no API key needed) |
| Vector Store | FAISS (persistent to disk) |
| Retrieval | MMR (Maximal Marginal Relevance) |
| Framework | LangChain |
| UI | Streamlit |
| CI/CD | GitHub Actions |
