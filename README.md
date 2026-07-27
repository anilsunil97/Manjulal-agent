# 📄 Petpooja Document Q&A

A RAG (Retrieval-Augmented Generation) chatbot that answers questions from your PDF and Excel documents, powered by **Groq**, **LangChain**, and **Streamlit**.

## Features

- Reads PDF and Excel (`.xlsx`, `.xls`) files from the `petpooja_docs/` folder
- Embeds documents using Google Gemini embeddings
- Answers questions using Groq LLM (`openai/gpt-oss-120b`)
- Shows relevant source document chunks for every answer
- CI/CD pipeline via GitHub Actions

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/anilsunil97/petpooja-document-qa.git
cd petpooja-document-qa
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API keys
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Add documents
Place your PDF or Excel files inside the `petpooja_docs/` folder.

### 5. Run the app
```bash
streamlit run app.py
```

Then open http://localhost:8501, click **Load & Embed Documents**, and start asking questions.

## Deployment

Deployable to [Streamlit Community Cloud](https://share.streamlit.io):
1. Connect this GitHub repo
2. Set main file to `app.py`
3. Add `GROQ_API_KEY` and `GOOGLE_API_KEY` in the Streamlit Cloud Secrets UI

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector Store | FAISS |
| Framework | LangChain |
| UI | Streamlit |
| CI/CD | GitHub Actions |
