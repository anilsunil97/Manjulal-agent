import os
import base64
import hashlib
import json
import pandas as pd
import streamlit as st
from docx import Document as DocxDocument
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

load_dotenv()

def get_secret(key: str) -> str:
    value = os.getenv(key)
    if not value:
        try:
            value = st.secrets[key]
        except (KeyError, FileNotFoundError):
            pass
    return value or ""

groq_api_key = get_secret("GROQ_API_KEY")

DOCS_DIR  = os.path.join(os.path.dirname(__file__), "petpooja_docs")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
HASH_FILE = os.path.join(INDEX_DIR, "docs_hash.json")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Manjulal Agent", page_icon="💃", layout="centered")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide sidebar */
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

/* Centred header */
.pp-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.2rem 0 0.6rem;
    gap: 0.6rem;
}
.pp-header img, .pp-header video {
    width: 96px; height: 96px;
    border-radius: 50%;
    object-fit: cover;
}
.pp-header h1 {
    margin: 0;
    font-size: 1.9rem;
    text-align: center;
}

/* New-chat button row */
.pp-newchat {
    display: flex;
    justify-content: center;
    margin: 0.4rem 0;
}
.pp-newchat button {
    background: #f0f2f6;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 0.35rem 1.2rem;
    font-size: 0.9rem;
    cursor: pointer;
}
.pp-newchat button:hover { background: #e2e5ea; }
</style>
""", unsafe_allow_html=True)


# ── Header (centred logo + title via HTML) ────────────────────────────────────
def load_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

gif_path = os.path.join(os.path.dirname(__file__), "static", "motu.gif")
mp4_path = os.path.join(os.path.dirname(__file__), "static",
                        "WhatsApp Video 2026-07-28 at 20.12.36.mp4")

if os.path.exists(gif_path):
    b64   = load_as_base64(gif_path)
    media = f'<img src="data:image/gif;base64,{b64}" alt="logo">'
elif os.path.exists(mp4_path):
    b64   = load_as_base64(mp4_path)
    media = (f'<video autoplay loop muted playsinline>'
             f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>')
else:
    media = '<img src="https://media.giphy.com/media/l3vR4yk0X20KimqJ2/giphy.gif" alt="logo">'

st.markdown(f"""
<div class="pp-header">
    {media}
    <h1>Manjulal Agent</h1>
</div>
""", unsafe_allow_html=True)


# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")


# ── Prompts ───────────────────────────────────────────────────────────────────
doc_prompt = ChatPromptTemplate.from_template(
    """You are a Petpooja support assistant. Answer precisely and concisely using ONLY the context below.

Rules:
- Give a direct, specific answer. No filler or preamble.
- Context may include table rows like "Header\\nService | Price1 | Price2..." — read carefully.
- For pricing questions always include: Service Name, New Price (w/o tax), New Price (with tax), Renewal Price (w/o tax), Renewal Price (with tax).
- Use bullet points or numbered steps only for processes or lists.
- If the answer is not in the context, respond with exactly: "We are updating more information."
- Do NOT use general knowledge or make up answers.

Context:
{context}

Question: {input}

Answer:"""
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Petpooja support assistant. Help only with Petpooja dashboard functionality and pricing.

Rules:
1. Greetings (hi, hello, good morning, etc.) → reply briefly and suggest asking about Petpooja dashboard features or pricing.
2. Any other off-topic question → respond with exactly:
   "I'm here to help with Petpooja dashboard functionality and pricing. Please ask me something related to that!"
3. Do NOT answer general knowledge, coding, weather, or unrelated questions."""),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# ── File loaders ──────────────────────────────────────────────────────────────
def load_pdf(filepath):
    source = os.path.basename(filepath)
    if not PDFPLUMBER_AVAILABLE:
        return PyPDFLoader(filepath).load()
    docs = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            header = None
            for table in (page.extract_tables() or []):
                for row_idx, row in enumerate(table):
                    cells = [str(c or "").strip() for c in row]
                    if not any(cells):
                        continue
                    row_text = " | ".join(cells)
                    if row_idx == 0:
                        header = row_text
                        continue
                    content = f"{header}\n{row_text}" if header else row_text
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": source, "page": i, "type": "table_row"}
                    ))
            plain = page.extract_text() or ""
            if plain.strip():
                docs.append(Document(
                    page_content=plain.strip(),
                    metadata={"source": source, "page": i, "type": "text"}
                ))
    return docs if docs else PyPDFLoader(filepath).load()

def load_excel(filepath):
    docs = []
    xl = pd.ExcelFile(filepath)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        docs.append(Document(
            page_content=f"Sheet: {sheet}\n\n{df.to_string(index=False)}",
            metadata={"source": os.path.basename(filepath), "sheet": sheet}
        ))
    return docs

def load_docx(filepath):
    doc = DocxDocument(filepath)
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return [Document(page_content="\n".join(parts),
                     metadata={"source": os.path.basename(filepath)})]

def load_all_docs(folder):
    all_docs = []
    for filename in os.listdir(folder):
        if filename.startswith("~$"):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".pdf", ".xlsx", ".xls", ".docx"):
            continue
        fp = os.path.join(folder, filename)
        try:
            if ext == ".pdf":
                all_docs.extend(load_pdf(fp))
            elif ext in (".xlsx", ".xls"):
                all_docs.extend(load_excel(fp))
            elif ext == ".docx":
                all_docs.extend(load_docx(fp))
        except Exception as e:
            st.warning(f"⚠️ Could not load `{filename}`: {e}")
    return all_docs


# ── Hash helpers ──────────────────────────────────────────────────────────────
def compute_docs_hash(folder):
    entries = sorted([
        (f, os.path.getmtime(os.path.join(folder, f)))
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in (".pdf", ".xlsx", ".xls", ".docx")
        and not f.startswith("~$")
    ])
    return hashlib.md5(json.dumps(entries).encode()).hexdigest()

def load_saved_hash():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            return json.load(f).get("hash", "")
    return ""

def save_hash(h):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(HASH_FILE, "w") as f:
        json.dump({"hash": h}, f)


# ── Embeddings — cached across all reruns ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Index — cached keyed on docs hash so it rebuilds only when docs change ───
@st.cache_resource(show_spinner="Setting up knowledge base…")
def get_vectorstore(docs_hash: str):          # docs_hash arg forces rebuild on change
    """Load from disk if current, otherwise build and save."""
    index_file = os.path.join(INDEX_DIR, "index.faiss")
    embeddings = get_embeddings()

    if os.path.exists(index_file) and load_saved_hash() == docs_hash:
        try:
            return FAISS.load_local(INDEX_DIR, embeddings,
                                    allow_dangerous_deserialization=True)
        except Exception:
            pass  # fall through to rebuild

    # Build
    all_docs = load_all_docs(DOCS_DIR)
    if not all_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    vectorstore = None
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    save_hash(docs_hash)
    return vectorstore


# ── Bootstrap: ensure dirs exist and load vectorstore ────────────────────────
if not os.path.isdir(DOCS_DIR):
    os.makedirs(DOCS_DIR)

_files = [f for f in os.listdir(DOCS_DIR)
          if os.path.splitext(f)[1].lower() in (".pdf", ".xlsx", ".xls", ".docx")
          and not f.startswith("~$")]

if _files:
    _hash = compute_docs_hash(DOCS_DIR)
    vectorstore = get_vectorstore(_hash)   # cached — runs once per hash value
else:
    vectorstore = None


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── New Chat — rendered as a plain Streamlit button (NO columns) ──────────────
# Centred via empty columns trick that doesn't interfere with chat_input
_, mid, _ = st.columns([2, 1, 2])
with mid:
    if st.button("🗑️ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Chat input — top-level call, nothing wrapping it ─────────────────────────
question = st.chat_input("Ask me anything about Petpooja…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                if vectorstore:
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.6}
                    )
                    rag_chain = (
                        {"context": retriever | format_docs, "input": RunnablePassthrough()}
                        | doc_prompt | llm | StrOutputParser()
                    )
                    answer = rag_chain.invoke(question)
                else:
                    chain  = chat_prompt | llm | StrOutputParser()
                    answer = chain.invoke({"input": question})

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"❌ Error: {e}")
