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

# Load environment variables
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

# ── Hide sidebar completely ───────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"]         { display: none !important; }
    [data-testid="collapsedControl"]  { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
def load_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

col_logo, col_title = st.columns([1, 5])
with col_logo:
    gif_path = os.path.join(os.path.dirname(__file__), "static", "motu.gif")
    mp4_path = os.path.join(os.path.dirname(__file__), "static", "WhatsApp Video 2026-07-28 at 20.12.36.mp4")
    if os.path.exists(gif_path):
        b64 = load_as_base64(gif_path)
        st.markdown(f'<img src="data:image/gif;base64,{b64}" width="80" style="border-radius:12px;">',
                    unsafe_allow_html=True)
    elif os.path.exists(mp4_path):
        b64 = load_as_base64(mp4_path)
        st.markdown(f'<video width="80" autoplay loop muted playsinline style="border-radius:12px;">'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<img src="https://media.giphy.com/media/l3vR4yk0X20KimqJ2/giphy.gif" width="80" style="border-radius:12px;">',
                    unsafe_allow_html=True)
with col_title:
    st.title("Manjulal Agent")

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")

# ── Prompts ───────────────────────────────────────────────────────────────────
doc_prompt = ChatPromptTemplate.from_template(
    """You are a Petpooja support assistant. Answer the question precisely and concisely using ONLY the context below.

Rules:
- Give a direct, specific answer. No filler or preamble.
- Context may include table rows in "Header\\nService | Price1 | Price2..." format — read them carefully.
- For pricing questions, always state: Service Name, New Price (without tax), New Price (with tax), Renewal Price (without tax), Renewal Price (with tax).
- Use bullet points or numbered steps only when the answer is a process or list.
- If the answer is not in the context, respond with exactly: "We are updating more information."
- Do NOT use general knowledge or make up answers.

Context:
{context}

Question: {input}

Answer:"""
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Petpooja support assistant. Your sole purpose is to help users with Petpooja dashboard functionality and pricing.

Rules:
1. For greetings (hello, hi, good morning, how are you, etc.) reply briefly and friendly, then suggest the user ask about Petpooja dashboard features or pricing. Example: "Hi there! Feel free to ask me about Petpooja dashboard functionality or pricing."
2. For ANY other topic that is not related to Petpooja dashboard functionality or pricing, respond with exactly:
   "I'm here to help with Petpooja dashboard functionality and pricing. Please ask me something related to that!"
3. Do NOT answer general knowledge, coding, weather, or any off-topic questions.
4. Keep all responses concise and on-topic."""),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ── File loaders ──────────────────────────────────────────────────────────────
def load_pdf(filepath):
    """Load PDF using pdfplumber.
    - Table rows are emitted as individual Documents (one row = one doc) so
      each service/price entry gets its own chunk and is retrieved precisely.
    - Non-table text is kept as a single page-level Document.
    - Falls back to PyPDFLoader only if pdfplumber is not available.
    """
    source = os.path.basename(filepath)
    if not PDFPLUMBER_AVAILABLE:
        return PyPDFLoader(filepath).load()

    docs = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            # Collect header row for this page (first row of first table)
            header = None
            tables = page.extract_tables()

            # Extract table rows as individual documents
            for table in tables:
                for row_idx, row in enumerate(table):
                    cells = [str(c or "").strip() for c in row]
                    row_text = " | ".join(cells)
                    if not any(cells):
                        continue
                    # Treat first row as header — combine with every data row
                    if row_idx == 0:
                        header = row_text
                        continue
                    # Prepend header so each chunk is self-contained
                    content = f"{header}\n{row_text}" if header else row_text
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": source, "page": i, "type": "table_row"}
                    ))

            # Also capture plain text (paragraphs outside tables)
            plain_text = page.extract_text() or ""
            if plain_text.strip():
                docs.append(Document(
                    page_content=plain_text.strip(),
                    metadata={"source": source, "page": i, "type": "text"}
                ))

    return docs if docs else PyPDFLoader(filepath).load()

def load_excel(filepath):
    docs = []
    xl = pd.ExcelFile(filepath)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        text = f"Sheet: {sheet}\n\n{df.to_string(index=False)}"
        docs.append(Document(page_content=text,
                             metadata={"source": os.path.basename(filepath), "sheet": sheet}))
    return docs

def load_docx(filepath):
    """Extract paragraphs and table rows from a docx file."""
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

def load_all_docs_from_folder(folder):
    all_docs = []
    supported = (".pdf", ".xlsx", ".xls", ".docx")
    for filename in os.listdir(folder):
        if filename.startswith("~$"):
            continue
        if not filename.lower().endswith(supported):
            continue
        filepath = os.path.join(folder, filename)
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == ".pdf":
                all_docs.extend(load_pdf(filepath))
            elif ext in (".xlsx", ".xls"):
                all_docs.extend(load_excel(filepath))
            elif ext == ".docx":
                all_docs.extend(load_docx(filepath))
        except Exception as e:
            st.warning(f"⚠️ Could not load `{filename}`: {e}")
    return all_docs


# ── Hash helpers ──────────────────────────────────────────────────────────────
def compute_docs_hash(folder):
    supported = (".pdf", ".xlsx", ".xls", ".docx")
    entries = sorted([
        (f, os.path.getmtime(os.path.join(folder, f)))
        for f in os.listdir(folder)
        if f.lower().endswith(supported) and not f.startswith("~$")
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


# ── Embeddings (cached — loads once per session) ──────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Build index silently (no progress bar shown to user) ─────────────────────
def build_index_silent(docs_dir):
    embeddings = get_embeddings()
    all_docs = load_all_docs_from_folder(docs_dir)
    if not all_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    split_docs = splitter.split_documents(all_docs)

    BATCH_SIZE = 100
    vectorstore = None
    for i in range(0, len(split_docs), BATCH_SIZE):
        batch = split_docs[i: i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    save_hash(compute_docs_hash(docs_dir))
    return vectorstore

def load_index_from_disk():
    return FAISS.load_local(INDEX_DIR, get_embeddings(),
                            allow_dangerous_deserialization=True)


# ── Background auto-indexing on every startup ─────────────────────────────────
if not os.path.isdir(DOCS_DIR):
    os.makedirs(DOCS_DIR)

files_present = [f for f in os.listdir(DOCS_DIR)
                 if f.lower().endswith((".pdf", ".xlsx", ".xls", ".docx"))
                 and not f.startswith("~$")]

index_exists = os.path.exists(os.path.join(INDEX_DIR, "index.faiss"))

if "vectors" not in st.session_state:
    if files_present:
        current_hash = compute_docs_hash(DOCS_DIR)
        saved_hash   = load_saved_hash()

        if index_exists and current_hash == saved_hash:
            # Index is fresh — load from disk instantly
            try:
                st.session_state.vectors = load_index_from_disk()
            except Exception:
                st.session_state.vectors = build_index_silent(DOCS_DIR)
        else:
            # Docs added/changed or no index yet — build silently
            st.session_state.vectors = build_index_silent(DOCS_DIR)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── New Chat button + Chat input ──────────────────────────────────────────────
col_new, col_input = st.columns([1, 5])

with col_new:
    if st.button("🗑️ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

with col_input:
    question = st.chat_input("Ask me anything about Petpooja…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                if "vectors" in st.session_state and st.session_state.vectors:
                    retriever = st.session_state.vectors.as_retriever(
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
