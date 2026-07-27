import os
import io
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
# Works locally via .env and in production via Streamlit Cloud secrets
load_dotenv()

def get_secret(key: str) -> str:
    """Read from .env locally, fall back to st.secrets on Streamlit Cloud."""
    value = os.getenv(key)
    if not value:
        try:
            value = st.secrets[key]
        except (KeyError, FileNotFoundError):
            pass
    return value or ""

groq_api_key = get_secret("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = get_secret("GOOGLE_API_KEY")

DOCS_DIR = os.path.join(os.path.dirname(__file__), "petpooja_docs")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Petpooja Document Q&A", page_icon="📄", layout="centered")
st.title("📄 Petpooja — Document Q&A")
st.caption(f"Reads all PDF and Excel files from the `petpooja_docs` folder automatically.")

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
)

# ── Prompt template ───────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the provided context.
Give the most accurate and concise response possible.

Context:
{context}

Question: {input}

Answer:"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ── File loaders ──────────────────────────────────────────────────────────────
def load_pdf(filepath: str) -> list[Document]:
    loader = PyPDFLoader(filepath)
    return loader.load()


def load_excel(filepath: str) -> list[Document]:
    docs = []
    xl = pd.ExcelFile(filepath)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        text = f"Sheet: {sheet}\n\n{df.to_string(index=False)}"
        docs.append(
            Document(
                page_content=text,
                metadata={"source": os.path.basename(filepath), "sheet": sheet},
            )
        )
    return docs


def load_all_docs_from_folder(folder: str) -> list[Document]:
    all_docs = []
    supported = (".pdf", ".xlsx", ".xls")
    files = [f for f in os.listdir(folder) if f.lower().endswith(supported)]

    if not files:
        return []

    for filename in files:
        filepath = os.path.join(folder, filename)
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == ".pdf":
                all_docs.extend(load_pdf(filepath))
            elif ext in (".xlsx", ".xls"):
                all_docs.extend(load_excel(filepath))
        except Exception as e:
            st.warning(f"⚠️ Could not load `{filename}`: {e}")

    return all_docs


# ── Show files present in folder ──────────────────────────────────────────────
if not os.path.isdir(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    st.warning("📁 `petpooja_docs` folder was just created. Add your PDF/Excel files and click **Load & Embed**.")
else:
    files_present = [
        f for f in os.listdir(DOCS_DIR)
        if f.lower().endswith((".pdf", ".xlsx", ".xls"))
    ]
    if files_present:
        with st.expander(f"📁 Files in `petpooja_docs` ({len(files_present)} found)", expanded=False):
            for f in files_present:
                st.write(f"• {f}")
    else:
        st.warning("📁 No PDF or Excel files found in `petpooja_docs`. Please add files and click **Load & Embed**.")

# ── Buttons ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])
with col1:
    embed_btn = st.button("📥 Load & Embed Documents", use_container_width=True)
with col2:
    if st.button("🗑️ Clear & Reset", use_container_width=True):
        for key in ["vectors", "doc_count", "file_count"]:
            st.session_state.pop(key, None)
        st.success("Reset complete.")

# ── Embedding logic ───────────────────────────────────────────────────────────
if embed_btn:
    with st.spinner("Reading files from `petpooja_docs`…"):
        all_docs = load_all_docs_from_folder(DOCS_DIR)

    if not all_docs:
        st.error("No content found. Please add PDF or Excel files to the `petpooja_docs` folder.")
    else:
        with st.spinner("Embedding documents…"):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(all_docs)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            st.session_state.vectors = FAISS.from_documents(split_docs, embeddings)
            st.session_state.doc_count = len(split_docs)
            st.session_state.file_count = len(files_present)

        st.success(
            f"✅ Embedded {st.session_state['doc_count']} chunks "
            f"from {st.session_state['file_count']} file(s). Ask your question below!"
        )

st.divider()

# ── Question input ────────────────────────────────────────────────────────────
question = st.text_input("💬 Enter your question about the documents")

if question:
    if "vectors" not in st.session_state:
        st.warning("Please click **Load & Embed Documents** first.")
    else:
        retriever = st.session_state.vectors.as_retriever()

        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Searching documents…"):
            start = time.process_time()
            answer = rag_chain.invoke(question)
            elapsed = time.process_time() - start

        st.subheader("Answer")
        st.write(answer)
        st.caption(f"Response time: {elapsed:.2f}s")


