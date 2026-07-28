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
os.environ["GOOGLE_API_KEY"] = get_secret("GOOGLE_API_KEY")

DOCS_DIR = os.path.join(os.path.dirname(__file__), "petpooja_docs")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Petpooja Assistant", page_icon="💃", layout="centered")

# ── Header with dancing GIF logo ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown(
        """
        <img src="https://media.giphy.com/media/l0MYGb1LuZ3n7dRnO/giphy.gif"
             width="80" style="border-radius: 12px;" alt="dancing logo">
        """,
        unsafe_allow_html=True,
    )
with col_title:
    st.title("Petpooja Assistant")

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
)

# ── Prompts ───────────────────────────────────────────────────────────────────
doc_prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using the provided document context.
If the context does not contain enough information, answer from your general knowledge and mention that.

Context:
{context}

Question: {input}

Answer:"""
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, friendly assistant. Answer clearly and concisely."),
    ("human", "{input}"),
])


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
        docs.append(Document(
            page_content=text,
            metadata={"source": os.path.basename(filepath), "sheet": sheet},
        ))
    return docs


def load_all_docs_from_folder(folder: str) -> list[Document]:
    all_docs = []
    supported = (".pdf", ".xlsx", ".xls")
    files = [f for f in os.listdir(folder) if f.lower().endswith(supported)]
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


# ── Resolve files in folder ───────────────────────────────────────────────────
if not os.path.isdir(DOCS_DIR):
    os.makedirs(DOCS_DIR)

files_present = [
    f for f in os.listdir(DOCS_DIR)
    if f.lower().endswith((".pdf", ".xlsx", ".xls"))
] if os.path.isdir(DOCS_DIR) else []

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document Mode")
    embed_btn = st.button("📥 Load & Embed Documents", use_container_width=True)

    if "vectors" in st.session_state:
        st.success(f"✅ {st.session_state.get('doc_count', 0)} chunks loaded")
        st.caption("Questions will be answered from documents + general knowledge.")
    else:
        st.info("No documents embedded yet.\nChat works in general mode.")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset Documents", use_container_width=True):
        for key in ["vectors", "doc_count"]:
            st.session_state.pop(key, None)
        st.rerun()

# ── Embedding logic ───────────────────────────────────────────────────────────
if embed_btn:
    with st.spinner("Reading and embedding documents…"):
        all_docs = load_all_docs_from_folder(DOCS_DIR)
        if not all_docs:
            st.sidebar.error("No files found in `petpooja_docs`.")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(all_docs)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            st.session_state.vectors = FAISS.from_documents(split_docs, embeddings)
            st.session_state.doc_count = len(split_docs)
            st.rerun()

# ── Display chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                if "vectors" in st.session_state:
                    # Document-grounded RAG answer
                    retriever = st.session_state.vectors.as_retriever()
                    rag_chain = (
                        {"context": retriever | format_docs, "input": RunnablePassthrough()}
                        | doc_prompt
                        | llm
                        | StrOutputParser()
                    )
                    answer = rag_chain.invoke(question)
                else:
                    # Plain conversational answer
                    chain = chat_prompt | llm | StrOutputParser()
                    answer = chain.invoke({"input": question})

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"❌ Error: {e}")
