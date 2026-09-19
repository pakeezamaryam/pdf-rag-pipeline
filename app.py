# app.py
# Week 5 — Streamlit RAG app: upload a PDF, ask questions, get grounded answers.

import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from groq import Groq

# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📄 PDF Q&A Assistant")
st.write("Upload a PDF and ask questions about it.")


# ============================================================
# Load models ONCE at startup (cached across reruns)
# ============================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_chroma():
    # In-memory client — fresh each time the app restarts
    return chromadb.Client()


@st.cache_resource
def load_groq():
    """Load Groq client using either Streamlit Secrets (deployed) or .env (local)."""
    api_key = None
    
    # Try Streamlit Secrets first (for deployed apps)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    
    # Fall back to .env (for local development)
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to .env (local) or Streamlit Secrets (deployed)."
        )
    
    return Groq(api_key=api_key)


embedder = load_embedder()
chroma_client = load_chroma()
groq_client = load_groq()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = 5


# ============================================================
# Ingestion helpers
# ============================================================
def extract_text(pdf_file):
    """Return list of (page_number, page_text)."""
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def chunk_page(text, chunk_words=150, overlap=30):
    """Split a page's text into word-based chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def ingest_pdf(pdf_file):
    """PDF → chunks → embeddings → Chroma collection."""
    # 1. Extract
    pages = extract_text(pdf_file)

    # 2. Chunk (with page metadata)
    docs, metadatas, ids = [], [], []
    cid = 0
    for page_num, text in pages:
        if not text.strip():
            continue
        for chunk in chunk_page(text):
            docs.append(chunk)
            metadatas.append({
                "source": pdf_file.name,
                "page": page_num,
                "chunk_id": cid,
            })
            ids.append(f"chunk_{cid}")
            cid += 1

    if not docs:
        raise ValueError("No text could be extracted from this PDF. "
                         "It may be scanned (image-only) or empty.")

    # 3. Embed
    embeddings = embedder.encode(docs, show_progress_bar=False).tolist()

    # 4. Store in a fresh collection
    try:
        chroma_client.delete_collection("handbook")
    except Exception:
        pass
    collection = chroma_client.create_collection("handbook")
    collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return collection, len(docs)


# ============================================================
# RAG answer function
# ============================================================
def answer_question(question, collection):
    """Retrieve top-K chunks and ask the LLM to answer using only them."""
    # 1. Embed the question
    q_vec = embedder.encode([question]).tolist()

    # 2. Retrieve
    results = collection.query(query_embeddings=q_vec, n_results=TOP_K)
    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    # 3. Build context
    context_parts = []
    for i, (doc, meta) in enumerate(zip(chunks, metas), start=1):
        context_parts.append(f"[Source {i} — page {meta['page']}]\n{doc}")
    context = "\n\n".join(context_parts)

    # 4. Prompt
    system_prompt = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the provided context. If the answer is not in the context, respond "
    "exactly with: 'I don't have enough information in the document to answer that.' "
    "Do not use any outside knowledge. "
    "Write answers of 2 to 4 sentences. Include enough background from the context "
    "that a reader unfamiliar with the document can fully understand the answer. "
    "If the answer mentions specific names, modules, teams, or phases, briefly "
    "explain what they are using information from the context."
)
    user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

    # 5. Ask the LLM
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    answer = response.choices[0].message.content
    sources = [(m["page"], m["chunk_id"]) for m in metas]
    return answer, sources


# ============================================================
# PDF uploader + ingestion
# ============================================================
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Only ingest when the file is NEW
    if st.session_state.get("ingested_file") != uploaded_file.name:
        try:
            with st.spinner("Processing document... (this takes ~10 seconds)"):
                collection, n_chunks = ingest_pdf(uploaded_file)
                st.session_state["collection"] = collection
                st.session_state["ingested_file"] = uploaded_file.name
                st.session_state["n_chunks"] = n_chunks
                st.session_state.messages = []   # reset chat
        except Exception as e:
            st.error(f"❌ Failed to process PDF: {e}")
            st.stop()

    st.success(
        f"✅ Ready: **{uploaded_file.name}** "
        f"({st.session_state['n_chunks']} chunks indexed)"
    )
else:
    st.info("👆 Upload a PDF to get started")

st.divider()


# ============================================================
# Chat
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
question = st.chat_input("Ask a question about your document...")

if question:
    # 1. Show the user's message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # 2. Generate the answer
    collection = st.session_state.get("collection")
    if collection is None:
        with st.chat_message("assistant"):
            st.warning("Please upload a PDF first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(question, collection)
            st.write(answer)

            with st.expander(f"📄 Sources ({len(sources)})"):
                for i, (page, cid) in enumerate(sources, start=1):
                    st.write(f"[{i}] page {page}, chunk {cid}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )