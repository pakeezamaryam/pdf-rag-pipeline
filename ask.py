# ask.py
# Week 4 — Full RAG loop: question in, grounded answer out.

import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# SETUP — runs once when the script starts
# ------------------------------------------------------------
load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = 3

print("⏳ Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("⏳ Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("handbook")

print("⏳ Connecting to Groq...")
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

print(f"✅ Ready. Model: {MODEL_NAME} | Chunks: {collection.count()}\n")


# ------------------------------------------------------------
# RAG FUNCTION — one question → one answer
# ------------------------------------------------------------
def ask(question: str):
    # 1. Embed the question
    q_vec = embedder.encode([question]).tolist()

    # 2. Retrieve top-K chunks
    results = collection.query(query_embeddings=q_vec, n_results=TOP_K)
    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    # 3. Build context block with page labels
    context_parts = []
    for i, (doc, meta) in enumerate(zip(chunks, metas), start=1):
        context_parts.append(f"[Source {i} — page {meta['page']}]\n{doc}")
    context = "\n\n".join(context_parts)

    # 4. Build the prompt
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the provided context. If the answer is not in the context, respond "
        "exactly with: 'I don't have enough information in the document to answer that.' "
        "Do not use any outside knowledge. Be concise."
    )

    user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

    # 5. Send to the LLM
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    # 6. Print answer + sources
    print(f"\n{'='*70}")
    print(f"Q: {question}")
    print('='*70)
    print(f"\n{answer}\n")

    print("Sources retrieved:")
    for i, (doc, meta) in enumerate(zip(chunks, metas), start=1):
        preview = doc[:90].replace("\n", " ")
        print(f"  [{i}] page {meta['page']} — {preview}...")


# ------------------------------------------------------------
# INTERACTIVE LOOP
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Ask questions about the handbook. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("👋 Bye.")
            break
        try:
            ask(q)
        except Exception as e:
            print(f"❌ Error: {e}")