# store.py
# ------------------------------------------------------------
# Purpose: Load chunks from chunks.json, embed them, store
#          them in a persistent ChromaDB collection, then
#          run a few test queries to prove retrieval works.
# ------------------------------------------------------------

import json
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# STEP 1 — Load the chunks produced by chunk_with_meta.py
# ============================================================
# chunks.json is a list of dicts, each looking like:
#   {"id": 0, "text": "...", "metadata": {...}}
# We just read it back into a Python list.

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"✅ Loaded {len(chunks)} chunks from chunks.json")


# ============================================================
# STEP 2 — Load the embedding model
# ============================================================
# all-MiniLM-L6-v2:
#   - Small (~90 MB), fast on CPU
#   - Produces 384-dimensional vectors
#   - Good enough for retrieval on English text
# First run downloads it into ~/.cache/huggingface — later runs
# load from disk instantly.

print("⏳ Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded")


# ============================================================
# STEP 3 — Connect to a persistent ChromaDB
# ============================================================
# PersistentClient(path="./chroma_db") writes the DB to disk
# in a folder called chroma_db/. Next time you run the script,
# data is still there. (Client() in-memory would be wiped.)

client = chromadb.PersistentClient(path="./chroma_db")


# ============================================================
# STEP 4 — Wipe any previous "handbook" collection
# ============================================================
# If you re-run the script and the collection already exists,
# Chroma will complain or you'll get duplicate IDs. Easiest
# fix during development: delete and recreate.
# We wrap it in try/except because delete fails if the
# collection doesn't exist yet (first ever run).

try:
    client.delete_collection("handbook")
    print("🗑️  Deleted existing 'handbook' collection")
except Exception:
    print("ℹ️  No existing 'handbook' collection — creating fresh")

collection = client.create_collection(name="handbook")
print("✅ Created 'handbook' collection")


# ============================================================
# STEP 5 — Prepare data in the shapes Chroma expects
# ============================================================
# Chroma wants three parallel lists of the same length:
#   documents : list of strings (the chunk text)
#   metadatas : list of dicts (must be primitive values only:
#               str, int, float, bool — no nested dicts/lists)
#   ids       : list of unique strings
#
# We build them with list comprehensions.

texts = [c["text"] for c in chunks]
metadatas = [c["metadata"] for c in chunks]
ids = [f"chunk_{c['id']}" for c in chunks]

print(f"📦 Prepared {len(texts)} documents")


# ============================================================
# STEP 6 — Embed every chunk (the expensive step)
# ============================================================
# model.encode(list_of_strings) returns a NumPy array of shape
# (N, 384). Chroma wants plain Python lists, so we call .tolist().
# show_progress_bar=True prints a nice progress bar.

print("⏳ Embedding chunks... (this may take a few seconds)")
embeddings = model.encode(texts, show_progress_bar=True).tolist()
print(f"✅ Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")


# ============================================================
# STEP 7 — Add everything to Chroma in one call
# ============================================================
# collection.add stores text + metadata + vector + id together.
# Later, query() will only need the question text — Chroma will
# embed it with the same model... actually no! Chroma does NOT
# embed. It uses whatever vector you pass to query(). We'll
# embed the question ourselves in Step 8.

collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids,
)
print("✅ Added all chunks to ChromaDB")


# ============================================================
# STEP 8 — Test retrieval
# ============================================================
# For each test question:
#   1. Embed the question with the SAME model
#   2. Ask Chroma for the top-3 closest chunks (cosine similarity)
#   3. Print the chunk text + which page/chunk it came from

test_questions = [
    "What are the core collaboration hours?",
    "How do I report a lost company laptop?",
    "Who handles benefit questions?",
]

for q in test_questions:
    print(f"\n{'='*70}")
    print(f"Q: {q}")
    print('='*70)

    q_embedding = model.encode([q]).tolist()

    results = collection.query(
        query_embeddings=q_embedding,
        n_results=3,
    )

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"\n--- Rank {i+1} | page {meta['page']} | "
              f"chunk_id {meta['chunk_id']} | distance {dist:.4f} ---")
        # Show first 350 chars so it fits on screen
        print(doc[:350].replace("\n", " "))


print("\n🎉 Done. ChromaDB saved to ./chroma_db/")