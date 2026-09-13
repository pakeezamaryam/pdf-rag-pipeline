# PDF RAG Pipeline

Ask questions about any PDF and get grounded answers with page citations.
The system refuses to answer when information isn't in the document.

## What it does

1. Extracts text from a PDF (`pypdf`)
2. Splits it into overlapping chunks (150 words, 30-word overlap)
3. Embeds each chunk with `sentence-transformers` (all-MiniLM-L6-v2)
4. Stores embeddings in a persistent ChromaDB
5. On a question, retrieves the top-3 nearest chunks
6. Sends chunks + question to a Groq-hosted LLM
7. Returns a grounded answer with source pages

## Why RAG

LLMs hallucinate. By retrieving only the relevant chunks from *your* document
and instructing the LLM to answer using **only** that context, we get answers
that are traceable back to a specific page — and honest refusals when the
answer isn't there.

## Files

| File | Purpose |
|---|---|
| `extract.py` | PDF → text |
| `chunk_with_meta.py` | text → 150-word chunks with overlap + metadata |
| `store.py` | chunks → embeddings → ChromaDB |
| `ask.py` | interactive RAG loop (question → answer) |
| `test_groq.py` | sanity check for the Groq API connection |

## Setup

```bash
pip install pypdf chromadb sentence-transformers groq python-dotenv
```

Create a `.env` file at the root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq key at https://console.groq.com

## Run

```bash
python extract.py
python chunk_with_meta.py
python store.py
python ask.py
```

## Example session

```
You: Who handles benefit questions?

Human Resources.

Sources retrieved:
  [1] page 7  — Employee Benefits section
  [2] page 19 — Quick Reference FAQ
```

```
You: Who is the CEO?

I don't have enough information in the document to answer that.
```

## Stack

Python · pypdf · sentence-transformers · ChromaDB · Groq (openai/gpt-oss-120b)