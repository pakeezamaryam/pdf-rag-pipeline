from pypdf import PdfReader
import json

pdf_path = "RAG_Practice_20_Page_Company_Handbook.pdf"
reader = PdfReader(pdf_path)

CHUNK_WORDS = 150   # ~300 words per chunk
OVERLAP_WORDS = 30    # 60-word overlap

chunks = []
chunk_id = 0

for page_num, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ""
    words = page_text.split()          # split on whitespace
    start = 0
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "source": pdf_path,
                "page": page_num,
                "chunk_id": chunk_id,
                "word_start": start,
                "word_end": end,
            },
        })
        chunk_id += 1
        if end == len(words):
            break
        start = end - OVERLAP_WORDS   # slide with overlap

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"Total chunks: {len(chunks)}")
print("First chunk preview:", chunks[0]["text"][:200])
print("Last chunk preview:", chunks[-1]["text"][-200:])