from pypdf import PdfReader
pdf_path = "RAG_Practice_20_Page_Company_Handbook.pdf"
reader = PdfReader(pdf_path)
print(len(reader.pages))
full_text = ""

for page in reader.pages:
    full_text += page.extract_text() + "\n"

print(full_text[:1000])
with open("handbook.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Saved to handbook.txt")
with open("handbook.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Saved to handbook.txt")
# chunk.py

with open("handbook.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunk_size = 800      # characters per chunk
overlap = 150         # characters of overlap between chunks

chunks = []

start = 0
while start < len(text):
    end = start + chunk_size
    chunk = text[start:end]
    chunks.append(chunk)
    start = end - overlap   # step forward, but back up by `overlap`

print(f"Total chunks: {len(chunks)}")
print("\n--- First chunk ---\n")
print(chunks[0])
print("\n--- Second chunk ---\n")
print(chunks[1])
with open("chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"\n===== CHUNK {i} =====\n")
        f.write(chunk)
        f.write("\n")

print("Saved to chunks.txt")