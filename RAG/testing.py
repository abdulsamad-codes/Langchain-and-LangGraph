from pathlib import Path

raw_text = Path("data.md").read_text(encoding="utf-8")

# --- old version: fixed-size chunking ---
def chunk_document_fixed(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c.strip()]

# --- new version: header-based chunking ---
import re
def chunk_document_headers(text: str) -> list[str]:
    sections = re.split(r"(?m)(?=^## )", text)
    return [s.strip() for s in sections if s.strip().startswith("## ")]

fixed_chunks = chunk_document_fixed(raw_text)
header_chunks = chunk_document_headers(raw_text)

print(f"Fixed-size chunks: {len(fixed_chunks)}")
print(f"Header-based chunks: {len(header_chunks)}")

# bonus: see what each header chunk actually starts with, to sanity-check the split
print("\n--- Header chunk previews ---")
for i, c in enumerate(header_chunks, start=1):
    first_line = c.split("\n", 1)[0]
    print(f"[{i}] {first_line}  ({len(c)} chars)")