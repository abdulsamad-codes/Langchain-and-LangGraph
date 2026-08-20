from pathlib import Path
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_document(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def chunk_document(text: str) -> list[dict]:
    h1_blocks = re.split(r"(?m)(?=^# )", text)
    chunks = []

    for block in h1_blocks:
        block = block.strip()
        if not block.startswith("# "):
            continue

        category = block.split("\n", 1)[0].lstrip("# ").strip()
        h2_sections = re.split(r"(?m)(?=^## )", block)

        for section in h2_sections:
            section = section.strip()
            if not section.startswith("## "):
                continue

            title = section.split("\n", 1)[0].lstrip("# ").strip()

            chunks.append({
                "category": category.upper(),
                "title": title.upper(),
                "content": section,
            })

    return chunks


def embed_and_store(chunks: list[dict], persist_directory: str = "./chroma_db") -> None:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [c["content"] for c in chunks]
    metadatas = [{"category": c["category"], "title": c["title"]} for c in chunks]

    Chroma.from_texts(
        texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory,
    )


if __name__ == "__main__":
    text = load_document("data.md")
    chunks = chunk_document(text)
    embed_and_store(chunks)
    print(f"Ingested {len(chunks)} chunks into ./chroma_db")