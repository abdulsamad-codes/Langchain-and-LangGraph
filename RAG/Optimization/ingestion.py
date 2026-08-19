from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re



def load_document(file_path: str) -> str:
    """Reads the markdown file from disk and returns its raw text."""
    return Path(file_path).read_text(encoding="utf-8")


def chunk_document(text: str) -> list[dict]:
    """Splits raw text into chunks by level-2 (##) heading sections.
    Each chunk gets tagged with its parent level-1 (#) heading as 'category'
    and its own level-2 (##) heading as 'title'.

    Returns a list of dicts: {"category": ..., "title": ..., "content": ...}
    """
    h1_blocks = re.split(r"(?m)(?=^# )", text)

    chunks = []

    for block in h1_blocks:
        block = block.strip()
        if not block.startswith("# "):
            continue  # skips any stray text before the first '#' heading

        category = block.split("\n", 1)[0].lstrip("# ").strip()

        h2_sections = re.split(r"(?m)(?=^## )", block)

        for section in h2_sections:
            section = section.strip()
            if not section.startswith("## "):
                continue  # this is the H1 intro paragraph — skipped, same as before

            title = section.split("\n", 1)[0].lstrip("# ").strip()

            chunks.append({
                "category": category.upper(),
                "title": title.upper(),
                "content": section,
            })

    return chunks



def embed_and_store(chunks: list[dict], persist_directory: str = "./chroma_db") -> None:
    """Embeds all chunks and saves them to a Chroma vectorstore on disk."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    texts = [c["content"] for c in chunks]
    metadatas = [{"category": c["category"], "title": c["title"]} for c in chunks]

    Chroma.from_texts(
        texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory,
    )



def retrieval(query: str = None, category: str= None, title: str = None, k :int = 1):
    pass





if __name__ == "__main__":
    text = load_document("data.md")
    chunks = chunk_document(text)
    embed_and_store(chunks)
    print(f"Ingested {len(chunks)} chunks into ./chroma_db")
    retrieval('What is the fee required for driving license renewal')