from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from typing import Optional

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

all_data = vectorstore.get()
all_metadatas = all_data["metadatas"]

seen = set()
catalog = []
for m in all_metadatas:
    key = (m["category"], m["title"])
    if key not in seen:
        seen.add(key)
        catalog.append({"category": m["category"], "title": m["title"]})


from langchain.tools import tool
from typing import Optional

@tool
def retrieve(query: str, category: Optional[str] = None, title: Optional[str] = None, k: int = 1) -> str:
    """Retrieve document content from the Dastak KP Citizens app documentation.
    If the query clearly matches one of the known titles or categories listed
    in the system prompt, pass it as the 'title' or 'category' argument for an
    exact lookup. Otherwise, just pass the query and leave title/category empty
    for a semantic search."""

    if title:
        results = vectorstore.get(where={"title": title.upper()})
        if results["documents"]:
            return "\n\n".join(results["documents"])

    if category:
        results = vectorstore.get(where={"category": category.upper()})
        if results["documents"]:
            return "\n\n".join(results["documents"])

    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)