from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool

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


@tool
def retrieve(query: str) -> str:
    """Retrieve document content relevant to the user's query.
    Tries matching by category first, then by title, then falls back
    to semantic vector search if neither matches."""

    query_upper = query.upper()

    matched_categories = [
        c["category"] for c in catalog
        if c["category"] in query_upper
        ]

    if matched_categories:
        results = vectorstore.get(
            where={"category": {"$in": list(set(matched_categories))}}
        )
        return "\n\n".join(results["documents"])

    # --- 2nd weapon: title match ---
    matched_titles = [
    c["title"] for c in catalog
    if c["title"] in query_upper
]

    if matched_titles:
        results = vectorstore.get(
            where={"title": {"$in": list(set(matched_titles))}}
        )
        return "\n\n".join(results["documents"])
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(d.page_content for d in docs)



