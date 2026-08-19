from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def retrieve(query: str = None, category: str = None, title: str = None, k: int = 3) -> str:
    if title:
        results = vectorstore.get(where={"title": title.upper()})
        if results["documents"]:
            return "\n\n".join(results["documents"])

    if category:
        results = vectorstore.get(where={"category": category.upper()})
        if results["documents"]:
            return "\n\n".join(results["documents"])

    if query:
        docs = vectorstore.similarity_search(query, k=k)
        return "\n\n".join(d.page_content for d in docs)

    return "No query, category, or title provided."


if __name__ == "__main__":
    print(retrieve(title="SOME REAL TITLE HERE"))