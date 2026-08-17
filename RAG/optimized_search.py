from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, START,END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re

class RAGState(TypedDict):
    file_path: str
    raw_text: str        
    chunks: list[str]    
    metadatas: list[dict] 
    vectorstore: Chroma   
    query: str            
    results: list[str]



CATEGORY_KEYWORDS = {
    "Arms License": ["arms", "weapon", "firearm", "bore", "license", "gun", "retainer", "transfer", "renewal"],
    "Driving License": ["driving", "driver", "learner", "permit", "endorsement", "psv", "idp", "renewal"],
    "MVRS (Motor Vehicle Registration System)": ["vehicle", "registration", "token tax", "smart card", "ownership"],
    "Asan Karobar (Business Licensing)": ["business", "firm", "plot", "tourism", "hotel", "restaurant", "halal", "godown", "mill"],
    "FoodGrain Licensing": ["foodgrain", "grain", "wholesaler", "retailer", "shop licence", "mill licence", "renewal"],
}


def detect_categories(query: str) -> list[str]:
    """Returns every category whose keywords appear in the query."""
    query_lower = query.lower()
    matched = [
        category
        for category, keywords in CATEGORY_KEYWORDS.items()
        if any(keyword in query_lower for keyword in keywords)
    ]
    return matched



def load_document(state: RAGState) -> dict:
    """Node: reads the markdown file from disk."""
    text = Path(state["file_path"]).read_text(encoding="utf-8")
    return {"raw_text": text}


# def chunk_document(state: RAGState, chunk_size: int = 1000, overlap: int = 100) -> dict:
#     """Node: splits the raw text into fixed-size overlapping chunks."""
#     text = state["raw_text"]
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         if end >= len(text):
#             break
#         start = end - overlap
#     return {"chunks": [c for c in chunks if c.strip()]}


# def chunk_document(state: RAGState) -> dict:
#     """Node: splits raw text into chunks by level-2 (##) heading sections.

#     Each chunk = one '## heading' plus everything under it, up to
#     (but not including) the next '## heading' or end of file.
#     Content before the first '##' (e.g. H1 intro paragraphs) is
#     intentionally skipped for now — that's a separate decision for later.
#     """
#     text = state["raw_text"]

#     sections = re.split(r"(?m)(?=^## )", text)

#     chunks = [s.strip() for s in sections if s.strip().startswith("## ")]

#     return {"chunks": chunks}

def chunk_document(state: RAGState) -> dict:
    """Node: splits raw text into chunks by level-2 (##) heading sections,
    tagging each chunk with the level-1 (#) heading it falls under (its 'category').

    Each chunk = one '## heading' plus everything under it, up to
    (but not including) the next '## heading' or end of file.
    Content before the first '##' within each '#' block (e.g. H1 intro
    paragraphs) is intentionally skipped for now — separate decision for later.
    """
    text = state["raw_text"]

    h1_blocks = re.split(r"(?m)(?=^# )", text)

    chunks = []
    metadatas = []

    for block in h1_blocks:
        block = block.strip()
        if not block.startswith("# "):
            continue  

        h1_title = block.split("\n", 1)[0].lstrip("# ").strip()

        # Step 2: within this H1 block, split into H2 ("##") sections, same as before
        h2_sections = re.split(r"(?m)(?=^## )", block)

        for section in h2_sections:
            section = section.strip()
            if not section.startswith("## "):
                continue  # this piece is the H1 heading + its intro paragraph — skip
            chunks.append(section)
            metadatas.append({"category": h1_title})

    return {"chunks": chunks, "metadatas": metadatas}

def embed_and_store(state: RAGState) -> dict:
    """Node: embeds all chunks and stores them in a Chroma vectorstore."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(state["chunks"], embedding=embeddings,metadatas=state["metadatas"])
    return {"vectorstore": vectorstore}

def get_query(state: RAGState) -> dict:
    """Node: prompts the user for a query (or 'exit' to quit)."""
    query = input("\nEnter a query (or 'exit' to quit): ")
    return {"query": query}

def check_exit(state: RAGState) -> str:
    """Router: decides whether to end the loop or continue to the search step."""
    if state["query"].strip().lower() in {"exit", "quit"}:
        return "end"
    return "continue"

def search_chroma(state: RAGState) -> dict:
    """ Node for searching our query (vector) in the chroma db"""
    categories = detect_categories(state["query"])

    if categories:
        search_filter = {"category": {"$in": categories}}
        results = state["vectorstore"].similarity_search(
            state["query"], k=3, filter=search_filter
        )
    else:
        results = state["vectorstore"].similarity_search(state["query"], k=3)

    return {"results": [r.page_content for r in results]}

def show_results(state: RAGState) -> dict:
    """Node: displays the top-k matching chunks to the user."""
    print("\n--- Top matches ---")
    for i, chunk in enumerate(state["results"], start=1):
        print(f"\n[{i}] {chunk}")
    print("\n")
    return {}


graph = StateGraph(RAGState)

graph.add_node("load_document", load_document)
graph.add_node("chunk_document", chunk_document)
graph.add_node("embed_and_store", embed_and_store)
graph.add_node("get_query", get_query)
graph.add_node("search_chroma", search_chroma)
graph.add_node("show_results", show_results)

graph.set_entry_point("load_document")
graph.add_edge("load_document", "chunk_document")
graph.add_edge("chunk_document", "embed_and_store")
graph.add_edge("embed_and_store", "get_query")
graph.add_edge("search_chroma", "show_results")
graph.add_conditional_edges(
    "get_query",
    check_exit,
    {
        "end": END,
        "continue": "search_chroma",
    }
)
graph.add_edge("show_results", "get_query")

workflow = graph.compile()
final_state = workflow.invoke({'file_path': 'data.md'})
# print(final_state)