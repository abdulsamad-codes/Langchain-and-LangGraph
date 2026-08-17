# Session handoff — Simple RAG (semantic search) with LangGraph + LangChain

> **Instructions for the AI picking this up:** The user is mid-way through building a project with me, step by step. Read this whole document before responding. Then continue exactly where "What's next" says to continue — do not restart, do not skip ahead, and do not dump all remaining code at once. See "How to guide the user" at the bottom for the interaction style to match.

## Project goal

A **simple RAG pipeline for semantic search only** — no LLM, no answer generation. Given a markdown file, the user wants to:
1. Chunk it
2. Embed the chunks
3. Store them in ChromaDB
4. Let the user type queries in a loop and get back the most relevant chunks (the raw text itself, not a generated answer)

The whole thing is being orchestrated as a **LangGraph** workflow, using **LangChain** only for thin wrappers (embeddings + vectorstore), not for the chunking logic.

## Design decisions already made (preserve these — don't relitigate)

- **No LLM anywhere.** This is retrieval-only. The "answer" to a query is the top-k matching chunks, displayed as-is.
- **Chunking is deliberately simple**: fixed-size character chunks, not header/section-aware splitting. The user explicitly asked to simplify away from a more complex header-based chunker earlier in the conversation. Header-aware chunking may come back "later" per the user, but don't reintroduce it unprompted.
  - `chunk_size = 1000` characters, `overlap = 100` characters
- **Embedding model**: local, via `sentence-transformers`, wrapped by LangChain's `HuggingFaceEmbeddings`. Model id: `sentence-transformers/all-MiniLM-L6-v2`. Runs on CPU, no API key, no network calls at inference time.
- **Vector store**: ChromaDB, wrapped by LangChain's `Chroma` class.
- **Orchestration**: LangGraph `StateGraph`, split conceptually into two phases:
  - **Indexing graph** (linear, runs once): load file → chunk → embed & store
  - **Query loop graph** (cyclic): get query → exit check → (if not exiting) embed query → search → show results → loop back to get query
  - The loop is a real graph cycle (a conditional edge pointing backward to an earlier node), not just a Python `while` loop wrapped around the graph. This was the user's explicit request.

## Tech stack & why (already explained to the user — don't re-explain unless asked)

```bash
pip install langgraph langchain-core langchain-huggingface langchain-chroma chromadb sentence-transformers
```

| Package | Why |
|---|---|
| `langgraph` | The graph engine — turns node functions + edges (including the loop-back edge) into a runnable workflow with shared state. |
| `langchain-core` | Base types shared by LangGraph/LangChain components. Pulled in automatically. |
| `langchain-huggingface` | Wraps `sentence-transformers` into `HuggingFaceEmbeddings`, a standard `embed_query`/`embed_documents` interface. |
| `langchain-chroma` | Wraps `chromadb` into a `Chroma` vectorstore class (`add_texts`, `similarity_search`). |
| `chromadb` / `sentence-transformers` | The actual underlying libraries the two wrappers above depend on. |

Note given to the user: LangChain has a built-in `CharacterTextSplitter` that could replace our hand-rolled chunker. We're intentionally keeping the hand-rolled version for now since the user already understands it — mention this as an easy future swap if it comes up, don't do it unprompted.

## Full code so far (copy-paste ready)

This is everything built up to this point, in one file (`rag_search.py`). It is **incomplete** — see "What's next" below.

```python
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class RAGState(TypedDict):
    file_path: str        # path to the markdown file, set once at the start
    raw_text: str          # holds the file contents between load and chunk
    chunks: list[str]      # populated by the chunking node during indexing
    vectorstore: Chroma    # created once during indexing, reused by every query
    query: str              # the current question, overwritten each loop turn
    results: list[str]      # chunks returned by the last search, overwritten each loop turn


def load_document(state: RAGState) -> dict:
    """Node: reads the markdown file from disk."""
    text = Path(state["file_path"]).read_text(encoding="utf-8")
    return {"raw_text": text}


def chunk_document(state: RAGState, chunk_size: int = 1000, overlap: int = 100) -> dict:
    """Node: splits the raw text into fixed-size overlapping chunks."""
    text = state["raw_text"]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return {"chunks": [c for c in chunks if c.strip()]}


graph = StateGraph(RAGState)

graph.add_node("load_document", load_document)
graph.add_node("chunk_document", chunk_document)

graph.set_entry_point("load_document")
graph.add_edge("load_document", "chunk_document")

# --- everything below this line is NOT YET BUILT ---
```

## Workflow shape (for reference — already shown to the user as two diagrams)

**Indexing phase (linear, runs once):**
```
start → load_document → chunk_document → embed_and_store → (enters loop)
```

**Query loop phase (cyclic):**
```
get_query → exit? ──yes──> END
              │
              no
              ↓
         embed_query → search_chroma → show_results ──┐
              ↑                                        │
              └────────────── loops back ───────────────┘
```

## What's done

- [x] Dependencies identified, install command given, each package explained
- [x] Imports explained
- [x] `RAGState` schema defined (including `raw_text` added when the chunk node needed it)
- [x] `load_document` node written and explained
- [x] `chunk_document` node written and explained
- [x] Graph created, both nodes added via `add_node`, entry point set, first edge wired via `add_edge`

## What's next, in order — continue here

1. **`embed_and_store` node** — instantiate `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`, build a `Chroma` vectorstore from `state["chunks"]` using that embedding function, and return it as `{"vectorstore": ...}`. Then wire `graph.add_edge("chunk_document", "embed_and_store")`.
2. **`get_query` node** — prompts via `input()` for a question, returns `{"query": ...}`. This is the entry point of the loop phase.
3. **Exit-check routing function** — a function (not a full node, just a router) that inspects `state["query"]` for something like `exit`/`quit`, used with `graph.add_conditional_edges()` to send flow to `END` or onward into the loop body.
4. **Search step** — decide with the user whether this is a separate `embed_query` node, or whether `vectorstore.similarity_search(query, k=...)` (which embeds internally) makes a manual embed step redundant. Flag this as an open decision, don't just pick one silently — the diagram shown to the user has them as separate boxes, so raise it before collapsing them.
5. **`show_results` node** — prints/displays the matched chunks from `state["results"]`.
6. **Wire the loop** — `add_conditional_edges` routing the exit-check, and a normal `add_edge` from `show_results` back to `get_query` to close the cycle.
7. **`graph.compile()`** — turns the graph into a runnable app. Do this only after every node and edge above exists.
8. **Run it** — `app.invoke({"file_path": "..."})` and confirm the loop actually runs end-to-end against a real markdown file.

## How to guide the user (interaction style — match this)

- **One step at a time.** Present one node/piece, explain it, then stop and wait for the user to say "next" (or similar) before continuing. Do not jump ahead or bundle multiple remaining steps into one response unless asked.
- **Explain the "why", not just the "what".** For every dependency, import, or design choice, say why it's needed/structured that way — this was explicitly requested early on and has been the pattern for the whole conversation.
- **Bias toward simplicity.** The user has already pushed back once on over-engineering (rejected header-aware chunking in favor of simple fixed-size chunking). Default to the simplest version of anything; offer more sophisticated versions as optional add-ons, not the default.
- **Code goes directly in the chat response** as markdown code blocks, not as separate files/artifacts — that's how this whole conversation has been done.
- The user is comfortable reading and writing code; they just want the reasoning made explicit alongside it.
