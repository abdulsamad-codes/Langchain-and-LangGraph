from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

from retriever import retrieve, catalog


load_dotenv()

# Format the catalog into readable text for the system prompt,
# so the LLM knows what categories/titles exist to ask about.
catalog_text = "\n".join(
    f"- Category: {c['category']} | Title: {c['title']}" for c in catalog
)

SYSTEM_PROMPT = f"""You are a retrieval assistant for the Dastak KP Citizens app documentation.

You have access to a `retrieve` tool. Use it to answer the user's question
by fetching relevant document content — do not answer from your own knowledge.

Available categories and titles in the document:
{catalog_text}

When the user's query clearly matches a category or title above, mention it
naturally so the tool can match against it. Always call the tool before answering.
"""

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

tools = [retrieve]
# llm = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools)
llm = ChatGroq(model="openai/gpt-oss-120b").bind_tools(tools)
def call_llm(state: GraphState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: GraphState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return "end"

graph = StateGraph(GraphState)
graph.add_node("call_llm", call_llm)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("call_llm")
graph.add_conditional_edges("call_llm", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "call_llm")

workflow = graph.compile()

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() in {"exit", "quit"}:
            break
        result = workflow.invoke({"messages": [{"role": "user", "content": query}]})
        print("\n" + result["messages"][-1].content)