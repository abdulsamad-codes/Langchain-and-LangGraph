import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.7
)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    print("Multiply tool is called")
    return a * b

tools = [multiply]
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, operator.add]

def call_model(state: State):
    system_prompt = """ you are a helpful assistant, Always use the available tools to answer the query"""
    result = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [result]}

# def call_tool(state: State):
#     last_message = state["messages"][-1]

#     if not last_message.tool_calls:
#         return {"messages": state["messages"]}

#     tool_call = last_message.tool_calls[0]
#     tool_result = multiply.invoke(tool_call["args"])

#     return {
#         "messages": state["messages"] + [
#             ToolMessage(
#                 content=str(tool_result),
#                 tool_call_id=tool_call["id"],
#                 name=tool_call["name"]
#             )
#         ]
#     }

def should_continue(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"



workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

app = workflow.compile()

result = app.invoke({
    "messages": [HumanMessage(content="what is 4 * 8?")]
})

print(result["messages"][-1])