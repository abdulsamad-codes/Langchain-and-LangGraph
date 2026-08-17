from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain.tools import tool, BaseTool
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import requests
import operator
import os 
# from pydantic import 

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
config = {"configurable": {"thread_id": "my_chat"}}



load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, api_key=api_key)
# print(llm.invoke("hi how are you?").content)



class WeatherState(TypedDict):
    messages: Annotated[list, operator.add]


@tool
def temperature_tool(city: str) -> float:
    ''' This tools has been given a city name and it returns the temperature of that city.'''
    my_api = '4753bbf1a1c715a1440c52412a447bcc'
    
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={my_api}'
    response = requests.get(url)
    data = response.json()
    return data['main']['temp']

@tool
def convert_temperature(current: float):
    ''' This tool has been given an input temperature in kelvin and it convert return it in celsius'''
    return (current - 273.15)



tools = [temperature_tool, convert_temperature]
llm_with_tools = llm.bind_tools(tools)
def agent(state: WeatherState):
    system_prompt = (
    "You are an experienced weather analyzer. Your job is to find the temperature of a city requested by the user. "
    "Follow these steps:\n"
    "1. Identify the city name from the user's message.\n"
    "2. Use the `temperature_tool` to fetch the weather data for that city.\n"
    "3. Look at the raw temperature returned (which is in Kelvin).\n"
    "4. Immediately pass that Kelvin temperature into the `convert_temperature` tool to get Celsius.\n"
    "5. Provide the final temperature to the user in a clear, friendly sentence.\n"
    "Only use the provided tools. Do not invent weather data.")
    # messages = [system_prompt] + state['messages']
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

3
def should_continue(state: WeatherState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    return 'end'

graph = StateGraph(WeatherState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)


graph.add_edge("tools", "agent")
# print(temperature_tool.invoke('Peshawar')['main']['temp'])
# print("hello")

app = graph.compile(checkpointer=memory)
conversation_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break
    # conversation_history.append(HumanMessage(content=user_input))
    # result = app.invoke({"messages": conversation_history})
    # conversation_history = result["messages"]
    # print("AI:", conversation_history)
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    print("AI:", result["messages"][-1].content)