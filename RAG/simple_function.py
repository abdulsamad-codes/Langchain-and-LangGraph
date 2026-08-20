from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
import re
import os 
load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-120b')
# print(llm.invoke("What is AI?").content)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


def retrieval(query: str = None, category: str = None, title: str = None, k: int = 1):
    pass
    if title:
        pass
    elif category:
        pass
    elif query:
        pass
    else:
        return 'Please provide a query, category, or title.'

# retrieval()
# print("hello")
print(retrieval())