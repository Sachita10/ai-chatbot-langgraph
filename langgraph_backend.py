#!/usr/bin/env python
# coding: utf-8

# In[13]:


from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama


# In[14]:


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOllama(model="llama2")


# In[15]:


def chat_node(state):
    messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages": [response]}


# In[16]:


checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# In[ ]:




