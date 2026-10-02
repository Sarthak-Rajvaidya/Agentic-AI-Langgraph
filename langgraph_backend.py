from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# GROQ LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)


# --------------------------------------------------
# STATE
# --------------------------------------------------

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# --------------------------------------------------
# CHAT NODE
# --------------------------------------------------

def chat_node(state: ChatState):

    messages = state["messages"]

    try:

        response = llm.invoke(messages)

        return {
            "messages": [response]
        }

    except Exception as e:

        print(f"LLM Error: {e}")

        return {
            "messages": [
                AIMessage(
                    content=(
                        "Sorry, I am temporarily unable "
                        "to process your request. "
                        "Please try again."
                    )
                )
            ]
        }


# --------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------

builder = StateGraph(ChatState)

builder.add_node(
    "chat_node",
    chat_node
)

builder.add_edge(
    START,
    "chat_node"
)

builder.add_edge(
    "chat_node",
    END
)


# --------------------------------------------------
# CHECKPOINTER
# --------------------------------------------------

checkpointer = InMemorySaver()


# --------------------------------------------------
# COMPILE GRAPH
# --------------------------------------------------

chatbot = builder.compile(
    checkpointer=checkpointer
)


chatbot.stream(
    ...,
    stream_mode="messages",
    version="v2"
)