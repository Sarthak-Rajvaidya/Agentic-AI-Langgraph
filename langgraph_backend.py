
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GROQ MODEL
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
)


# ============================================================
# GRAPH STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# CHAT NODE
# ============================================================

def chat_node(state: ChatState):
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    except Exception as e:
        print(f"LLM Error: {e}")

        return {
            "messages": [
                AIMessage(
                    content=(
                        "Sorry, I couldn't process that request. "
                        "Please try again."
                    )
                )
            ]
        }


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(ChatState)

builder.add_node("chat_node", chat_node)

builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)


# ============================================================
# CHECKPOINTER
# ============================================================

checkpointer = InMemorySaver()

chatbot = builder.compile(checkpointer=checkpointer)


# ============================================================
# HELPER: BUILD THREAD CONFIG
# ============================================================

def get_config(thread_id: str):
    return {
        "configurable": {
            "thread_id": str(thread_id)
        }
    }


# ============================================================
# HELPER: LOAD CONVERSATION FROM LANGGRAPH
# ============================================================

def get_chat_history(thread_id: str):
    config = get_config(thread_id)

    snapshot = chatbot.get_state(config)

    return snapshot.values.get("messages", [])


# ============================================================
# STREAMING FUNCTION
# ============================================================

def chat_stream(message: str, thread_id: str):
    config = get_config(thread_id)

    for chunk in chatbot.stream(
        {
            "messages": [
                HumanMessage(content=message)
            ]
        },
        config=config,
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] == "messages":
            message_chunk, metadata = chunk["data"]

            # Stream only output from our chat node.
            if (
                metadata.get("langgraph_node") == "chat_node"
                and message_chunk.content
            ):
                # Handle ordinary text chunks.
                if isinstance(message_chunk.content, str):
                    yield message_chunk.content
