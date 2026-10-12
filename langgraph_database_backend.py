#Here look now As we are building our bot but our bot doesnt store histyory ok...that not there as i reload it it will get removed ok..so to solve these error what we need we need a databse so we will use sqlite with langgraph to store databse 


from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


# =========================
# 1. Load environment
# =========================

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
)


# =========================
# 2. Define graph state
# =========================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =========================
# 3. Define chatbot node
# =========================

def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# =========================
# 4. Build LangGraph
# =========================

builder = StateGraph(ChatState)

builder.add_node("chat_node", chat_node)

builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)


# =========================
# 5. Add conversation memory
# =========================

checkpointer = InMemorySaver()

chatbot = builder.compile(
    checkpointer=checkpointer
)


# =========================
# 6. Thread configuration
# =========================

def get_config(thread_id: str):
    return {
        "configurable": {
            "thread_id": str(thread_id)
        }
    }


# =========================
# 7. Retrieve chat history
# =========================

def get_chat_history(thread_id: str):
    config = get_config(thread_id)

    snapshot = chatbot.get_state(config)

    return snapshot.values.get("messages", [])


# =========================
# 8. Stream assistant response
# =========================

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
        if chunk["type"] != "messages":
            continue

        message_chunk, metadata = chunk["data"]

        # Stream only the assistant's chatbot-node output.
        if metadata.get("langgraph_node") != "chat_node":
            continue

        content = message_chunk.content

        # Handle normal text responses.
        if isinstance(content, str) and content:
            yield content

        # Handle content blocks, if the model returns them.
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and block.get("text")
                ):
                    yield block["text"]
