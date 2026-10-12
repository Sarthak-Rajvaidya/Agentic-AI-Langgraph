
from pathlib import Path
import sqlite3
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


# ==========================================
# 1. Environment and model
# ==========================================

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
)


# ==========================================
# 2. Graph state
# ==========================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================
# 3. Chatbot node
# ==========================================

def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ==========================================
# 4. Build graph
# ==========================================

builder = StateGraph(ChatState)

builder.add_node("chat_node", chat_node)
builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)


# ==========================================
# 5. SQLite database and checkpointer
# ==========================================

# Use a fixed path beside this Python file.
# This prevents accidentally creating a different DB
# when the working directory changes.
DB_PATH = Path(__file__).resolve().parent / "chatbot.db"

conn = sqlite3.connect(
    str(DB_PATH),
    check_same_thread=False,
    timeout=30,
)

# Create LangGraph checkpoint tables if necessary.
checkpointer = SqliteSaver(conn)
checkpointer.setup()

# Compile graph with persistent checkpoints.
chatbot = builder.compile(checkpointer=checkpointer)


# ==========================================
# 6. Persistent conversation registry
# ==========================================

# LangGraph checkpoints save messages, but the sidebar
# also needs a persistent list of conversation IDs.
with conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New Chat',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def get_config(thread_id: str):
    return {
        "configurable": {
            "thread_id": str(thread_id)
        }
    }


def register_thread(thread_id: str, title: str = "New Chat"):
    """Save a conversation ID so the sidebar can find it later."""
    with conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO chat_threads (thread_id, title)
            VALUES (?, ?)
            """,
            (str(thread_id), title),
        )


def update_thread_title(thread_id: str, title: str):
    """Update a conversation's sidebar title."""
    register_thread(thread_id, title)

    with conn:
        conn.execute(
            """
            UPDATE chat_threads
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE thread_id = ?
            """,
            (title, str(thread_id)),
        )


def list_threads():
    """Return saved conversations, newest activity first."""
    cursor = conn.execute(
        """
        SELECT thread_id, title
        FROM chat_threads
        ORDER BY updated_at DESC, created_at DESC
        """
    )

    return [
        {"thread_id": row[0], "title": row[1]}
        for row in cursor.fetchall()
    ]


# ==========================================
# 7. Retrieve conversation history
# ==========================================

def get_chat_history(thread_id: str):
    config = get_config(thread_id)

    snapshot = chatbot.get_state(config)

    return snapshot.values.get("messages", [])


# ==========================================
# 8. Stream assistant response
# ==========================================

def chat_stream(message: str, thread_id: str):
    thread_id = str(thread_id)
    config = get_config(thread_id)

    register_thread(thread_id)

    # Stream model output as it is generated.
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

        # Ignore messages from other graph nodes.
        if metadata.get("langgraph_node") != "chat_node":
            continue

        content = message_chunk.content

        if isinstance(content, str) and content:
            yield content

        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and block.get("text")
                ):
                    yield block["text"]


# ==========================================
# 9. Close database connection when appropriate
# ==========================================

def close_database():
    conn.close()
