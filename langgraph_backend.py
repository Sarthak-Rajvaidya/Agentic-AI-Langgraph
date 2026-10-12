
from pathlib import Path
import sqlite3
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================
# 1. ENVIRONMENT AND LLM
# ============================================================

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
)


# ============================================================
# 2. DEFINE GRAPH STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 3. CHATBOT NODE
# ============================================================

def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


# ============================================================
# 4. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(ChatState)

builder.add_node("chat_node", chat_node)

builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)


# ============================================================
# 5. SQLITE DATABASE
# ============================================================

# Always create the database beside this Python file.
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chatbot.db"

print(f"Backend file: {Path(__file__).resolve()}")
print(f"SQLite database: {DB_PATH}")

# Connect to SQLite.
conn = sqlite3.connect(
    str(DB_PATH),
    check_same_thread=False,
    timeout=30,
)

# Enable foreign-key support for future relational tables.
conn.execute("PRAGMA foreign_keys = ON")

# Initialize LangGraph checkpoint tables.
checkpointer = SqliteSaver(conn)
checkpointer.setup()

# Compile the graph with persistent checkpointing.
chatbot = builder.compile(
    checkpointer=checkpointer
)


# ============================================================
# 6. PERSISTENT CONVERSATION REGISTRY
# ============================================================

# Checkpoints store graph state and messages.
# This table stores conversation IDs and sidebar titles.

with conn:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New Chat',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def get_config(thread_id: str):
    """Create LangGraph configuration for a conversation."""

    return {
        "configurable": {
            "thread_id": str(thread_id)
        }
    }


def register_thread(
    thread_id: str,
    title: str = "New Chat",
):
    """Register a conversation in SQLite."""

    with conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO chat_threads
                (thread_id, title)
            VALUES (?, ?)
            """,
            (str(thread_id), title),
        )


def update_thread_title(
    thread_id: str,
    title: str,
):
    """Save a conversation title and update its activity time."""

    register_thread(thread_id, title)

    with conn:
        conn.execute(
            """
            UPDATE chat_threads
            SET title = ?,
                updated_at = CURRENT_TIMESTAMP
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
        {
            "thread_id": row[0],
            "title": row[1],
        }
        for row in cursor.fetchall()
    ]


# ============================================================
# 7. RETRIEVE CHAT HISTORY
# ============================================================

def get_chat_history(thread_id: str):
    """Retrieve checkpointed messages for a conversation."""

    config = get_config(thread_id)

    snapshot = chatbot.get_state(config)

    return snapshot.values.get("messages", [])


# ============================================================
# 8. STREAM ASSISTANT RESPONSE
# ============================================================

def chat_stream(message: str, thread_id: str):
    """
    Stream the assistant response token by token.

    LangGraph saves the conversation state using the
    persistent SQLite checkpointer.
    """

    thread_id = str(thread_id)
    config = get_config(thread_id)

    # Determine whether this is the first user message.
    previous_messages = get_chat_history(thread_id)

    is_new_conversation = not any(
        getattr(msg, "type", None) == "human"
        for msg in previous_messages
    )

    # Ensure the conversation appears in the registry.
    register_thread(thread_id)

    if is_new_conversation:
        title = " ".join(message.strip().split())

        if len(title) > 35:
            title = title[:35].rstrip() + "..."

        update_thread_title(
            thread_id,
            title or "New Chat",
        )

    # Update activity time for existing conversations.
    else:
        with conn:
            conn.execute(
                """
                UPDATE chat_threads
                SET updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
                """,
                (thread_id,),
            )

    # Run LangGraph and stream message chunks.
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

        # Only stream output from our chatbot node.
        if metadata.get("langgraph_node") != "chat_node":
            continue

        content = message_chunk.content

        if isinstance(content, str) and content:
            yield content

        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue

                if (
                    block.get("type") == "text"
                    and block.get("text")
                ):
                    yield block["text"]


# ============================================================
# 9. DATABASE CLEANUP
# ============================================================

def close_database():
    """Close the SQLite connection when the app shuts down."""

    conn.close()
