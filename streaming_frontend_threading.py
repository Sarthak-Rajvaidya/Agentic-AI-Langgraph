
import streamlit as st
import uuid

from langgraph_backend import chat_stream, get_chat_history


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LangGraph AI Assistant",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def generate_thread_id():
    # Store UUIDs as strings for consistent thread IDs.
    return str(uuid.uuid4())


def format_message(message):
    """Convert a LangChain message into a Streamlit chat message."""

    if message.type == "human":
        role = "user"
    elif message.type == "ai":
        role = "assistant"
    else:
        return None

    content = message.content

    # Handle text and basic text-block content.
    if isinstance(content, str):
        text_content = content
    elif isinstance(content, list):
        text_content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    else:
        text_content = str(content)

    return {
        "role": role,
        "content": text_content,
    }


def load_thread(thread_id):
    """Switch the active thread and restore its checkpointed messages."""

    st.session_state["thread_id"] = thread_id

    try:
        saved_messages = get_chat_history(thread_id)

        st.session_state["message_history"] = [
            formatted
            for message in saved_messages
            if (formatted := format_message(message)) is not None
        ]

    except Exception as e:
        st.error(f"Could not load conversation: {e}")
        st.session_state["message_history"] = []


def reset_chat():
    """Create a new, empty conversation."""

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def add_thread(thread_id):
    """Register a conversation in the sidebar history."""

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)

    if thread_id not in st.session_state["chat_titles"]:
        st.session_state["chat_titles"][thread_id] = "New Chat"


def select_thread(thread_id):
    """Callback for sidebar conversation buttons."""
    load_thread(thread_id)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

add_thread(st.session_state["thread_id"])


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 LangGraph AI")

    # Keep the concepts section at the top.
    st.subheader("🧠 Agentic AI Concepts")

    st.markdown("""
    **Architecture**

    🖥️ Streamlit  
    ↓  
    🧠 LangGraph  
    ↓  
    💬 Groq LLM
    """)

    st.markdown("**Concepts Implemented**")

    st.markdown("""
    - StateGraph
    - State management
    - Nodes and edges
    - START / END
    - `add_messages`
    - Checkpointer
    - Thread-based memory
    - Conversation history
    - Token-by-token streaming
    """)

    st.divider()

    # Chat history section.
    st.subheader("💬 Chat History")

    if st.button("＋ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.caption("Select a conversation to reopen it")

    # Newest conversations appear first.
    for thread_id in st.session_state["chat_threads"]:

        title = st.session_state["chat_titles"].get(
            thread_id,
            "New Chat",
        )

        # Indicate which conversation is currently selected.
        if thread_id == st.session_state["thread_id"]:
            label = f"🟢 {title}"
        else:
            label = f"💬 {title}"

        st.button(
            label,
            key=f"thread_{thread_id}",
            use_container_width=True,
            on_click=select_thread,
            args=(thread_id,),
        )


# ============================================================
# MAIN CHAT AREA
# ============================================================

st.title("🤖 LangGraph AI Assistant")

st.caption("Learning Agentic AI with LangGraph + Groq")


# ============================================================
# DISPLAY ACTIVE CONVERSATION
# ============================================================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input("Message LangGraph AI...")


# ============================================================
# HANDLE USER INPUT
# ============================================================

if user_input:

    thread_id = st.session_state["thread_id"]

    # Create a title from the first message.
    if not st.session_state["message_history"]:
        title = user_input[:35]

        if len(user_input) > 35:
            title += "..."

        st.session_state["chat_titles"][thread_id] = title

    # Save and display the user message.
    user_message = {
        "role": "user",
        "content": user_input,
    }

    st.session_state["message_history"].append(user_message)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the assistant response token by token.
    with st.chat_message("assistant"):

        try:
            ai_message = st.write_stream(
                chat_stream(
                    message=user_input,
                    thread_id=thread_id,
                )
            )

        except Exception as e:
            st.error(f"An error occurred while generating a response: {e}")
            ai_message = ""

    # Save the completed response for the UI history.
    if ai_message:
        st.session_state["message_history"].append(
            {
                "role": "assistant",
                "content": ai_message,
            }
        )
