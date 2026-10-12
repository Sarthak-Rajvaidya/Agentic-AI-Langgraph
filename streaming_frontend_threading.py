
import streamlit as st
import uuid

from langgraph_backend import chat_stream, get_chat_history


# ============================================================
# PAGE CONFIGURATION
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
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())


def format_message(message):
    """Convert a LangChain message into a Streamlit chat message."""

    if message.type == "human":
        role = "user"
    elif message.type == "ai":
        role = "assistant"
    else:
        # Ignore tool messages and other internal message types.
        return None

    content = message.content

    if isinstance(content, str):
        text_content = content

    elif isinstance(content, list):
        # Support text blocks returned by different model providers.
        text_parts = []

        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")

                if isinstance(text, str):
                    text_parts.append(text)

        text_content = "".join(text_parts)

    else:
        text_content = str(content)

    return {
        "role": role,
        "content": text_content,
    }


def make_chat_title(user_message, max_length=35):
    """Create a readable title from the first user message."""

    title = " ".join(user_message.strip().split())

    if not title:
        return "New Chat"

    if len(title) > max_length:
        return title[:max_length].rstrip() + "..."

    return title


def add_thread(thread_id):
    """Register a conversation in the sidebar."""

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)

    if thread_id not in st.session_state["chat_titles"]:
        st.session_state["chat_titles"][thread_id] = "New Chat"


def load_thread(thread_id):
    """Switch conversations and load messages from the backend."""

    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)

    try:
        saved_messages = get_chat_history(thread_id)

        formatted_messages = []

        for message in saved_messages:
            formatted = format_message(message)

            if formatted is not None:
                formatted_messages.append(formatted)

        st.session_state["message_history"] = formatted_messages

        # Restore the title from the first human message when possible.
        first_user_message = next(
            (
                message["content"]
                for message in formatted_messages
                if message["role"] == "user"
            ),
            None,
        )

        if first_user_message:
            st.session_state["chat_titles"][thread_id] = make_chat_title(
                first_user_message
            )

    except Exception as e:
        st.error(f"Could not load conversation: {e}")
        st.session_state["message_history"] = []


def reset_chat():
    """Create a new conversation with a fresh thread ID."""

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def select_thread(thread_id):
    """Sidebar callback for selecting a conversation."""

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

    st.subheader("🧠 Agentic AI Concepts")

    st.markdown(
        """
        **Architecture**

        🖥️ Streamlit  
        ↓  
        🧠 LangGraph  
        ↓  
        💬 Groq LLM
        """
    )

    st.markdown("**Concepts Implemented**")

    st.markdown(
        """
        - StateGraph
        - State management
        - Nodes and edges
        - START / END
        - `add_messages`
        - Checkpointer
        - Thread-based memory
        - Conversation history
        - Token-by-token streaming
        """
    )

    st.divider()

    st.subheader("💬 Chat History")

    if st.button("＋ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.caption("Select a conversation to reopen it.")

    for thread_id in st.session_state["chat_threads"]:

        title = st.session_state["chat_titles"].get(
            thread_id,
            "New Chat",
        )

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

st.caption(
    "Learning Agentic AI with LangGraph + Groq | "
    "Persistent conversation memory"
)


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
# HANDLE USER INPUT AND STREAM RESPONSE
# ============================================================

if user_input:

    thread_id = st.session_state["thread_id"]

    # Create a title from the first user message.
    if not any(
        message["role"] == "user"
        for message in st.session_state["message_history"]
    ):
        st.session_state["chat_titles"][thread_id] = make_chat_title(
            user_input
        )

    # Add the user message to the UI history.
    user_message = {
        "role": "user",
        "content": user_input,
    }

    st.session_state["message_history"].append(user_message)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the response from LangGraph.
    with st.chat_message("assistant"):

        try:
            response_stream = chat_stream(
                message=user_input,
                thread_id=thread_id,
            )

            ai_message = st.write_stream(response_stream)

            # Save only a successfully generated, non-empty response.
            if isinstance(ai_message, str) and ai_message.strip():

                st.session_state["message_history"].append(
                    {
                        "role": "assistant",
                        "content": ai_message,
                    }
                )

            else:
                st.warning(
                    "The assistant did not return a text response."
                )

        except Exception as e:
            st.error(
                "An error occurred while generating the response. "
                "Please try again."
            )

            # Show technical details during development.
            st.exception(e)
