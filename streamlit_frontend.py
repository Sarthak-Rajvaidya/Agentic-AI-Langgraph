import streamlit as st
import uuid

from langgraph_backend import chat_stream


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LangGraph AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

# Store all conversations
if "chats" not in st.session_state:
    st.session_state["chats"] = {}


# Create first chat
if "current_chat_id" not in st.session_state:

    chat_id = str(uuid.uuid4())

    st.session_state["current_chat_id"] = chat_id

    st.session_state["chats"][chat_id] = {
        "title": "New Chat",
        "messages": []
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # APPLICATION TITLE
    # --------------------------------------------------------

    st.title("🤖 LangGraph AI")


    # --------------------------------------------------------
    # AGENTIC AI CONCEPTS
    # --------------------------------------------------------

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
    - State
    - Nodes & Edges
    - START / END
    - `add_messages`
    - Checkpointer
    - Thread-based memory
    - Conversation history
    - LLM Streaming
    """)


    st.divider()


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    st.subheader("💬 Chat History")


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        chat_id = str(uuid.uuid4())

        st.session_state["current_chat_id"] = chat_id

        st.session_state["chats"][chat_id] = {
            "title": "New Chat",
            "messages": []
        }

        st.rerun()


    st.markdown("### Recent")


    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    chats = list(
        st.session_state["chats"].items()
    )

    # Newest chats first
    chats.reverse()


    for chat_id, chat in chats:

        title = chat["title"]

        # Highlight active chat
        if chat_id == st.session_state["current_chat_id"]:

            button_label = f"🟢 {title}"

        else:

            button_label = f"💬 {title}"


        if st.button(
            button_label,
            key=f"chat_{chat_id}",
            use_container_width=True
        ):

            st.session_state["current_chat_id"] = chat_id

            st.rerun()


# ============================================================
# CURRENT CHAT
# ============================================================

current_chat_id = st.session_state["current_chat_id"]

current_chat = st.session_state["chats"][current_chat_id]


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🤖 LangGraph AI Assistant")

st.caption(
    "Learning Agentic AI with LangGraph + Groq"
)


# ============================================================
# DISPLAY CURRENT CHAT
# ============================================================

for message in current_chat["messages"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Message LangGraph AI..."
)


# ============================================================
# HANDLE USER MESSAGE
# ============================================================

if user_input:

    # --------------------------------------------------------
    # CREATE CHAT TITLE
    # --------------------------------------------------------

    if current_chat["title"] == "New Chat":

        current_chat["title"] = (
            user_input[:35]
            + ("..." if len(user_input) > 35 else "")
        )


    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    current_chat["messages"].append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_input)


    # --------------------------------------------------------
    # STREAM AI RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            chat_stream(
                message=user_input,
                thread_id=current_chat_id
            )
        )


    # --------------------------------------------------------
    # SAVE COMPLETE AI RESPONSE
    # --------------------------------------------------------

    current_chat["messages"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )