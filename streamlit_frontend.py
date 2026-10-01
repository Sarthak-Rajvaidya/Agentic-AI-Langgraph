import streamlit as st
import uuid

from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="LangGraph AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chats" not in st.session_state:
    st.session_state["chats"] = {}

if "current_chat_id" not in st.session_state:
    chat_id = str(uuid.uuid4())

    st.session_state["current_chat_id"] = chat_id

    st.session_state["chats"][chat_id] = {
        "title": "New Chat",
        "messages": []
    }


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🤖 LangGraph AI")

    # New Chat
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

    st.divider()

    st.subheader("💬 Chat History")

    # Display previous chats
    for chat_id, chat in st.session_state["chats"].items():

        title = chat["title"]

        if st.button(
            title,
            key=f"chat_{chat_id}",
            use_container_width=True
        ):

            st.session_state["current_chat_id"] = chat_id

            st.rerun()


# --------------------------------------------------
# CURRENT CHAT
# --------------------------------------------------

current_chat_id = st.session_state["current_chat_id"]

current_chat = st.session_state["chats"][current_chat_id]


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 LangGraph AI Assistant")

st.caption(
    "Agentic AI learning project • LangGraph + Groq"
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

CONFIG = {
    "configurable": {
        "thread_id": current_chat_id
    }
}


# --------------------------------------------------
# DISPLAY CURRENT CHAT
# --------------------------------------------------

for message in current_chat["messages"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

user_input = st.chat_input(
    "Message LangGraph AI..."
)


if user_input:

    # ----------------------------------------------
    # Create title from first message
    # ----------------------------------------------

    if current_chat["title"] == "New Chat":

        current_chat["title"] = (
            user_input[:30]
            + ("..." if len(user_input) > 30 else "")
        )


    # ----------------------------------------------
    # Save user message
    # ----------------------------------------------

    current_chat["messages"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)


    # ----------------------------------------------
    # Call LangGraph
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = chatbot.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },
                config=CONFIG
            )

            ai_message = response[
                "messages"
            ][-1].content

            st.markdown(ai_message)


    # ----------------------------------------------
    # Save AI response
    # ----------------------------------------------

    current_chat["messages"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )