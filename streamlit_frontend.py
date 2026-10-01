import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LangGraph AI Assistant",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 LangGraph AI Assistant")
st.caption("Built while learning Agentic AI with LangGraph + Groq")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Agent Configuration")

    st.markdown("""
    **Architecture**

    🖥️ Streamlit  
    ↓  
    🧠 LangGraph  
    ↓  
    💬 Groq LLM  
    """)

    st.divider()

    st.subheader("🧠 Concepts Implemented")

    st.markdown("""
    - StateGraph
    - State
    - Nodes
    - START / END
    - `add_messages`
    - Checkpointer
    - Thread-based memory
    - Conversation history
    """)

    st.divider()

    st.subheader("🔗 Thread")

    st.code("thread-1")

    if st.button("🗑️ Clear Chat"):

        st.session_state["message_history"] = []

        st.rerun()


# --------------------------------------------------
# THREAD CONFIGURATION
# --------------------------------------------------

CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

user_input = st.chat_input(
    "Ask something about AI, LangGraph, Python..."
)


# --------------------------------------------------
# HANDLE USER MESSAGE
# --------------------------------------------------

if user_input:

    # ----------------------------------------------
    # Display user message
    # ----------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)


    # ----------------------------------------------
    # Send message to LangGraph
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = chatbot.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG
            )

            ai_message = response["messages"][-1].content

            st.markdown(ai_message)


    # ----------------------------------------------
    # Save assistant response
    # ----------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )