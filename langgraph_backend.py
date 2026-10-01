from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. INITIALIZE GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
)


# ============================================================
# 3. DEFINE CHAT STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 4. CHAT NODE
# ============================================================

def chat_node(state: ChatState):

    messages = state["messages"]

    try:
        response = llm.invoke(messages)

        return {
            "messages": [response]
        }

    except Exception as e:

        print(f"LLM Error: {e}")

        # Graceful fallback
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "Sorry, I am temporarily unable to "
                        "process your request. Please try again."
                    )
                }
            ]
        }


# ============================================================
# 5. CREATE GRAPH
# ============================================================

def build_graph():

    graph = StateGraph(ChatState)

    # Add node
    graph.add_node(
        "chat_node",
        chat_node
    )

    # START → chat_node
    graph.add_edge(
        START,
        "chat_node"
    )

    # chat_node → END
    graph.add_edge(
        "chat_node",
        END
    )

    return graph


# ============================================================
# 6. CHECKPOINTER
# ============================================================

checkpointer = InMemorySaver()


# ============================================================
# 7. COMPILE GRAPH
# ============================================================

builder = build_graph()

chatbot = builder.compile(
    checkpointer=checkpointer
)


# ============================================================
# 8. CHAT FUNCTION
# ============================================================

def chat(
    message: str,
    thread_id: str = "default"
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = chatbot.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        },
        config=config
    )

    return result["messages"][-1].content