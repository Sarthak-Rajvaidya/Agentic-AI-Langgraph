#Here in this file we will build reume chat like feature that is present in gpt
import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid 
#uuid - random naya thread id generate karnekye liye"
# st.session_state -> dict -> 

# ****************utility functionb************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

#new utily fun -(1) new thread id

def reset_chat():
    thread_id = generate_thread_id()
    

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
    
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    
#*************************SidebarUI***********
st.sidebar.title('Langgraph chatbot')
st.sidebar.button('New Chat')
st.sidebar.header('My Conversation')

st.sidebar.text(st.session_state['thread_id'])

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
        
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # first add the message to message_history
    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config= CONFIG,
                stream_mode= 'messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})