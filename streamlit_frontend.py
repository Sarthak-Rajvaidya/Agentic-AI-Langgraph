import streamlit as st

with st.chat_message('user'):
    st.text('HI')
    
with st.chat_message('assistant'):
    st.text('How can I help You?')