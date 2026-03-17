import streamlit as st
from rag_engine import ask_ai

st.set_page_config(page_title="Smart Node AI Assistant")

st.title("Smart Node AI Assistant")

# store conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# user input
prompt = st.chat_input("Ask a question")

if prompt:

    # show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # get AI response
    answer = ask_ai(prompt)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )