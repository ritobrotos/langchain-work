import streamlit as st

from employee_rag_be import answer_questions

st.title("Employee Information Chatbot")

# Initialize rag history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display rag messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(""):
    # Add user message to rag history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in rag message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in rag message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = answer_questions(prompt)
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})