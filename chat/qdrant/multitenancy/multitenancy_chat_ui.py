import streamlit as st
import rag_chat_bot

def creds_entered():
    entered_username = st.session_state["user"].strip()
    entered_password = st.session_state["passwd"].strip()

    users = [
        {'username': 'Emily', 'password': 'pass123', 'department': 'user_1'},
        {'username': 'Benjamin', 'password': 'pass456', 'department': 'user_2'}
    ]

    authenticated = False
    selectedUser = None

    # Iterate through the list of users to check if the entered username and password matches
    for user in users:
        if entered_username == user['username'] and entered_password == user['password']:
            authenticated = True
            selectedUser = user
            break

    st.session_state["authenticated"] = authenticated
    if authenticated == True:
        # If the user is successfully authenticated then store the username and the user department in the session state
        st.session_state["user"] = selectedUser['username']
        st.session_state["department"] = selectedUser['department']

    if not authenticated:
        # If the user is not authenticated then throw error
        st.error("Invalid username or password")


def authenticate_user():
    if "authenticated" not in st.session_state:
        st.text_input(label="Username: ", value="", key="user")
        st.text_input(label="Password: ", value="", key="passwd", type="password", on_change=creds_entered)
        return False
    else:
        if st.session_state["authenticated"]:
            return True
        else:
            st.text_input(label="Username: ", value="", key="user", on_change=creds_entered)
            st.text_input(label="Password: ", value="", key="passwd", type="password", on_change=creds_entered)
            return False

if authenticate_user():
    if "user" in st.session_state:
        username = st.session_state["user"]
        st.title(f"Hello {username}")
        st.session_state["user"] = username
        st.session_state["department"] = st.session_state["department"]

    department_name = st.session_state["department"]
    query_generator_llm_chain = rag_chat_bot.query_generator_llm()
    quadrant_retriever = rag_chat_bot.get_qdrant_retriever(department_name)

    st.title("Student RAG Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if human_message := st.chat_input(""):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": human_message})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(human_message)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            # Form question based on the users last message and chat history
            formed_question = query_generator_llm_chain.run(human_message)
            # Fetch document from Vector store
            relevant_doc = rag_chat_bot.get_docs_from_vector_store(formed_question, quadrant_retriever)
            # Send the fetched document from Vector store and the formed question to LLM to extract answer from the question
            answer = rag_chat_bot.answer_questions(relevant_doc, formed_question)
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
