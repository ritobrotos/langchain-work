import streamlit as st
import multitenancy_rag_chatbot
import multitenancy_chat_auth

def credentials_entered():
    print("Checking credentials")
    entered_username = st.session_state["usernameInput"].strip()
    entered_password = st.session_state["passwdInput"].strip()

    user = multitenancy_chat_auth.validate_credentials(entered_username, entered_password)
    if user is not None:
        # If the user is successfully authenticated then store the username and the user department in the session state
        st.session_state["username"] = user['username']
        st.session_state["department"] = user['department']
    else:
        # If the user is not authenticated then throw error
        st.error("Invalid username or password")


def authenticate_user():
    if "username" not in st.session_state or "department" not in st.session_state:
        st.text_input(label="Username: ", value="", key="usernameInput")
        st.text_input(label="Password: ", value="", key="passwdInput", type="password", on_change=credentials_entered)
        return False
    else:
        return True



if authenticate_user():
    if "username" in st.session_state and "department" in st.session_state:
        username = st.session_state["username"]
        department = st.session_state["department"]
        print("Found username and department in session: ", username, department)
        st.title(f"Hello {username}")
        # st.session_state["user"] = username
        # st.session_state["department"] = department

    department_name = st.session_state["department"]
    query_generator_llm_chain = multitenancy_rag_chatbot.query_generator_llm()
    quadrant_retriever = multitenancy_rag_chatbot.get_qdrant_retriever(department_name)

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
            relevant_doc = multitenancy_rag_chatbot.get_docs_from_vector_store(formed_question, quadrant_retriever)
            # Send the fetched document from Vector store and the formed question to LLM to extract answer from the question
            answer = multitenancy_rag_chatbot.answer_questions(relevant_doc, formed_question)
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
