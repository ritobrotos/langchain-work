from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import qdrant_client
from constants import QDRANT_URL, QDRANT_API_KEY
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain.chat_models import ChatOpenAI

@st.cache_resource
def query_generator_llm():
    template = """Given the following user prompt and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.
      You should follow the following rules when generating and answer:
      - Always prioritize the user prompt over the conversation log.
      - Ignore any conversation log that is not directly related to the user prompt.
      - Only attempt to answer if a question was posed.
      - The question should be a single sentence.
      - You should remove any punctuation from the question.
      - You should remove any words that are not relevant to the question.
      - If you are unable to formulate a question, respond with the same USER PROMPT you got.

    {chat_history}
    {human_input}
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    print("Instantiating query generator LLM chain with memory")

    return LLMChain(
        llm=OpenAI(),
        prompt=prompt,
        verbose=False,
        memory=memory
    )


@st.cache_resource
def get_qdrant_retriever(department_name):
    print("Creating retriever for department: ", department_name)
    embeddings = OpenAIEmbeddings()
    qdrantClient = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY)
    qdrant = Qdrant(qdrantClient, "my_documents", embeddings)
    return qdrant.as_retriever(search_kwargs={'filter': {'group_id': department_name}})


def get_docs_from_vector_store(formed_question, retriever):
    print("LLM formed question: ", formed_question)
    resultDocs = retriever.get_relevant_documents(formed_question)
    print("Number of documents found: ", len(resultDocs))
    if len(resultDocs) > 0:
        retrieved_doc = resultDocs[0].page_content
        print("Retrieved document: ", retrieved_doc)
        return retrieved_doc
    else:
        print("No documents found")
        return "No response"


def answer_questions(document_data, formed_question):
    template = """You are a question answering bot. You will be given a QUESTION and a set of paragraphs in the CONTENT section. You need to answer the question using the text present in the CONTENT section. If the answer is not present in the CONTENT text then reply `I don't have answer to the question`
    CONTENT: {document}
    QUESTION: {question}
    """

    prompt = PromptTemplate(
        input_variables=["document", "question"], template=template
    )

    print("Calling LLM to fetch answer of the question: ", formed_question)
    llm_answer = LLMChain(
        llm=ChatOpenAI(),
        prompt=prompt,
        verbose=False,
    ).run(document=document_data, question=formed_question)
    print("Answer returned by LLM: ", llm_answer)
    return llm_answer