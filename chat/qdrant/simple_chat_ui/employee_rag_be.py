import os
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import qdrant_client
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Configuration for Qdrant DB
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def get_qdrant_retriever(qdrant_collection):
    embeddings = OpenAIEmbeddings()
    qdrantClient = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY)
    qdrant = Qdrant(qdrantClient, qdrant_collection, embeddings, content_payload_key="text")
    return qdrant.as_retriever()


def get_docs_from_vector_store(user_question):
    retriever = get_qdrant_retriever("employee_collection")
    print("User question: ", user_question)
    retrieved_docs = retriever.get_relevant_documents(user_question)
    print("Number of documents found: ", len(retrieved_docs))
    return retrieved_docs


def answer_questions(user_question):
    retrieved_docs = get_docs_from_vector_store(user_question)

    template = """
    You are a question answering bot. You will be given a QUESTION and a set of paragraphs in the CONTENT section. 
    You need to answer the question using the text present in the CONTENT section. 
    If the answer is not present in the CONTENT text then reply `I don't have answer to the question`
    
    CONTENT: {document}
    QUESTION: {question}
    """

    prompt = PromptTemplate(
        input_variables=["document", "question"], template=template
    )

    output_parser = StrOutputParser()
    chain = prompt | ChatOpenAI(model_name="gpt-3.5-turbo") | output_parser
    llm_answer = chain.invoke({"document": retrieved_docs, "question": user_question})
    return llm_answer


# user_ques = "Emily Brown is from which department?"
# user_ques = "Which projects did Jane Smith worked on?"
# print(answer_questions(user_ques))
