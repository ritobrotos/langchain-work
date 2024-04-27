from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging


def retrieve_data(faiss_db, question):
    llm = ChatOpenAI(temperature=0)
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=faiss_db.as_retriever(), llm=llm)

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    return multi_query_retriever.get_relevant_documents(query=question)
