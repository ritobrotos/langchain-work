from retriever.simple_retriever_example import retrieve_data
from vector_store_utilities.faiss_db_utility import store_embeddings


def run_retriever():
    # Fetch content by scraping data
    content_list = ""
    faiss_db = store_embeddings(content_list)
    question = "What is Vector DB?"
    return retrieve_data(faiss_db, question)