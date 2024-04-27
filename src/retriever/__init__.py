from retriever.simple_retriever_example import retrieve_data
from vector_store_utilities.faiss_db_utility import store_embeddings
from web_utilities.scraper import get_scraped_data


def run_retriever():
    # Fetch content by scraping webpage
    urls = [
        "https://www.pinecone.io/learn/vector-embeddings/",
        # "https://www.pinecone.io/learn/vector-similarity/",
        # "https://www.pinecone.io/learn/vector-database/",
        # "https://www.pinecone.io/learn/vector-embeddings-for-developers/",
        # "https://www.pinecone.io/learn/vector-search-basics/"
    ]

    content_list = get_scraped_data(urls)
    appended_text = ""
    for content in content_list:
        appended_text += content

    faiss_db = store_embeddings(appended_text)
    question = "What is Vector DB?"
    return retrieve_data(faiss_db, question)


# result = run_retriever()
# print(result)