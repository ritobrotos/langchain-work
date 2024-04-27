from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


def store_embeddings(content_list):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=3000, chunk_overlap=300)
    docs = text_splitter.create_documents(content_list)

    embeddings = OpenAIEmbeddings()
    faiss_db = FAISS.from_documents(docs, embeddings)
    return faiss_db