import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

from rag.qdrant.multitenancy.multitenancy_constants import VECTOR_DB_COLLECTION, HISTORY_DEPARTMENT_NAME, \
    SCIENCE_DEPARTMENT_NAME

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

def chunk_data(file_name):
    loader = TextLoader(file_name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n\n", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_documents(documents)


def tag_documents(docs, department):
    taggedDocs = []
    for doc in docs:
        doc.metadata = {"group_id": department}
        taggedDocs.append(doc)
    return taggedDocs



def insert_data_to_vector_store(docs):
    qdrant = Qdrant.from_documents(
        docs,
        OpenAIEmbeddings(),
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
        collection_name=VECTOR_DB_COLLECTION,
    )


def perform_data_insertion(department, file_name):
    docs = chunk_data(file_name)
    taggedDocs = tag_documents(docs, department)
    insert_data_to_vector_store(taggedDocs)


perform_data_insertion(HISTORY_DEPARTMENT_NAME, "history_dept_data.txt")
perform_data_insertion(SCIENCE_DEPARTMENT_NAME, "science_dept_data.txt")