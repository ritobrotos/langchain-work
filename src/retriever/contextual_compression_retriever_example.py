from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI


def retrieve_data(faiss_db, question):
    llm = ChatOpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                           base_retriever=faiss_db.as_retriever())

    return compression_retriever.get_relevant_documents(question)
