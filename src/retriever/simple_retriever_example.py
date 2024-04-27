def retrieve_data(faiss_db, question):
    simple_retriever = faiss_db.as_retriever(search_kwargs={"k": 2})
    return simple_retriever.get_relevant_documents(question)
