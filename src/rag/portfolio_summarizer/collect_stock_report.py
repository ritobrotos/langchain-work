import os

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.document import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant

from rag.portfolio_summarizer.portfolio_summarizer_constants import VECTOR_DB_COLLECTION, portfolio

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 75

llm = Ollama(model="mistral")
ollamaEmbeddings = OllamaEmbeddings(model="mistral")

cnbc_quarter_report = {
    'GOOGL': {"report_link": 'https://www.cnbc.com/2023/10/24/alphabet-googl-earnings-q3-2023.html', "quarter": "Q3", "year": "2023"},
    'AAPL': {"report_link": 'https://www.cnbc.com/2023/11/02/apple-aapl-earnings-report-q4-2023.html', "quarter": "Q3", "year": "2023"},
    'TSLA': {"report_link": 'https://www.cnbc.com/2023/10/18/tesla-tsla-earnings-q3-2023.html', "quarter": "Q3", "year": "2023"},
    'MSFT': {"report_link": 'https://www.cnbc.com/2023/10/24/microsoft-msft-q1-earnings-report-2024.html', "quarter": "Q3", "year": "2023"},
    'WMT': {"report_link": 'https://www.cnbc.com/2023/11/16/walmart-wmt-earnings-q3-2024-.html', "quarter": "Q3", "year": "2023"}
}


# To locate the earning report, we're meticulously extracting relevant information like financial data,
# key metrics, and analyst commentary from this webpage, while discarding distractions such as navigation menus,
# page reference links, banner ads, social media widgets, contact information, and legal disclaimers.
def extract_content(url):
    template = """You are an experienced equity research analyst and you do a fantastic job of extracting company's earning information from the `company's earning report`. 

        You are instructed that, if the given text doesnot belong to `company's earning report` then ignore the text and return only the text `**NA**`.

        You are instructed to extract the exact lines from the `company's earning report` as it is. Don't update or modify the extracted lines.

        Below is the `company's earning report`:
        {earning_report}
    """

    chunked_docs = chunk_web_data(url)
    extracted_text_content = "";
    for doc in chunked_docs:
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        data = chain.invoke({"earning_report": doc}).strip()
        if "**NA**" in data:
            continue
        extracted_text_content += data

    return extracted_text_content


# Breaking down the webpage content into small documents so that it can be passed to the LLM to remove the noise
# from the financial data
def chunk_web_data(url):
    documents = scrape_content(url)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_documents(documents)


# We are using Selenium to scrape the webpage content of the given URL
def scrape_content(url):
    urls = [url]
    loader = SeleniumURLLoader(urls=urls)
    return loader.load()


# The LLM filtered data is now broken down smaller documents before storing them in the Qdrant Vector store. In the
# metadata we are passing the company ticker, and the quarter and the year of the earning report. This will help in
# fetching the relevant information.
def chunk_text_data(text, ticker, quarter, year):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    metadata_source = ticker + "-" + quarter + "-" + year
    document = Document(page_content=text, metadata={"source": metadata_source})
    return text_splitter.split_documents([document])


# Using this function we are inserting the docs in the Qdrant DB
def insert_data_to_vector_store(docs):
    Qdrant.from_documents(
        docs,
        ollamaEmbeddings,
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
        collection_name=VECTOR_DB_COLLECTION,
    )


# This is the main function which orchestrates the entire flow from fetching content to storing them in the vector store.
def main():
    for entry in portfolio:
        ticker = entry["ticker"]
        company_name = entry["company_name"]

        report_dict = cnbc_quarter_report[ticker]
        report_link = report_dict["report_link"]
        year = report_dict["year"]
        quarter = report_dict["quarter"]

        print("Extracting content for: ", company_name)
        extracted_text_content = extract_content(report_link)

        print("Chunking document for " + ticker + "-" + quarter + "-" + year)
        chunked_docs = chunk_text_data(extracted_text_content, ticker, quarter, year)

        print("Inserting Report to Qdrant for " + company_name)
        insert_data_to_vector_store(chunked_docs)