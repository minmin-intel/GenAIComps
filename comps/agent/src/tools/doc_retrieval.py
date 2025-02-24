try:
    from ingest_data import get_search_result
    from utils import generate_answer
    from utils import get_args
except:
    from tools.ingest_data import get_search_result
    from tools.utils import generate_answer
    from tools.utils import get_args

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from sentence_transformers import CrossEncoder
import numpy as np
import pandas as pd
from openai import OpenAI


WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'datasets/financebench_data/dataprep/')

model = "BAAI/bge-base-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=model)

# reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
# compressor = CrossEncoderReranker(model=reranker_model, top_n=1)
reranker_model = CrossEncoder("BAAI/bge-reranker-base")

def get_company_mapping():
    company_mapping = {}
    df = pd.read_json(os.path.join(WORKDIR, "financebench/data/financebench_open_source.jsonl"), lines=True)
    company = df["company"].unique().tolist()
    for c in company:
        doc_name = df[df["company"] == c]["doc_name"].unique().tolist()[0]
        company_mapping[c] = doc_name.split("_")[0]
        company_mapping[c.lower()] = doc_name.split("_")[0]
        company_mapping[c.upper()] = doc_name.split("_")[0]
    return company_mapping

COMPANY_MAPPING = get_company_mapping()
# COMPANY_MAPPING["Coca Cola"] = "COCACOLA"
# COMPANY_MAPPING["The Coca-Cola Company"] = "COCACOLA"
# COMPANY_MAPPING["AES"] = "AES"
# COMPANY_MAPPING["The AES Corporation"] = "AES"
# COMPANY_MAPPING["Apple Inc."] = "APPLE"
# COMPANY_MAPPING["Block (formerly Square)"] = "BLOCK"

PROMPT_TEMPLATE="""\
Here is the list of company names in the knowledge base:
{company_list}

This is the company of interest: {company}

Map the company of interest to the company name in the knowledge base. Output the company name in  {{}}. Example: {{3M}}
"""
# def get_company_list():
#     df = pd.read_json(os.path.join(WORKDIR, "financebench/data/financebench_open_source.jsonl"), lines=True)
#     company_list = df["company"].unique().tolist()
#     return company_list

# COMPANY_LIST = get_company_list()

def parse_company_name(response):
    # response {3M}
    try:
        company = response.split("{")[1].split("}")[0]
    except:
        company = ""
    return company



def map_company_with_llm(company):
    prompt = PROMPT_TEMPLATE.format(company_list=COMPANY_LIST, company=company)
    llm_endpoint_url = "http://localhost:8086"
    client = OpenAI(
        base_url=f"{llm_endpoint_url}/v1",
        api_key="token-abc123",
    )

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "user", "content": prompt}
        ]
        )

    # get response
    response = completion.choices[0].message.content
    print("Response: ", response)
    mapped_company = parse_company_name(response) #COMPANY_MAPPING[response]
    print("Mapped company: ", mapped_company)
    return mapped_company


def get_content(searched_doc):
    return searched_doc.metadata["chunk"]

def get_table(searched_doc):
    return searched_doc.metadata["table"]

def rerank_docs(docs, top_n=3):
    doc_content = []
    for doc in docs:
        doc_content.append(doc.page_content)

    cross_scores = reranker_model.predict([[query, doc] for doc in doc_content])

    sorted_cross_scores = sorted(cross_scores, reverse=True)
    print(cross_scores)

    reranked_docs = []
    for score in sorted_cross_scores:
        idx = np.where(cross_scores == score)[0][0]
        print(f"idx: {idx}, score: {score}")
        reranked_docs.append(docs[idx])
    return reranked_docs[:top_n]

def get_context(query, company, year, quarter=None):
    """
    Search the knowledge base for the most relevant document
    """
    k = 3
    top_n = 3
    vector_store = Chroma(
        collection_name="doc_collection",
        embedding_function=embeddings,
        persist_directory=os.path.join(DATAPATH, "test_3M_section_summary"),
    )
    # try:
    #     company = COMPANY_MAPPING[company]
    # except:
    #     print("Using LLM to map company name...")
    #     company = map_company_with_llm(company)
    company = company.upper()
    if company in COMPANY_LIST:
        print(f"Company {company} found in the list")
    else:
        company = map_company_with_llm(company)
        print(f"Mapped to {company}")

    print(f"Searcing for company: {company}, year: {year}, quarter: {quarter}")

    docs = vector_store.similarity_search(query, k=k, filter={"company_year_quarter": f"{company}_{year}_{quarter}"})

    if not docs: # if no relevant document found, relax the filter
        print("No relevant document found with company, year and quarter filter, only search with comany and year")
        docs = vector_store.similarity_search(query, k=k, filter={"company_year_quarter": f"{company}_{year}_"})
        
    if not docs: # if no relevant document found, relax the filter
        print("No relevant document found with company_year filter, only serach with company.....")
        docs = vector_store.similarity_search(query, k=k, filter={"company": f"{company}"})
    
    if not docs: # if no relevant document found, relax the filter
        return "No relevant document found. Change your query and try again."

    # rerank docs
    # docs = rerank_docs(docs, top_n=top_n)

    context = ""
    for i, doc in enumerate(docs):
        # result = get_search_result(doc)
        result = get_content(doc)
        context += f"Doc[{i+1}]:\n{result}\n"
    
    return context

import json
with open(os.path.join(DATAPATH, "table_store.json"), "r") as f:
    table_store = f.readlines()

table_store = [json.loads(ts) for ts in table_store]

# print(len(table_store))
# for ts in table_store:
#     print(ts.keys())

from langchain_community.retrievers import BM25Retriever
def retrieve_tables_with_bm25(tables, query):
    """
    tables: list of tables
    Retrieve tables with BM25
    """
    retriever = BM25Retriever.from_texts(tables, k=3)
    results = retriever.invoke(query)

    table_content = ""
    for i, result in enumerate(results):
        table_content += f"Table {i+1}:\n{result.page_content}\n\n"
    # print(table_content)
    return table_content

def get_tables_with_key(key):
    tables = []
    for ts in table_store:
        if key in ts:
            tables = ts[key]
            break
    return tables


def get_company_list():
    with open(os.path.join(DATAPATH, "company_list.txt"), "r") as f:
        company_list = f.readlines()
    company_list = [c.strip() for c in company_list]
    print("Number of companies: ", len(company_list))
    return company_list

COMPANY_LIST = get_company_list()

def get_tables(query, company, year="", quarter=""):
    """
    Get top3 tables for a company for a given year and quarter
    """
    # try:
    #     company = COMPANY_MAPPING[company]
    # except:
    #     print("Using LLM to map company name...")
    #     company = map_company_with_llm(company)
    company = company.upper()
    if company in COMPANY_LIST:
        print(f"Company {company} found in the list")
    else:
        company = map_company_with_llm(company)
        print(f"Mapped to {company}")


    print(f"Getting tables for company: {company}, year: {year}, quarter: {quarter}")

    ## dense retriever approach
    k = 1
    vector_store = Chroma(
        collection_name="table_collection",
        embedding_function=embeddings,
        persist_directory=os.path.join(DATAPATH, "test_3M_table_store"),
    )

    docs = vector_store.similarity_search(query, k=k, filter={"company_year_quarter": f"{company}_{year}_{quarter}"})

    if not docs: # if no relevant document found, relax the filter
        print("No relevant document found with company, year and quarter filter, only search with comany and year filter")
        docs = vector_store.similarity_search(query, k=k, filter={"company_year_quarter": f"{company}_{year}_"})
        
    if not docs: # if no relevant document found, relax the filter
        print("No relevant document found with company_year filter, only serach with company filter.....")
        docs = vector_store.similarity_search(query, k=k, filter={"company": f"{company}"})
    
    if not docs: # if no relevant document found, relax the filter
        return "No relevant document found. Change your query and try again."
    else:
        context = ""
        for i, doc in enumerate(docs):
            # result = get_search_result(doc)
            result = get_table(doc)
            context += f"Table[{i+1}]:\n{result}\n\n"
        return context

    ### BM25 approach
    # key = f"{company}_{year}{quarter}"      
    
    # tables = get_tables_with_key(key)
    
    # if not tables:
    #     key = f"{company}_{year}"
    #     tables = get_tables_with_key(key)

    # if tables:
    #     return retrieve_tables_with_bm25(tables, query)
    # else:
    #     return "No tables found for the given company, year and quarter"




if __name__ == "__main__":
    # query = "FY2019 balance sheet"
    # result = search_knowledge_base(query, "Nike", "2019", "")
    # print(result)

    # test_cases = ["Block", "AMD"]
    # for tc in test_cases:
    #     company = map_company_with_llm(tc)

    query = "major acquisitions"
    company = "AMCOR"
    year = ["2023", "2022", "2021"]
    for y in year:
        result = get_context(query, company, y, "")
        print(result)
        print("=================")
    # result = get_tables(query, company, "2016", "")
    # print(result)
    # print("=================")
    # result = get_context(query, "3M", "2023", "Q2")
    # print(result)

    # args = get_args()

    # test_cases = ["MGM Resorts International", "MGM"]

    # for tc in test_cases:
    #     mapped_company = map_company_with_llm(args, tc)
    #     print(f"Company: {tc}, mapped company: {mapped_company}")


