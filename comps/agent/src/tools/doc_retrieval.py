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

PROMPT_TEMPLATE="""\
Here is the list of company names in the knowledge base:
{company_list}

This is the company of interest: {company}

Map the company of interest to the company name in the knowledge base. Output the company name in  {{}}. Example: {{3M}}
"""

def parse_company_name(response):
    # response {3M}
    try:
        company = response.split("{")[1].split("}")[0]
    except:
        company = ""
    return company

def generate_answer_with_llm(prompt):
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
    return response

def map_company_with_llm(company):
    prompt = PROMPT_TEMPLATE.format(company_list=COMPANY_LIST, company=company)
    response = generate_answer_with_llm(prompt)
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


def get_docs_matching_metadata(metadata, docs):
    """
    metadata: ("company_year", "3M_2023")
    docs: list of documents
    """
    print(f"Searching for docs with metadata: {metadata}")
    matching_docs = []
    key = metadata[0]
    value = metadata[1]

    for doc in docs:
        if doc.metadata[key] == value:
            matching_docs.append(doc)
    print(f"Number of matching docs: {len(matching_docs)}")
    return matching_docs


DOC_GRADER_PROMPT = """\
Search Query: {query}.
Documents:
{documents}

Read the documents and decide if any useful infomation can be extracted from them regarding the Search Query. 
If yes, output the useful information in {{}}. Include financial figures when available. Example: {{The company has a revenue of $100 million.}}
If no, output "No relevant information found."
"""

def convert_docs_to_text(docs):
    # doc content is saved in metadata["chunk"]
    text = []
    for doc in docs:
        text.append(doc.metadata["chunk"])
    return text

def get_context_bm25_llm(query, company, year, quarter=None):
    vector_store = Chroma(
        collection_name="doc_collection",
        embedding_function=embeddings,
        persist_directory=os.path.join(DATAPATH, "test_cocacola_v7"),
    )
    collection = vector_store.get()
    id_list = collection['ids']
    all_docs = vector_store.get_by_ids(id_list)

    metadata = ("company_year_quarter",f"{company}_{year}_{quarter}")
    docs = get_docs_matching_metadata(metadata, all_docs)
    if not docs:
        metadata = ("company_year",f"{company}_{year}")
        docs = get_docs_matching_metadata(metadata, all_docs)
    if not docs:
        metadata = ("company",f"{company}")
        docs = get_docs_matching_metadata(metadata, all_docs)
    
    if not docs:
        return "No relevant document found. Change your query and try again."
    
    print(f"Number of docs found matching metadata: {len(docs)}")
    # use BM25 to narrow down
    docs_text = convert_docs_to_text(docs)
    k = 10
    retriever = BM25Retriever.from_texts(docs_text, k=k)
    results = retriever.invoke(query)
    print(f"Number of docs found with BM25: {len(results)}")

    # similarity search
    docs_sim = similarity_search(vector_store, k, query, company, year, quarter)
    docs_sim_text = convert_docs_to_text(docs_sim)
    results = results + docs_sim_text

    # get unique results
    results = list(set(results))
    print(f"Number of unique docs found: {len(results)}")

    # use LLM to judge if there is any useful information
    context = ""
    for i, doc in enumerate(results):
        result = get_content(doc)
        context += f"Doc[{i+1}]:\n{result}\n"
    
    prompt = DOC_GRADER_PROMPT.format(query=query, documents=context)
    response = generate_answer_with_llm(prompt)
    print("Doc grader LLM response: ", response)
    return response

def similarity_search(vector_store, k, query, company, year, quarter=None):
    docs = vector_store.similarity_search(query, k=k, filter={"company_year_quarter": f"{company}_{year}_{quarter}"})

    if not docs: # if no relevant document found, relax the filter
        print("No relevant document found with company, year and quarter filter, only search with comany and year")
        docs = vector_store.similarity_search(query, k=k, filter={"company_year": f"{company}_{year}"})
        
    if not docs: # if no relevant document found, relax the filter
        print("No relevant document found with company_year filter, only serach with company.....")
        docs = vector_store.similarity_search(query, k=k, filter={"company": f"{company}"})
    
    if not docs:
        return None
    else:
        return docs


def get_context(query, company, year, quarter=None):
    """
    Search the knowledge base for the most relevant document
    """
    k = 3
    top_n = 2
    vector_store = Chroma(
        collection_name="doc_collection",
        embedding_function=embeddings,
        persist_directory=os.path.join(DATAPATH, "test_cocacola_v7"),
    )

    company = company.upper()
    if company in COMPANY_LIST:
        print(f"Company {company} found in the list")
    else:
        company = map_company_with_llm(company)
        print(f"Mapped to {company}")

    print(f"Searcing for company: {company}, year: {year}, quarter: {quarter}")

    docs = similarity_search(vector_store, k, query, company, year, quarter)
    
    if not docs: # if no relevant document found, relax the filter
        return "No relevant document found. Change your query and try again."

    # rerank docs
    #docs = rerank_docs(docs, top_n=top_n)

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
    with open(os.path.join(DATAPATH, "new_company_list.txt"), "r") as f:
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
        persist_directory=os.path.join(DATAPATH, "test_cocacola_v7"),
    )

    docs = vector_store.similarity_search(query, k=k, filter={"company_year_quarter": f"{company}_{year}_{quarter}"})

    if not docs: # if no relevant document found, relax the filter
        print("No relevant document found with company, year and quarter filter, only search with comany and year filter")
        docs = vector_store.similarity_search(query, k=k, filter={"company_year": f"{company}_{year}"})
        
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
    query = "debt securities traded on national exchanges"
    # result = get_tables(query, "3M", "2023", "Q2")
    # print("="*50)
    result = get_context_bm25_llm(query, "3M", "2023", "Q2")
    print(result)
    print("="*50)

    query = "key agenda of 8K filing"
    result = get_context_bm25_llm(query, "AMCOR", "2022", "Q2")
    print(result)

    # test_cases = ["Block", "AMD"]
    # for tc in test_cases:
    #     company = map_company_with_llm(tc)

    #query = "major acquisitions"
    #company = "AMCOR"
    #year = ["2023", "2022", "2021"]
    #for y in year:
    #    result = get_context(query, company, y, "")
    #    print(result)
    #    print("=================")
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


