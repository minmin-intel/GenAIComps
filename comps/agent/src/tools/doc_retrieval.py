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
DATAPATH=os.path.join(WORKDIR, 'datasets/financebench/dataprep/')

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

Map the company of interest to the company name in the knowledge base. Only output the company name.

Your answer:
"""
def get_company_list():
    df = pd.read_json(os.path.join(WORKDIR, "financebench/data/financebench_open_source.jsonl"), lines=True)
    company_list = df["company"].unique().tolist()
    return company_list

COMPANY_LIST = get_company_list()

def map_company_with_llm(company):
    prompt = PROMPT_TEMPLATE.format(company_list=COMPANY_LIST, company=company)
    llm_endpoint_url = "http://localhost:8085"
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
    mapped_company = COMPANY_MAPPING[response]
    # print("Mapped company: ", mapped_company)
    return mapped_company



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

def search_knowledge_base(query, company, year, quarter=None):
    """
    Search the knowledge base for the most relevant document
    """
    k = 3
    top_n = 3
    vector_store = Chroma(
        collection_name="doc_collection",
        embedding_function=embeddings,
        persist_directory=os.path.join(DATAPATH, "all_docs_chroma_db"),
    )
    
    # retriever = vector_store.as_retriever(
    #     search_type="mmr", search_kwargs={"k": 1, "fetch_k": 50},
    # )
    try:
        company = COMPANY_MAPPING[company]
    except:
        print("Using LLM to map company name...")
        company = map_company_with_llm(company)

    print(f"Searcing for company: {company}, year: {year}, quarter: {quarter}")

    docs = vector_store.similarity_search(query, k=k, filter={"company_year": f"{company}_{year}{quarter}"})

    if not docs: # if no relevant document found, relax the filter
        print("No relevant document found with company, year and quarter filter, only search with comany and year filter")
        docs = vector_store.similarity_search(query, k=k, filter={"company_year": f"{company}_{year}"})
        
    if not docs: # if no relevant document found, relax the filter
        print("No relevant document found with year filter, only serach with company filter.....")
        docs = vector_store.similarity_search(query, k=k, filter={"company_year": f"{company}"})
    
    if not docs: # if no relevant document found, relax the filter
        return "No relevant document found. Change your query and try again."

    # rerank docs
    # docs = rerank_docs(docs, top_n=top_n)

    context = ""
    for i, doc in enumerate(docs):
        result = get_search_result(doc)
        context += f"Doc[{i+1}]:\n{result}\n"
    
    return context

if __name__ == "__main__":
    # query = "FY2019 balance sheet"
    # result = search_knowledge_base(query, "Nike", "2019", "")
    # print(result)

    args = get_args()

    test_cases = ["MGM Resorts International", "MGM"]

    for tc in test_cases:
        mapped_company = map_company_with_llm(args, tc)
        print(f"Company: {tc}, mapped company: {mapped_company}")


