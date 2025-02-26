try:
    from ingest_data import get_search_result
    from utils import generate_answer
    from utils import get_args
    from utils import get_test_data
except:
    from tools.ingest_data import get_search_result
    from tools.utils import generate_answer
    from tools.utils import get_args
    from tools.utils import get_test_data

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

def parse_response(response):
    # response {3M}
    if "{" in response:
        company = response.split("{")[1].split("}")[0]
    else:
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
    mapped_company = parse_response(response) #COMPANY_MAPPING[response]
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


def get_docs_matching_metadata(metadata, vector_store):
    """
    metadata: ("company_year", "3M_2023")
    docs: list of documents
    """
    collection = vector_store.get()
    id_list = collection['ids']
    all_docs = vector_store.get_by_ids(id_list)

    print(f"Searching for docs with metadata: {metadata}")
    matching_docs = []
    key = metadata[0]
    value = metadata[1]

    for doc in all_docs:
        if doc.metadata[key] == value:
            matching_docs.append(doc)
    print(f"Number of matching docs: {len(matching_docs)}")
    return matching_docs


DOC_GRADER_PROMPT = """\
Search Query: {query}.
Documents:
{documents}

Read the documents and decide if any relevant infomation can be extracted from them regarding the Search Query. 
If yes, output the relevant information. Include financial figures when available. Be concise. Output the extracted info in {{}} Example: {{The company has a revenue of $100 million.}}
If no, output "No relevant information found."
"""

ANSWER_PROMPT = """\
You are a financial analyst. Read the documents below and answer the question.
Documents:
{documents}

Question: {query}
Now take a deep breath and think step by step to answer the question. Wrap your final answer in {{}}. Example: {{The company has a revenue of $100 million.}}
"""

from langchain_core.documents import Document
def convert_docs(docs, doc_type="chunk"):
    text = []
    for doc in docs:
        # doc_title = doc.metadata["doc_title"]
        if doc_type=="chunk":
            content = doc.metadata["chunk"]
        elif doc_type=="table":
            content = doc.metadata["table"]
        else:
            content = doc.page_content
        
        converted = Document(
                page_content=content,
                metadata=doc.metadata
            )
        text.append(converted)

    return text


def bm25_search(query, metadata, vector_store, k=10, doc_type="chunk"):
    docs = get_docs_matching_metadata(metadata, vector_store)

    if docs:
        # BM25 search over content
        docs_text = convert_docs(docs, doc_type=doc_type)
        retriever = BM25Retriever.from_documents(docs_text, k=k)
        docs_bm25 = retriever.invoke(query)
        print(f"Number of docs found with BM25 over content: {len(docs_bm25)}")

        # BM25 search over summary/title
        retriever = BM25Retriever.from_documents(docs, k=k)
        docs_bm25_title = retriever.invoke(query)
        print(f"Number of docs found with BM25 over title: {len(docs_bm25_title)}")
        results = docs_bm25 + docs_bm25_title
    else:
        results = []
    return results


def bm25_search_broad(query, company, year, quarter, vector_store, k=10, doc_type="chunk"):
    metadata = ("company_year_quarter",f"{company}_{year}_{quarter}")
    print(f"BM25: Searching for docs with metadata: {metadata}")
    docs = bm25_search(query, metadata, vector_store, k=k, doc_type=doc_type)
    if not docs:
        print("BM25: No docs found with company, year and quarter filter, only search with company and year filter")
        metadata = ("company_year",f"{company}_{year}")
        docs = bm25_search(query, metadata, vector_store, k=k, doc_type=doc_type)
    if not docs:
        print("BM25: No docs found with company and year filter, only search with company filter")
        metadata = ("company",f"{company}")
        docs = bm25_search(query, metadata, vector_store, k=k, doc_type=doc_type)
    if docs:
        return docs
    else:
        return []


def hybrid_search(query, metadata, vector_store, k=10, doc_type="chunks"):
    # bm25 search
    chunks_bm25 = bm25_search(query, metadata, vector_store, k=k, doc_type=doc_type)
    print(f"Number of docs found with BM25 search: {len(chunks_bm25)}")

    # similarity search over summary/title
    chunks_sim = similarity_search(vector_store, k, query, company, "", "")
    print(f"Number of docs found with similarity search: {len(chunks_sim)}")
    chunks = chunks_bm25 + chunks_sim
    print(f"Total number of chunks found: {len(chunks)}")
    return chunks


def get_unique_docs(docs):
    results = []
    context = ""
    i = 1
    for doc in docs:
        if "chunk" in doc.metadata:
            content = doc.metadata["chunk"]
        elif "table" in doc.metadata:
            content = doc.metadata["table"]
        else:
            content = doc.page_content
        if content not in results:
            results.append(content)
            doc_title = doc.metadata["doc_title"]
            ret_doc = f"Doc [{i}] from {doc_title}:\n{content}\n"
            context += ret_doc
            i += 1
    print(f"Number of unique docs found: {len(results)}")
    return context

METADATA_PROMPT = """
Given the query, decide which company, year and quarter the query is related to. 
** Procedure: **
1. If the query is ambiguous, and you cannot determine the company from it, output "Need clarification."
2. If the query asks about multiple companies, output "Need clarification."
3. If the query asks about multiple years, leave the year blank.
4. If the query asks about multiple quarters, leave the quarter blank.
5. If the query only asks about year but not quarter, leave the quarter blank.
6. If the query did not mention any time information, leave the year and quarter blank.
7. If you can decide the company, year and quarter, output the metadata in json format.

Examples: 
Query: What is the revenue of 3M in Q2 of 2023?
Output:
```json
{"company": "3M", "year": "2023", "quarter": "Q2"}
```

Query: What is the revenue of 3M in 2023?
Output:
```json
{"company": "3M", "year": "2023"}
```

Query: What is the revenue of 3M?
Output:
```json
{"company": "3M"}
```

Query: What is the revenue in 2023?
Output: Need clarification.

"""


def get_context_bm25_llm(query, company, year, quarter = ""):
    k = 10
    vector_store = Chroma(
        collection_name="doc_collection",
        embedding_function=embeddings,
        persist_directory=os.path.join(DATAPATH, "test_cocacola_v7"),
    )

    company = get_company_name_in_kb(company)
    print(company)

    chunks_bm25 = bm25_search_broad(query, company, year, quarter, vector_store, k=k, doc_type="chunk")
    chunks_sim = similarity_search(vector_store, k, query, company, year, quarter)
    chunks = chunks_bm25 + chunks_sim
    # get text chunks for the company
    # chunks = hybrid_search(query, company, vector_store, k, doc_type="chunk")
    
    # # tables
    # vector_store_table = Chroma(
    #     collection_name="table_collection",
    #     embedding_function=embeddings,
    #     persist_directory=os.path.join(DATAPATH, "test_cocacola_v7"),
    # )

    # # get tables matching metadata
    # tables = hybrid_search(query, company, vector_store_table, k, doc_type="table")

    # get unique results
    context = get_unique_docs(chunks)
    print("Context:\n", context)

    if context:
        query = f"{query} for {company} in {year} {quarter}"
        prompt = ANSWER_PROMPT.format(query=query, documents=context)
        response = generate_answer_with_llm(prompt)
        response = parse_response(response)
    else:
        response = f"No relevant information found for {company} in {year} {quarter}."

    print("Response: ", response)
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
        return []
    else:
        return docs

def get_company_name_in_kb(company):
    # TODO: deal with cases where company not in kb.
    company = company.upper()
    if company in COMPANY_LIST:
        print(f"Company {company} found in the list")
    else:
        company = map_company_with_llm(company)
        print(f"Mapped to {company}")
    return company

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

    company = get_company_name_in_kb(company)

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
    company = company.upper()
    if company in COMPANY_LIST:
        print(f"Company {company} found in the list")
    else:
        company = map_company_with_llm(company)
        print(f"Mapped to {company}")


    print(f"Getting tables for company: {company}, year: {year}, quarter: {quarter}")

    ## dense retriever approach
    k = 3
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    if args.debug:
        # query = "Which debt securities are registered to trade on a national securities exchange under 3M's name as of Q2 of 2023?"
        query = "key agenda of 8K filing"
        company = "Amcor"
        year = "2022"
        quarter = "Q2"
        result = get_context_bm25_llm(query, company, year, quarter)
        print(result)
        print("="*50)

        # query = "key agenda of 8K filing"
        # result = get_context_bm25_llm(query, "AMCOR")
        # print(result)
    else:
        WORKDIR=os.getenv('WORKDIR')
        filename = os.path.join(WORKDIR, "financebench/data/financebench_open_source.jsonl")
        df = pd.read_json(filename, lines=True)
        output_list = []
        for _, row in df.iterrows():
            query = row["question"]
            company = row["company"]
            # map to company name in the knowledge base
            company = company.upper()
            if company.upper() in COMPANY_LIST:
                print(f"Company {company} found in the list")
            else:
                company = map_company_with_llm(company)
                print(f"Mapped to {company}")
                
            year = ""
            quarter = ""

            print(f"Query: {query}")
            print(f"Company: {company}")
            result = get_context_bm25_llm(query, company)
            output_list.append(result)

            with open(os.path.join(WORKDIR, "datasets/financebench/results/bm25_dense_llm.jsonl"), "a") as f:
                f.write(json.dumps({"question": query, "company": company, "answer": result}) + "\n")

        df["response"] = output_list
        df.to_csv(os.path.join(WORKDIR, "datasets/financebench/results/bm25_dense_llm_v2.csv"), index=False)

    # df_ref = pd.read_csv(os.path.join(WORKDIR, "datasets/financebench/results/finqa_agent_v7_all_t0p5_graded.csv"))
    # df.rename(columns={"answer": "response"}, inplace=True)
    # df["answer"] = df_ref["answer"]





