from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Redis
from docling.document_converter import DocumentConverter
from openai import OpenAI
import os
import json
import uuid
from tqdm import tqdm
from utils.redis_kv import RedisKVStore

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# TEI Embedding endpoints
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")

REDIS_URL_VECTOR = os.getenv("REDIS_URL_VECTOR", "redis://localhost:6379/")
REDIS_URL_KV = os.getenv("REDIS_URL_KV", "redis://localhost:6380/")
LLM_MODEL=os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
LLM_ENDPOINT=os.getenv("LLM_ENDPOINT", "http://localhost:8086")
MAX_TOKENS = os.getenv("MAX_TOKENS", 1024)
TEMPERATURE = os.getenv("TEMPERATURE", 0.2)


def get_embedder():
    if TEI_EMBEDDING_ENDPOINT:
        # create embeddings using TEI endpoint service
        embedder = HuggingFaceEndpointEmbeddings(model=TEI_EMBEDDING_ENDPOINT)
    else:
        # create embeddings using local embedding model
        embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)
    return embedder


IS_RELEVANT_PROMPT = """Should this chunk be part of the document?
Chunk: {chunk}
Document Title: {doc_title}
Give your answer in Yes or No. No other words.
"""

METADATA_PROMPT="""\
Read the following document and extract the following metadata:
- Company name: only the name of the company, do not include Company, Inc., Corp., etc.
- Year
- Quarter: can be empty
- Document type

Output in json format. Examples: 
```json
{{"company": "Apple", "year": "2020", "quarter": "", "doc_type": "10-K"}}
```
```json
{{"company": "Apple", "year": "2022", "quarter": "Q1", "doc_type": "10-Q"}}
```
```json
{{"company": "Apple", "year": "2024", "quarter": "Q4", "doc_type": "earnings call transcript"}}
```

Here is the document:
{document}

Only output the metadata in json format. Now start!
"""

CHUNK_SUMMARY_PROMPT="""\
You are a financial analyst. You are given a document extracted from a SEC filing. Read the document and summarize it in a few sentences.
Document:
{doc}
Only output your summary.
"""

TABLE_SUMMARY_PROMPT = """\
You are a financial analyst. You are given a table extracted from a SEC filing. Read the table and give it a descriptive title. 
If the table is a financial statement, for example, balance sheet, income statement, cash flow statement, statement of operations, or statement of shareholders' equity, you should specify the type of financial statement in the title.

Table:

{table_md}

Only output the table title.
"""

COMPANY_NAME_PROMPT="""\
Here is the list of company names in the knowledge base:
{company_list}

This is the company of interest: {company}

Determine if the company of interest is the same as any of the companies in the knowledge base. 
If yes, map the company of interest to the company name in the knowledge base. Output the company name in  {{}}. Example: {{3M}}.
If none of the companies in the knowledge base match the company of interest, output "NONE".
"""

def parse_metadata_json(metadata):
    """
    metadata: str
    """
    if "```json" in metadata:
        metadata = metadata.split("```json")[1]
        metadata = metadata.split("```")[0]
    elif "```" in metadata:
        metadata = metadata.split("```")[1]
        metadata = metadata.split("```")[0]
    metadata = metadata.strip()
    try:
        metadata = json.loads(metadata)
        return metadata
    except:
        print("Error in parsing metadata.")
        return {}

def post_process_text(text: str) -> str:
    text = text.replace("## Table of Contents", "")
    text = text.replace("Table of Contents", "")
    return text

def split_text(text: str) -> list:
    chunks = text.split("##")
    return chunks

def generate_answer(prompt):
    """
    Use vllm endpoint to generate the answer
    """
    # send request to vllm endpoint
    client = OpenAI(
        base_url=f"{LLM_ENDPOINT}/v1",
        api_key="token-abc123",
    )

    params = {
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        **params
        )

    # get response
    response = completion.choices[0].message.content

    return response

def generate_metadata_with_llm(full_doc):
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    # get the first 5000 characters
    prompt = METADATA_PROMPT.format(document=full_doc[:10000])
    metadata = generate_answer(prompt)
    print(metadata)
    metadata = parse_metadata_json(metadata)

    title = f"{metadata['company']} {metadata['year']} {metadata['quarter']} {metadata['doc_type']}"
    company_year_quarter = f"{metadata['company']}_{metadata['year']}_{metadata['quarter']}"
    metadata["doc_title"] = title
    metadata["company_year_quarter"] = company_year_quarter
    # metadata["doc_length"] = len(tokenizer.encode(full_doc))

    for k, v in metadata.items():
        if isinstance(v, str):
            metadata[k] = v.upper()

    for k, v in metadata.items():   
        print(f"{k}: {v}")

    return metadata

def save_full_doc(full_doc: str, metadata: dict):
    print("Saving full doc....")
    kvstore = RedisKVStore(redis_uri=REDIS_URL)
    kvstore.put(metadata["doc_title"], {"full_doc": full_doc, **metadata}, "full_docs")
    return None 


def save_company_name(metadata: dict):
    # get existing companies from KV store
    # collection: company_list
    # key: company
    print("Saving company name....")
    kvstore = RedisKVStore(redis_uri=REDIS_URL)
    company_list_dict = kvstore.get("company", "company_list")
    if company_list_dict:
        company_list = company_list_dict["company"]
        print("Found existing company list: ", company_list)
    else:
        print("No existing company list found. Creating new list.")
        company_list = []
        company_list.append(metadata["company"])
        kvstore.put("company", {"company": company_list}, "company_list")
        return metadata

    new_company = metadata["company"]
    # decide if new_company already in company_list
    if new_company in company_list:
        print("Company already in company list.")
        # no need to change metadata["company"]
        pass
    else:
        # use LLM to decide if new_company is alias of existing company
        print("Use LLM to decide if company is alias of existing company.")
        prompt = COMPANY_NAME_PROMPT.format(company_list=company_list, company=new_company)
        response = generate_answer(prompt)
        if "NONE" in response.upper():
            print(f"Company is not in company list. Add {new_company} to company list.")
            # add new_company to company_list
            company_list.append(new_company)
            kvstore.put("company", {"company": company_list}, "company_list")
        else:
            existing_company = response.strip("{}").upper()
            print(f"Company is alias of existing company. Map {new_company} to {existing_company}.")
            metadata["company"] = existing_company

    return metadata

    

def save_doc_title(doc_title: str, metadata: dict):
    print("Saving doc title....")
    embedder = get_embedder()
    _, keys = Redis.from_texts_return_keys(
            texts=[doc_title],
            embedding=embedder,
            index_name="titles",
            redis_url=REDIS_URL,
            metadatas=[metadata],
        )
    return keys

def parse_doc_and_extract_metadata(filename: str):
    # extract pdf or url with docling and convert into markdown full content and tables
    print("Parsing document....")
    converter = DocumentConverter()
    conv_res = converter.convert(filename)

    # convert full doc to markdown
    print("Converting to markdown....")
    full_doc = conv_res.document.export_to_markdown()

    # get doc title and metadata with llm
    print("Extracting metadata....")
    metadata = generate_metadata_with_llm(full_doc)
    return conv_res, full_doc, metadata


def split_markdown_and_summarize_save(text, metadata):
    print("Splitting markdown and processing chunks....")
    text = post_process_text(text)
    chunks = split_text(text)
    print(f"Number of chunks: {len(chunks)}")
    print("Average chunk size: ", sum([len(chunk) for chunk in chunks]) / len(chunks))
    print("Minimum chunk size: ", min([len(chunk) for chunk in chunks]))

    keys = []
    for chunk in tqdm(chunks):
        print("Chunk:\n", chunk[:50])
        print("Length of chunk: ", len(chunk))
        if len(chunk) < 100:
            print("Chunk is too short. Skipping summarization.")
            key = save_chunk((chunk, chunk), metadata)
            keys.append(key)
            continue
        print("Summarizing chunk........")
        prompt = CHUNK_SUMMARY_PROMPT.format(doc=chunk)
        summary = generate_answer(prompt)
        print("Summary of chunk:\n", summary)
        key = save_chunk((chunk, summary), metadata)
        keys.extend(key)
        print("="*50)
    return keys


def save_chunk(chunk: tuple, metadata: dict, doc_type="chunks"):
    # chunk: tuple of (content, summary)
    print(f"Saving {doc_type}....")
    assert doc_type in ["chunks", "tables"], "doc_type should be either chunks or tables"
    embedder = get_embedder()

    chunk_content = chunk[0]
    chunk_summary = chunk[1]

    doc_id = str(uuid.uuid4())
    metadata["doc_id"] = doc_id
    print(f"Chunk metadata: {metadata}")
    # embed summary and save to vector db
    _, key = Redis.from_texts_return_keys(
            texts=[chunk_summary],
            embedding=embedder,
            index_name=doc_type,
            redis_url=REDIS_URL,
            metadatas=[metadata],
        )
    
    # save chunk_content to kvstore
    kvstore = RedisKVStore(redis_uri=REDIS_URL)
    kvstore.put(doc_id, {"content": chunk_content, "summary":chunk_summary,**metadata}, doc_type)
    return key

def get_table_summary_with_llm(table_md):
    """
    table_md: str
    args: including llm_endpoint_url and model 
    """
    prompt = TABLE_SUMMARY_PROMPT.format(table_md=table_md)
    table_summary = generate_answer(prompt)
    print(f"Table summary:\n{table_summary}")
    return table_summary

def process_tables(conv_res, metadata):
    print("Processing tables....")
    keys = []
    for table_ix, table in enumerate(conv_res.document.tables):
        table_md = table.export_to_markdown()
        context = get_table_summary_with_llm(table_md)
        key = save_chunk((table_md, context), metadata, doc_type="tables")
        keys.extend(key)
    return keys 

def post_process_html(full_doc, doc_title):
    print("Post processing extracted webpage....")
    final_doc = ""
    chunks = split_text(full_doc)
    for chunk in chunks:
        prompt = IS_RELEVANT_PROMPT.format(chunk=chunk, doc_title=doc_title)
        llm_res = generate_answer(prompt)
        print(f"Chunk: {chunk[:100]}...")
        print(f"LLM Response: {llm_res}")
        if "YES" in llm_res.upper():
            final_doc += f"##{chunk}"
    return final_doc

def ingest_financial_data(filename: str):
    """
    4 collections:
    chunks
    tables
    doc_title
    full_doc
    """
    file_ids = []

    if filename.endswith(".pdf") or "https://" in filename or filename.endswith(".md"):
        if not filename.endswith(".md"):
            conv_res, full_doc, metadata = parse_doc_and_extract_metadata(filename)

        if "https://" in filename:
            full_doc = post_process_html(full_doc, metadata["doc_title"])

        # save company name
        metadata = save_company_name(metadata)

        # save full doc
        save_full_doc(full_doc, metadata)

        # save doc_title
        doc_title = metadata["doc_title"]
        keys = save_doc_title(doc_title, metadata)
        file_ids.extend(keys)
      
        # chunk and save
        keys = split_markdown_and_summarize_save(full_doc, metadata)
        file_ids.extend(keys)

        # process tables and save
        keys = process_tables(conv_res, metadata)
        file_ids.extend(keys)
    else:
        raise ValueError("File format not supported.")

################ retrieval functions ####################
from langchain_community.retrievers import BM25Retriever

def get_company_list():
    kvstore = RedisKVStore(redis_uri=REDIS_URL)
    company_list_dict = kvstore.get("company", "company_list")
    if company_list_dict:
        company_list = company_list_dict["company"]
        return company_list
    else:
        return []
    
def get_company_name_in_kb(company, company_list):
    if not company_list:
        return "Database is empty."
    prompt = COMPANY_NAME_PROMPT.format(company_list=company_list, company=company)
    response = generate_answer(prompt)
    if "NONE" in response.upper():
        return "Cannot find company name in knowledge base."
    else:
        return response.strip("{}").upper()

def get_docs_matching_metadata(metadata, doc_type="chunks"):
    """
    metadata: ("company_year", "3M_2023")
    docs: list of documents
    """
    key = metadata[0]
    value = metadata[1]
    kvstore= RedisKVStore(redis_uri=REDIS_URL)
    collection = kvstore.get_all(doc_type) # collection is a dict

    matching_docs = []
    for idx in collection:
        doc = collection[idx]
        if doc[key] == value:
            print(f"Found doc with matching metadata {metadata}")
            print(doc["doc_title"])
            matching_docs.append(doc)
    print(f"Number of docs found with metadata {metadata}: {len(matching_docs)}")
    return matching_docs
    
   

def convert_docs(docs):
    text = []
    for doc in docs:
        content = doc["content"]
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
#        print(f"Number of docs found with BM25 over content: {len(docs_bm25)}")

        # BM25 search over summary/title
        retriever = BM25Retriever.from_documents(docs, k=k)
        docs_bm25_title = retriever.invoke(query)
#        print(f"Number of docs found with BM25 over title: {len(docs_bm25_title)}")
        results = docs_bm25 + docs_bm25_title
    else:
        results = []
    return results


def bm25_search_broad(query, company, year, quarter, vector_store, k=10, doc_type="chunk"):
    # search with company filter, but query is query_company_quarter
    metadata = ("company",f"{company}")
    query1 = f"{query} {year} {quarter}"
    docs1 = bm25_search(query1, metadata, vector_store, k=k, doc_type=doc_type)

    # search with metadata filters
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

    docs = docs + docs1
    if docs:
        return docs
    else:
        return []


def get_context_bm25_llm(query, company, year, quarter = ""):
    k = 5
    
    company_list = get_company_list()
    company = get_company_name_in_kb(company, company_list)
    if "Cannot find company name" in company or "Database is empty" in company:
        return company
    
    embedder = get_embedder()

    # chunks
    vector_store = Redis(embedding=embedder, index_name="chunks", redis_url=REDIS_URL)
    chunks_bm25 = bm25_search_broad(query, company, year, quarter, vector_store, k=k, doc_type="chunk")
    chunks_sim = similarity_search(vector_store, k, query, company, year, quarter)
    chunks = chunks_bm25 + chunks_sim
    
    # tables
    vector_store_table = Redis(embedding=embedder, index_name="tables", redis_url=REDIS_URL)

    # get tables matching metadata
    tables_bm25 = bm25_search_broad(query, company, year, quarter, vector_store_table, k=k, doc_type="table")
    tables_sim = similarity_search(vector_store_table, k, query, company, year, quarter)
    tables = tables_bm25 + tables_sim

    # get unique results
    context = get_unique_docs(chunks+tables)
    # print("Context:\n", context)

    if context:
        query = f"{query} for {company} in {year} {quarter}"
        prompt = ANSWER_PROMPT.format(query=query, documents=context)
        response = generate_answer(prompt)
        response = parse_response(response)
    else:
        response = f"No relevant information found for {company} in {year} {quarter}."
    return response

if __name__ == "__main__":
    # # data ingestion
    # WORKDIR = os.getenv("WORKDIR")
    # DATAPATH= os.path.join(WORKDIR, "datasets/financebench/dataprep")
    # # files = ["3M_2018_10K-table-7.md"]
    # files = ["https://www.fool.com/earnings/call-transcripts/2025/03/04/progressive-pgr-q4-2024-earnings-call-transcript/"]
    # for f in files:
    #     print("Ingesting file:", f)
    #     if "https://" in f:
    #         ingest_financial_data(f)
    #     else:
    #         if not os.path.exists(os.path.join(DATAPATH, f)):
    #             print("File not found:", f)
    #             continue
    #         ingest_financial_data(os.path.join(DATAPATH, f))
    #     print("="*50)

    # # retrieval
    get_docs_matching_metadata(("company", "PROGRESSIVE"), "chunks")
    # from langchain_community.vectorstores.redis import RedisText
    # embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)
    # index_name="tables"
    # vector_store = Redis(embedding=embedder, index_name=index_name, redis_url=REDIS_URL)


    # query = "3M revenue in 2018"
    # results = vector_store.similarity_search(query, k=2, filter=(RedisText("company") == "3M"))

    # print("Simple Similarity Search Results:")
    # for doc in results:
    #     print(f"Length of chunk {len(doc.page_content)}")
    #     print(f"Content: {doc.page_content[:200]}...")
    #     print(f"Metadata: {doc.metadata}")
