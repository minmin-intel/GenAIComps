from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Redis
from docling.document_converter import DocumentConverter
from openai import OpenAI
import os
import json

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# TEI Embedding endpoints
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/")
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
    embedder = get_embedder()
    _, keys = Redis.from_texts_return_keys(
            texts=[full_doc],
            embedding=embedder,
            index_name="full_doc",
            redis_url=REDIS_URL,
            metadatas=[metadata],
        )
    return keys

def ingest_financial_data(filename: str):
    """
    4 collections:
    chunks
    tables
    doc_title
    full_doc
    """
    # create embeddings using local embedding model
    embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    file_ids = []

    if filename.endswith(".pdf"):
        converter = DocumentConverter()
        conv_res = converter.convert(filename)
        # convert full doc to markdown
        full_doc = conv_res.document.export_to_markdown()
        # get doc title and metadata with llm

        # process tables
        if conv_res.document.tables:
            pass
        pass 
    elif "https://" in filename:
        # extract pdf or url with docling and convert into markdown full content and tables
        converter = DocumentConverter()
        conv_res = converter.convert(filename)

        # convert full doc to markdown
        full_doc = conv_res.document.export_to_markdown()

        # get doc title and metadata with llm
        metadata = generate_metadata_with_llm(full_doc)
        doc_title = metadata["doc_title"]
        print(doc_title)

        # process markdown full content
        # # split by "##" and for each chunk, use llm to decide is it is related to doc_title.
        # # if it is, keep it. otherwise, discard it.
        final_doc = ""
        chunks = split_text(full_doc)
        for chunk in chunks:
            prompt = IS_RELEVANT_PROMPT.format(chunk=chunk, doc_title=doc_title)
            llm_res = generate_answer(prompt)
            print(f"Chunk: {chunk[:100]}...")
            print(f"LLM Response: {llm_res}")
            if "YES" in llm_res.upper():
                final_doc += chunk

        # save full content
        # output_file=os.path.join(DATAPATH, "ect_full_doc_processed.md")
        # with open(output_file, "w") as f:
        #     f.write(final_doc)
        save_full_doc(final_doc, metadata)
    elif filename.endswith(".md"):
        with open(filename, "r") as f:
            content = f.read()
        # get doc title and metadata with llm

        print(len(content))

        # save full content
        print("Saving full doc....")
        metadata = {"doc_title": "3M_2018_10K", "company": "3M", "year": "2018"}
        keys = save_full_doc(content, metadata)
        file_ids.extend(keys)
        print("Saved full doc....length of file_ids:", len(file_ids))

        # process markfown full content
        if "table" in filename:
            print("Saving tables....")
            # process markdown tables
            _, keys = Redis.from_texts_return_keys(
                texts=[content],
                embedding=embedder,
                index_name="tables",
                redis_url=REDIS_URL,
                metadatas=[{"doc_title": "3M_2018_10K", "company": "3M", "year": "2018"}],
            )
            file_ids.extend(keys)
            print("Saved tables....length of file_ids:", len(file_ids))
        else:
            print("Splitting into chunks....")
            chunks = content.split("##")
            chunks = [c for c in chunks if len(c) > 100]
            print("Saving chunks....")
            _, keys = Redis.from_texts_return_keys(
                texts=chunks,
                embedding=embedder,
                index_name="chunks",
                redis_url=REDIS_URL,
                metadatas=[{"doc_title": "3M_2018_10K", "company": "3M", "year": "2018"}]*len(chunks),
            )
            file_ids.extend(keys)
            print("Saved chunks....length of file_ids:", len(file_ids))
        pass
    else:
        # get doc title and metadata with llm

        # save full content
        pass



if __name__ == "__main__":
    # data ingestion
    WORKDIR = os.getenv("WORKDIR")
    DATAPATH= os.path.join(WORKDIR, "datasets/financebench/dataprep")
    # files = ["3M_2018_10K-table-7.md"]
    files = ["https://www.fool.com/earnings/call-transcripts/2025/03/04/progressive-pgr-q4-2024-earnings-call-transcript/"]
    for f in files:
        print("Ingesting file:", f)
        if "https://" in f:
            ingest_financial_data(f)
        else:
            if not os.path.exists(os.path.join(DATAPATH, f)):
                print("File not found:", f)
                continue
            ingest_financial_data(os.path.join(DATAPATH, f))
        print("="*50)

    # # retrieval
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
