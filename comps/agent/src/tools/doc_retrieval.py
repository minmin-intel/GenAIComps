try:
    from ingest_data import get_search_result
except:
    from tools.ingest_data import get_search_result
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from sentence_transformers import CrossEncoder
import numpy as np


WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'datasets/financebench/dataprep/')

model = "BAAI/bge-base-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=model)

# reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
# compressor = CrossEncoderReranker(model=reranker_model, top_n=1)
reranker_model = CrossEncoder("BAAI/bge-reranker-base")

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
        persist_directory=os.path.join(DATAPATH, "test_chroma_db"),
    )
    
    # retriever = vector_store.as_retriever(
    #     search_type="mmr", search_kwargs={"k": 1, "fetch_k": 50},
    # )
    docs = vector_store.similarity_search(query, k=k, filter={"company_year": f"{company}_{year}{quarter}"})

    if not docs: # if no relevant document found, relax the filter
        print("No relevant document found with year and quarter filter, only search with comany and year filter")
        docs = vector_store.similarity_search(query, k=k, filter={"company_year": f"{company}_{year}"})
        
    if not docs: # if no relevant document found, relax the filter
        print("No relevant document found with year filter, only serach with company filter.....")
        docs = vector_store.similarity_search(query, k=k, filter={"company_year": f"{company}"})


    # rerank
    # docs = rerank_docs(docs, top_n=top_n)

    context = ""
    for i, doc in enumerate(docs):
        result = get_search_result(doc)
        context += f"Doc[{i+1}]:\n{result}\n"
    return context

if __name__ == "__main__":
    query = "3M balance sheet PPNE"
    result = search_knowledge_base(query, "3M", "2018", "Q4")
    print(result)


