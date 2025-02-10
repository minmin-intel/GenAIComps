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


WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'datasets/financebench/dataprep/')

model = "BAAI/bge-base-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=model)

reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=1)


def search_knowledge_base(query, company, year):
    """
    Search the knowledge base for the most relevant document
    """
    vector_store = Chroma(
        collection_name="single_doc_collection",
        embedding_function=embeddings,
        persist_directory=os.path.join(DATAPATH, "test_chroma_db"),
    )
    
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50}, filter={"company_year": f"{company}_{year}"}
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    
    docs = compression_retriever.invoke(query)

    context = ""
    for i, doc in enumerate(docs):
        result = get_search_result(doc)
        context += f"{result}\n"
    return context

if __name__ == "__main__":
    query = "revenue of 3M in 2018"
    result = search_knowledge_base(query, "3M", "2018")
    print(result)


