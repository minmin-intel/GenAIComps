from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from ingest_data import generate_metadata_with_llm
from utils import get_args

def get_doc_ids_from_vectorstore(vector_store):
    results = vector_store.get()
    # print(results)
    id_list = results['ids']
    return id_list

def update_metadata_per_doc(vector_store, id, last_doc_name, last_metadata):
    doc = vector_store.get_by_ids([id])[0]
    doc_name = doc.metadata["doc_name"]
    print("Old metadata:", doc.metadata["company_year"])

    if doc_name != last_doc_name:
        # generate metadata with llm
        print(f"Generating metadata for {doc_name}........")
        doc_filepath = os.path.join(DATAPATH, doc_name+".md")
        with open(doc_filepath, "r") as f:
            full_doc = f.read()

        metadata = generate_metadata_with_llm(args, full_doc)
        if "table" in doc.metadata:
            print("Table exists in metadata")
            table = doc.metadata["table"]
            metadata["table"] = table
        print("Metadata generated for ", doc_name)
        print("New metadata:\n", metadata)
    else:
        print("Metadata remains the same for ", doc_name)
        metadata = last_metadata
        if "table" in doc.metadata:
            print("Table exists in metadata")
            table = doc.metadata["table"]
            metadata["table"] = table
        print("Metadata:\n", metadata)

    print("Updating metadata........")
    updated_document = Document(
        page_content=doc.page_content,
        metadata=metadata,
        id=id,
    )
    vector_store.update_document(document_id=id, document=updated_document)
    return vector_store, metadata, doc_name



WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'datasets/financebench/dataprep/')

if __name__ == "__main__":
    args = get_args()

    model = "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model)

    vector_store = Chroma(
            collection_name=args.db_collection,
            embedding_function=embeddings,
            persist_directory=os.path.join(DATAPATH, args.db_name),
        )
    
    id_list = get_doc_ids_from_vectorstore(vector_store)
    print("Number of documents to update:", len(id_list))

    if os.path.exists(os.path.join(DATAPATH, "company_list.txt")):
        with open(os.path.join(DATAPATH, "company_list.txt"), "r") as f:
            company_list = f.readlines()
        company_list = [c.strip() for c in company_list]
    else:
        company_list = []

    original_num_company = len(company_list)
    print(len(company_list))

    last_doc_name = ""
    metadata = {}
    for id in id_list:
        vector_store, metadata, last_doc_name = update_metadata_per_doc(vector_store, id, last_doc_name, metadata)
        company = metadata["company"]
        if company not in company_list:
            company_list.append(company)
        print("Checking updated metadata........")
        doc = vector_store.get_by_ids([id])[0]
        print(doc.metadata)
        print("="*50)

    # save company list
    if len(company_list) > original_num_company:
        print("Number of unique companies:", len(company_list))
        with open(os.path.join(DATAPATH, "company_list.txt"), "w") as f:
            for company in company_list:
                f.write(company + "\n")
    else:
        print("No new company added to the list")



