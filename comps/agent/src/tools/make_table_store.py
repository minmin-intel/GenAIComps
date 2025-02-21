import json
import os
import glob
from ingest_data import process_pdf_docling
from utils import get_doc_path, get_test_data, get_args
from docling.document_converter import DocumentConverter

WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'datasets/financebench/dataprep/')
TABLESTORE=os.path.join(DATAPATH, "table_store.json")

def extract_and_save_tables_from_pdf(doc_converter, doc_name, output_path):
    pdf_path = get_doc_path(doc_name)
    tables = []
    _, conv_res = process_pdf_docling(doc_converter, pdf_path)
    for table_ix, table in enumerate(conv_res.document.tables):
        tables.append(table.export_to_markdown())
    
    comapny = doc_name.split("_")[0]
    year_quarter = doc_name.split("_")[1]

    with open(output_path, "a") as f:
        table_json = {f"{comapny}_{year_quarter}": tables}
        json.dump(table_json, f)
        f.write("\n")


from ingest_data import get_table_summary_with_llm, index_tables_into_chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_tables_from_store(company_year_quarter):
    with open(TABLESTORE, "r") as f:
        table_store = f.readlines()
    
    table_store = [json.loads(ts) for ts in table_store]
    tables = []
    for ts in table_store:
        if company_year_quarter in ts:
            tables = ts[company_year_quarter]
            break

    return tables




if __name__ == "__main__":
    args = get_args()
    df = get_test_data(args)
    df = df.loc[df["company"] != "3M"]
    docs = df["doc_name"].unique().tolist()
    print(docs)
    # doc_converter = DocumentConverter()
    
    model = "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model)

    vector_store = Chroma(
        collection_name="table_collection",
        embedding_function=embeddings,
        persist_directory=os.path.join(DATAPATH, args.db_name),
    )

    for doc in docs:
        # extract_and_save_tables_from_pdf(doc_converter, doc, TABLESTORE)
        company_year = doc.split("_")[0] + "_" + doc.split("_")[1]
        metadata = {"doc_name": doc, "company_year": company_year}
        tables = get_tables_from_store(company_year)
        print(f"Found {len(tables)} tables for {doc}")
        for table_md in tables:
            summary = get_table_summary_with_llm(table_md, args)
            print("Table summary: ", summary)
            # index into chroma db
            vector_store=index_tables_into_chroma(vector_store, [(table_md,summary)], metadata)


# table_paths = glob.glob(os.path.join(DATAPATH, "3M_2018_10K-table-*.md"))
# print(len(table_paths))
# table_store = {}

# # tables = ""
# tables = []
# breaker ="*"*20
# for i, p in enumerate(table_paths):
#     with open(p, "r") as f:
#         table = f.read()
#         # tables += f"Table {i+1}\n{table}\n\n"
#         tables.append(table)

# table_store["3M_2018"] = tables

# with open(os.path.join(DATAPATH, "table_store.json"), "w") as f:
#     json.dump(table_store, f)