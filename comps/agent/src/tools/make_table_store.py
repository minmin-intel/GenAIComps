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

if __name__ == "__main__":
    args = get_args()
    df = get_test_data(args)
    df = df.loc[df["company"] == "3M"]
    df = df.loc[df["doc_name"]!="3M_2018_10K"]
    docs = df["doc_name"].unique().tolist()
    doc_converter = DocumentConverter()
    for doc in docs:
        extract_and_save_tables_from_pdf(doc_converter, doc, TABLESTORE)

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