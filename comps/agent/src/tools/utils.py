from openai import OpenAI
import pandas as pd
import os
import argparse
import requests

"""
    chunk_size: int = Form(1500),
    chunk_overlap: int = Form(100),
    process_table: bool = Form(False),
    table_strategy: str = Form("fast"),
"""
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_endpoint_url", type=str, default="http://localhost:8086")
    parser.add_argument("--agent_url", type=str, default="http://localhost:9095/v1/chat/completions")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ip_address", type=str, default="localhost")
    parser.add_argument("--chunk_size", type=int, default=1500)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument("--retrieval_endpoint_url", type=str, default="http://localhost:8889/v1/retrievaltool")
    parser.add_argument("--output", type=str, default="output.jsonl")
    parser.add_argument("--ingest_option", type=str, default="docling")
    parser.add_argument("--retriever_option", type=str, default="plain")
    parser.add_argument("--chunk_option", type=str, default="chunk_summarize", help="chunk_summarize, text_table")
    parser.add_argument("--filedir", type=str, default="./", help="test file directory")
    parser.add_argument("--filename", type=str, default="query.csv", help="query_list_file")
    parser.add_argument("--db_name", type=str, help="name of vector db")
    parser.add_argument("--db_collection", type=str, help="name of collection")
    parser.add_argument("--debug", action="store_true", help="ut")
    parser.add_argument("--read_processed", action="store_true", help="read processed data")
    parser.add_argument("--generate_metadata", action="store_true", help="generate metadata with LLM")
    parser.add_argument("--update_metadata", action="store_true", help="update metadata of docs in vector db")
    args = parser.parse_args()
    return args

WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'financebench/data/')
PDFPATH=os.path.join(WORKDIR, 'financebench/pdfs/')

# def get_test_data():
#     filename = "financebench_open_source.jsonl"
#     df = pd.read_json(DATAPATH + filename, lines=True)
#     return df

def get_test_data(args):
    if args.debug:
        test_questions = [
            # "Considering the data in the balance sheet, what is Block's (formerly known as Square) FY2016 working capital ratio? Define working capital ratio as total current assets divided by total current liabilities. Round your answer to two decimal places.",
            # "We need to calculate a financial metric by using information only provided within the balance sheet. Please answer the following question: what is Boeing's year end FY2018 net property, plant, and equipment (in USD millions)?",
            "What is Coca Cola's FY2021 COGS % margin? Calculate what was asked by utilizing the line items clearly shown in the income statement.",
            "Is CVS Health a capital-intensive business based on FY2022 data?",
            "What drove gross margin change as of FY2022 for JnJ? If gross margin is not a useful metric for a company like this, then please state that and explain why.",
            "In 2022 Q2, which of JPM's business segments had the highest net income?",
            "Which region had the Highest EBITDAR Contribution for MGM during FY2022?",
        ]

        df = pd.DataFrame({"question": test_questions})
    else:
        filename = os.path.join(args.filedir, args.filename)
        if filename.endswith(".csv"):
            df = pd.read_csv(filename)
        elif filename.endswith(".json") or filename.endswith(".jsonl"):
            df = pd.read_json(filename, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    return df

def get_doc_path(doc_name):
    return os.path.join(PDFPATH, doc_name)+".pdf"

def generate_answer(args, prompt):
    """
    Use vllm endpoint to generate the answer
    """
    # send request to vllm endpoint
    client = OpenAI(
        base_url=f"{args.llm_endpoint_url}/v1",
        api_key="token-abc123",
    )

    params = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }

    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        **params
        )

    # get response
    response = completion.choices[0].message.content

    return response


def run_agent(args, query):
    url = args.agent_url
    proxies = {"http": ""}
    payload = {
        "messages": query,
    }
    response = requests.post(url, json=payload, proxies=proxies)
    answer = response.json()["text"]
    return answer