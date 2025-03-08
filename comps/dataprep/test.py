import requests
import os
import json

WORKDIR = os.getenv("WORKDIR", "/workspace")
DATAPATH= os.path.join(WORKDIR, "financebench/pdfs")
def test_pdf(url):
    proxies = {"http": ""}
    file_list = os.listdir(DATAPATH)
    print(file_list[:2])
    file_list = file_list[:2]
    files = [("files", (f, open(os.path.join(DATAPATH, f), "rb"))) for f in file_list]
    
    try:
        resp = requests.request("POST", url=url, headers={}, files=files, proxies=proxies)
        print(resp.text)
        resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
        print("Request successful!")
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)


def test_html(url, link_list):
    proxies = {"http": ""}
    payload = {"link_list": json.dumps(link_list)}
    try:
        resp = requests.post(url=url, data=payload, proxies=proxies)
        print(resp.text)
        resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
        print("Request successful!")
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)

def test_delete(url, filename):
    proxies = {"http": ""}
    payload = {"file_path": filename}
    try:
        resp = requests.post(url=url, json=payload, proxies=proxies)
        print(resp.text)
        resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
        print("Request successful!")
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_option", type=str, default="pdf", help="pdf or html")
    args = parser.parse_args()

    url = "http://localhost:6007/v1/dataprep/ingest"
    if args.test_option == "pdf":
        test_pdf(url)
    elif args.test_option == "html":
        link_list = ["https://investors.3m.com/financials/sec-filings/content/0000066740-24-000101/0000066740-24-000101.pdf",
                     "https://www.fool.com/earnings/call-transcripts/2025/03/07/gap-gap-q4-2024-earnings-call-transcript/"]
        test_html(url, link_list)
    elif args.test_option == "delete":
        url = "http://localhost:6007/v1/dataprep/delete"
        filename = "https://investors.3m.com/financials/sec-filings/content/0000066740-24-000101/0000066740-24-000101.pdf"
        test_delete(url, filename)

    else:
        raise ValueError("Invalid test_option value. Please use pdf or html")
