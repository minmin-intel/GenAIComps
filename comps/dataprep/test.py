import requests
import os

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

if __name__ == "__main__":
    url = "http://localhost:6007/v1/dataprep/ingest"
    test_pdf(url)