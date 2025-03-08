import requests
import json

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_option", type=str, default="pdf", help="pdf or html")
    args = parser.parse_args()

    url = "http://localhost:6007/v1/dataprep/ingest"
    
    link_list = ["https://www.fool.com/earnings/call-transcripts/2025/03/07/gap-gap-q4-2024-earnings-call-transcript/"]

    test_html(url, link_list)
    