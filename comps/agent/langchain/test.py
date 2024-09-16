# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import traceback
from time import sleep

import pandas as pd
import requests
from src.utils import format_date, get_args
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage


def test_agent_local(args):
    from src.agent import instantiate_agent

    if args.q == 0:
        df = pd.DataFrame({"query": ["What is the Intel OPEA Project?"]})
    elif args.q == 1:
        df = pd.DataFrame({"query": ["what is the trade volume for Microsoft today?"]})
    elif args.q == 2:
        df = pd.DataFrame({"query": ["what is the hometown of Year 2023 Australia open winner?"]})

    agent = instantiate_agent(args, strategy=args.strategy)
    app = agent.app

    config = {"recursion_limit": args.recursion_limit}

    traces = []
    success = 0
    for _, row in df.iterrows():
        print("Query: ", row["query"])
        initial_state = {"messages": [{"role": "user", "content": row["query"]}]}
        try:
            trace = {"query": row["query"], "trace": []}
            for event in app.stream(initial_state, config=config):
                trace["trace"].append(event)
                for k, v in event.items():
                    print("{}: {}".format(k, v))

            traces.append(trace)
            success += 1
        except Exception as e:
            print(str(e), str(traceback.format_exc()))
            traces.append({"query": row["query"], "trace": str(e)})

        print("-" * 50)

    df["trace"] = traces
    df.to_csv(os.path.join(args.filedir, args.output), index=False)
    print(f"succeed: {success}/{len(df)}")


def test_agent_http(args):
    proxies = {"http": ""}
    ip_addr = args.ip_addr
    url = f"http://{ip_addr}:9090/v1/chat/completions"

    def process_request(query):
        content = json.dumps({"query": query})
        print(content)
        try:
            resp = requests.post(url=url, data=content, proxies=proxies)
            ret = resp.text
            resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
        except requests.exceptions.RequestException as e:
            ret = f"An error occurred:{e}"
        print(ret)
        return ret

    if args.quick_test:
        df = pd.DataFrame({"query": ["What is the weather today in Austin?"]})
    elif args.quick_test_multi_args:
        df = pd.DataFrame({"query": ["what is the trade volume for Microsoft today?"]})
    else:
        df = pd.read_csv(os.path.join(args.filedir, args.filename))
        df = df.sample(n=2, random_state=42)
    traces = []
    for _, row in df.iterrows():
        ret = process_request(row["query"])
        trace = {"query": row["query"], "trace": ret}
        traces.append(trace)

    df["trace"] = traces
    df.to_csv(os.path.join(args.filedir, args.output), index=False)


def test_assistants_http(args):
    proxies = {"http": ""}
    ip_addr = args.ip_addr
    url = f"http://{ip_addr}:9090/v1"

    def process_request(api, query, is_stream=False):
        content = json.dumps(query) if query is not None else None
        print(f"send request to {url}/{api}, data is {content}")
        try:
            resp = requests.post(url=f"{url}/{api}", data=content, proxies=proxies, stream=is_stream)
            if not is_stream:
                ret = resp.json()
                print(ret)
            else:
                for line in resp.iter_lines(decode_unicode=True):
                    print(line)
                ret = None

            resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
            return ret
        except requests.exceptions.RequestException as e:
            ret = f"An error occurred:{e}"
            print(ret)
            return False

    # step 1. create assistants
    query = {}
    if ret := process_request("assistants", query):
        assistant_id = ret.get("id")
        print("Created Assistant Id: ", assistant_id)
    else:
        print("Error when creating assistants !!!!")
        return

    # step 2. create threads
    query = {}
    if ret := process_request("threads", query):
        thread_id = ret.get("id")
        print("Created Thread Id: ", thread_id)
    else:
        print("Error when creating threads !!!!")
        return

    # step 3. add messages
    if args.query is None:
        query = {"role": "user", "content": "How old was Bill Gates when he built Microsoft?"}
    else:
        query = {"role": "user", "content": args.query}
    if ret := process_request(f"threads/{thread_id}/messages", query):
        pass
    else:
        print("Error when add messages !!!!")
        return

    # step 4. run
    print("You may cancel the running process with cmdline")
    print(f"curl {url}/threads/{thread_id}/runs/cancel -X POST -H 'Content-Type: application/json'")

    query = {"assistant_id": assistant_id}
    process_request(f"threads/{thread_id}/runs", query, is_stream=True)


def test_ut(args):
    from src.tools import get_tools_descriptions

    tools = get_tools_descriptions("tools/custom_tools.py")
    for tool in tools:
        print(tool)



def non_streaming_run(agent, query, config):
    initial_state = agent.prepare_initial_state(query)

    for s in agent.app.stream(initial_state, config=config, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, ToolMessage):
            pass
        else:
            message.pretty_print()
        # print(s["messages"])

    last_message = s["messages"][-1]
    print("******Response: ", last_message.content)
    trace = get_trace(s["messages"])
    num_llm_calls = count_llm_calls(s["messages"])
    return last_message.content, trace, num_llm_calls

def count_llm_calls(messages):
    count = 0
    for m in messages[1:]:
        if isinstance(m, ToolMessage):
            pass
        else:
            count += 1
    return count+1


def get_trace(messages):
    trace = []
    for m in messages:
        if isinstance(m, AIMessage):
            try:
                tool_calls = m.additional_kwargs["tool_calls"]
                trace.append(tool_calls)
            except:
                trace.append(m.content)
        if isinstance(m, ToolMessage):
            trace.append(m.content)
    return trace



def test_query_writer_llama(args):
    from src.strategy.ragagent.planner import QueryWriterLlama
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain_core.messages import HumanMessage

    query = "What is the Intel OPEA Project?"

    host_ip = os.getenv("host_ip", "localhost")
    port = os.getenv("port", 8085)
    endpoint = f"http://{host_ip}:{port}"

    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "return_full_text": args.return_full_text,
        "streaming": args.streaming,
    }

    print("generation_params:\n", generation_params)

    llm_endpoint = HuggingFaceEndpoint(
        endpoint_url=endpoint,  ## endpoint_url = "localhost:8080",
        task="text-generation",
        **generation_params,
    )
    
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    query_writer = QueryWriterLlama(llm_endpoint, model_id)
    initial_state = {"messages": [HumanMessage(content=query)]}
    print(query_writer(initial_state))

def test_local_ragagent_llama(args):
    from src.agent import instantiate_agent
    agent = instantiate_agent(args, strategy=args.strategy)
    config = {"recursion_limit": args.recursion_limit}

    # query = "What is the Intel OPEA Project?"
    # query = "who has had more number one hits on the us billboard hot 100 chart, michael jackson or elvis presley?"
    # query = "what's the most recent album from the founder of ysl records?"
    # query = "what is the hometown of Year 2023 Australia open winner?"
    # query = "Hello, how are you?"
    # response = non_streaming_run(agent, query, config)

    df = pd.read_csv(os.path.join(args.filedir, args.filename))
    # df = df.head(2)
    answers = []
    traces = []
    num_llm_calls = []
    for _, row in df.iterrows():
        q = row["query"]
        t = row["query_time"]
        query = "Question: {} \nThe question was asked at: {}".format(q, t)
        print("Query: ", query)  
        response, trace, n = non_streaming_run(agent, query, config)
        answers.append(response)  
        traces.append(trace)  
        num_llm_calls.append(n)
    if "answer" in df.columns:
        df.rename(columns={"answer": "ref_answer"}, inplace=True)    
    df["answer"] = answers
    df["trace"] = traces
    df["num_llm_calls"] = num_llm_calls
    df.to_csv(args.output, index=False)

PROMPT = """\
### You are a helpful, respectful and honest assistant.
You are given a Question and the time when it was asked in the Pacific Time Zone (PT), referred to as "Query
Time". The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT".
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer “invalid question”.
2. If you are uncertain or do not know the answer, respond with “I don’t know”.
3. Refer to the search results to form your answer.
5. Give concise, factual and relevant answers.

### Search results: {context} \n
### Question: {question} \n
### Query Time: {time} \n
### Answer:
"""

def generate_answer(args, query, context, time):
    from langchain_huggingface import ChatHuggingFace
    from src.utils import setup_hf_tgi_client
    prompt = PROMPT.format(context=context, question=query, time=time)
    llm = setup_hf_tgi_client(args)
    chat_llm = ChatHuggingFace(llm=llm, model_id=args.model)
    response = chat_llm.invoke(prompt)
    return response.content


def test_local_rag(args):
    from tools.worker_agent_tools import search_knowledge_base
    df = pd.read_csv(os.path.join(args.filedir, args.filename))
    # df = df.head(2)
    answers = []
    contexts = []
    for _, row in df.iterrows():
        q = row["query"]
        t = row["query_time"]
        print("========== Query: ", q)
        context = search_knowledge_base(q)
        print("========== Context: ", context)
        answer = generate_answer(args, q, context, t)  
        print("========== Answer: ", answer)
        answers.append(answer)
        contexts.append(context)

    if "answer" in df.columns:
        df.rename(columns={"answer": "ref_answer"}, inplace=True)    
    df["answer"] = answers
    df["context"] = contexts
    df.to_csv(args.output, index=False)



if __name__ == "__main__":
    args1, _ = get_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="react")
    parser.add_argument("--local_test", action="store_true", help="Test with local mode")
    parser.add_argument("--endpoint_test", action="store_true", help="Test with endpoint mode")
    parser.add_argument("--assistants_api_test", action="store_true", help="Test with endpoint mode")
    parser.add_argument("--test_llama", action="store_true", help="test llama3.1 based ragagent")
    parser.add_argument("--test_rag", action="store_true", help="test conventional rag")
    parser.add_argument("--q", type=int, default=0)
    parser.add_argument("--ip_addr", type=str, default="127.0.0.1", help="endpoint ip address")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--filedir", type=str, default="./", help="test file directory")
    parser.add_argument("--filename", type=str, default="query.csv", help="query_list_file")
    parser.add_argument("--output", type=str, default="output.csv", help="query_list_file")
    parser.add_argument("--ut", action="store_true", help="ut")

    args, _ = parser.parse_known_args()

    for key, value in vars(args1).items():
        setattr(args, key, value)

    if args.local_test:
        test_agent_local(args)
    elif args.endpoint_test:
        test_agent_http(args)
    elif args.ut:
        test_ut(args)
    elif args.assistants_api_test:
        test_assistants_http(args)
    elif args.test_llama:
        test_local_ragagent_llama(args)
    elif args.test_rag:
        test_local_rag(args)
    else:
        print("Please specify the test type")
