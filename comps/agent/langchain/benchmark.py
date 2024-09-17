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


def test_local_ragagent_llama(args):
    from src.agent import instantiate_agent
    agent = instantiate_agent(args, strategy=args.strategy)
    config = {"recursion_limit": args.recursion_limit}

    query=[
        # "what song topped the billboard chart on 2004-02-04?",
        # "Hello, how are you?",
        "tell me the most recent song or album by doris duke?",
    ]
    query_time = [
        "03/01/2024, 00:00:00 PT",
    ]

    df = pd.DataFrame({"query": query, "query_time": query_time})
    # df = pd.read_csv(os.path.join(args.filedir, args.filename))
    # df = df.head(1)
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
    # df.to_csv(args.output, index=False)

def test_multiagent(args):
    from src.agent import instantiate_agent

    # worker_agent = instantiate_agent(args, strategy=args.strategy)
    # supervisor_agent = instantiate_agent(args, strategy=args.strategy)
    config = {"recursion_limit": args.recursion_limit}

    query=[
        # "what song topped the billboard chart on 2004-02-04?",
        # "Hello, how are you?",
        "tell me the most recent song or album by doris duke?",
    ]
    query_time = [
        "03/01/2024, 00:00:00 PT",
    ]

    df = pd.DataFrame({"query": query, "query_time": query_time})
    # df = pd.read_csv(os.path.join(args.filedir, args.filename))
    # df = df.head(1)
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


##############################################################
############# Conventional RAG ###############################
##############################################################

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
    parser.add_argument("--test_llama", action="store_true", help="test llama3.1 based ragagent")
    parser.add_argument("--test_rag", action="store_true", help="test conventional rag")
    parser.add_argument("--filedir", type=str, default="./", help="test file directory")
    parser.add_argument("--filename", type=str, default="query.csv", help="query_list_file")
    parser.add_argument("--output", type=str, default="output.csv", help="query_list_file")
    parser.add_argument("--worker_tools", type=str, default="/home/user/tools/worker_agent_tools.yaml")
    parser.add_argument("--supervisor_tools", type=str, default="/home/user/tools/supervisor_agent_tools.yaml")

    args, _ = parser.parse_known_args()

    for key, value in vars(args1).items():
        setattr(args, key, value)
    
    if args.test_llama:
        test_local_ragagent_llama(args)
    elif args.test_rag:
        test_local_rag(args)
    else:
        print("Please specify the test type")
