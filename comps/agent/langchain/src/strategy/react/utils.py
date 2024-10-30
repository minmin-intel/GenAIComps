# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import uuid

import numpy as np
from huggingface_hub import ChatCompletionOutputFunctionDefinition, ChatCompletionOutputToolCall
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.output_parsers import BaseOutputParser


class ReActLlamaOutputParser(BaseOutputParser):
    def parse(self, text: str):
        print("raw output from llm: ", text)
        json_lines = text.split("\n")
        # print("json_lines: ", json_lines)
        output = []
        for line in json_lines:
            try:
                if "TOOL CALL:" in line:
                    line = line.replace("TOOL CALL:", "")
                if "FINAL ANSWER:" in line:
                    line = line.replace("FINAL ANSWER:", "")
                if "assistant" in line:
                    line = line.replace("assistant", "")
                parsed_line = json.loads(line)
                if isinstance(parsed_line, dict):
                    print("parsed line: ", parsed_line)
                    output.append(parsed_line)
            except Exception as e:
                print("Exception happened in output parsing: ", str(e))
        if output:
            return output
        else:
            return text  # None


def convert_json_to_tool_call(json_str):
    tool_name = json_str["tool"]
    tool_args = json_str["args"]
    tcid = str(uuid.uuid4())
    add_kw_tc = {
        "tool_calls": [
            ChatCompletionOutputToolCall(
                function=ChatCompletionOutputFunctionDefinition(arguments=tool_args, name=tool_name, description=None),
                id=tcid,
                type="function",
            )
        ]
    }
    tool_call = ToolCall(name=tool_name, args=tool_args, id=tcid)
    return add_kw_tc, tool_call

def get_tool_output(messages, id):
    for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                if msg.tool_call_id == id:
                    tool_output = msg.content
                    break
    return tool_output

def assemble_history(messages):
    """
    messages: AI, TOOL, AI, TOOL, etc.
    """
    query_history = ""
    breaker = "-"*10
    for m in messages[1:]:  # exclude the first message
        if isinstance(m, AIMessage):
            # if there is tool call
            if hasattr(m, "tool_calls") and len(m.tool_calls) > 0:
                for tool_call in m.tool_calls:
                    tool = tool_call["name"]
                    tc_args = tool_call["args"]
                    id = tool_call["id"]
                    tool_output = get_tool_output(messages, id)
                    query_history += f"Tool Call: {tool} - {tc_args}\nTool Output: {tool_output}\n{breaker}\n"
            else:
                # did not make tool calls
                query_history += f"Assistant Output: {m.content}\n"
    
    return query_history

def describe_tools(tools):
    function_info = []
    for tool in tools:
        function_name = tool.name
        doc_string = tool.description
        if function_name != "search_knowledge_base": # right now this is hard coded, in future can be a must-include list
            function_info.append("{}: {}".format(function_name, doc_string))
        # print("-"*50)
    return function_info

def sort_list(list1, list2):
    import numpy as np
    # Use numpy's argsort function to get the indices that would sort the second list
    idx = np.argsort(list2)# ascending order
    return np.array(list1)[idx].tolist()[::-1]# descending order

def get_topk_tools(topk, tools, similarities):
    sorted_tools = sort_list(tools, similarities)
    # print(sorted_tools)
    top_k_tools = sorted_tools[:topk]
    return [x.split(':')[0] for x in top_k_tools]

def select_tools_for_query(query, tools_embedding, model, topk, tools_descriptions):
    # tool descriptions is a list of strings as returned by describe_tools
    query_embedding = model.encode(query)
    similarities = model.similarity(query_embedding, tools_embedding).flatten() # 1D array
    top_k_tools = get_topk_tools(topk, tools_descriptions, similarities)
    return top_k_tools

def get_tool_with_name(tool_name, tools):
    # tools = get_all_available_tools()
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None

def get_selected_tools(top_k_tools, tools):
    selected_tools = []
    for tool_name in top_k_tools:
        tool = get_tool_with_name(tool_name, tools)
        if tool is not None:
            selected_tools.append(tool)
    try:
        selected_tools.append(get_tool_with_name("search_knowledge_base", tools))
    except:
        pass
    # print("Selected tools: ", selected_tools)
    return selected_tools