# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent as create_react_langchain_agent
from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import ChatHuggingFace
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from ...global_var import threads_global_kv
from ...storage.persistence_redis import RedisPersistence
from ...utils import filter_tools, has_multi_tool_inputs, tool_renderer
from ..base_agent import BaseAgent



###############################################################################
# ReActAgentLlama:
# Only validated with with Llama3.1-70B-Instruct model served with TGI-gaudi
# support multiple tools
# does not rely on langchain bind_tools API
# since tgi and vllm still do not have very good support for tool calling like OpenAI

import json
from typing import Annotated, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

from ...storage.persistence_memory import AgentPersistence, PersistenceConfig
from ...utils import setup_chat_model
from .utils import (
    assemble_history,
    assemble_memory,
    assemble_memory_from_store,
    convert_json_to_tool_call,
    save_state_to_store,
)


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_choice: Optional[List[str]] = None
    is_last_step: IsLastStep


class ReActAgentNodeLlama:
    """Do planning and reasoning and generate tool calls.

    A workaround for open-source llm served by TGI-gaudi.
    """

    def __init__(self, tools, args, store=None):
        from .prompt import REACT_AGENT_LLAMA_PROMPT
        from .utils import ReActLlamaOutputParser

        output_parser = ReActLlamaOutputParser()
        prompt = PromptTemplate(
            template=REACT_AGENT_LLAMA_PROMPT,
            input_variables=["input", "history", "tools"],
        )
        llm = setup_chat_model(args)
        self.tools = tools
        self.chain = prompt | llm
        self.output_parser = output_parser
        self.with_memory = args.with_memory
        self.memory_type = args.memory_type
        self.store = store

    def __call__(self, state, config):

        print("---CALL Agent node---")
        messages = state["messages"]

        # assemble a prompt from messages
        if self.with_memory:
            if self.memory_type == "volatile":
                query, history, thread_history = assemble_memory(messages)
            elif self.memory_type == "persistent":
                # use thread_id, assistant_id to search memory from store
                print("@@@ Load memory from store....")
                query, history, thread_history = assemble_memory_from_store(config, self.store)  # TODO
            else:
                raise ValueError("Invalid memory type!")
        else:
            query = messages[0].content
            history = assemble_history(messages)
            thread_history = ""

        # print("@@@ Turn History:\n", history)
        # print("@@@ Thread history:\n", thread_history)

        tools_used = self.tools
        if state.get("tool_choice") is not None:
            tools_used = filter_tools(self.tools, state["tool_choice"])

        tools_descriptions = tool_renderer(tools_used)
        # print("@@@ Tools description: ", tools_descriptions)

        # invoke chain: raw output from llm
        response = self.chain.invoke(
            {"input": query, "history": history, "tools": tools_descriptions, "thread_history": thread_history}
        )
        response = response.content

        # parse tool calls or answers from raw output: result is a list
        output = self.output_parser.parse(response)
        # print("@@@ Output from chain: ", output)

        # convert output to tool calls
        tool_calls = []
        if output:
            for res in output:
                if "tool" in res:
                    tool_call = convert_json_to_tool_call(res)
                    # print("Tool call:\n", tool_call)
                    tool_calls.append(tool_call)

            if tool_calls:
                ai_message = AIMessage(content=response, tool_calls=tool_calls)
            elif "answer" in output[0]:
                ai_message = AIMessage(content=str(output[0]["answer"]))
        else:
            ai_message = AIMessage(content=response)

        return {"messages": [ai_message]}


class FinReActAgentLlama(BaseAgent):
    def __init__(self, args, **kwargs):
        super().__init__(args, local_vars=globals(), **kwargs)

        agent = ReActAgentNodeLlama(tools=self.tools_descriptions, args=args, store=self.store)
        tool_node = ToolNode(self.tools_descriptions)

        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", agent)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "tools",
                # Otherwise we finish.
                "end": END,
            },
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        workflow.add_edge("tools", "agent")

        if args.with_memory:
            if args.memory_type == "volatile":
                self.app = workflow.compile(checkpointer=self.checkpointer)
            elif args.memory_type == "persistent":
                self.app = workflow.compile(store=self.store)
            else:
                raise ValueError("Invalid memory type!")
        else:
            self.app = workflow.compile()

    # Define the function that determines whether to continue or not
    def should_continue(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    def prepare_initial_state(self, query):
        print("---Prepare initial state---")
        return {"messages": [HumanMessage(content=query)]}

    async def stream_generator(self, query, config, thread_id=None):
        initial_state = self.prepare_initial_state(query)
        if "tool_choice" in config:
            initial_state["tool_choice"] = config.pop("tool_choice")

        try:
            print("---Start running---")
            async for event in self.app.astream(initial_state, config=config, stream_mode=["updates"]):
                print(event)
                event_type = event[0]
                data = event[1]
                if event_type == "updates":
                    for node_name, node_state in data.items():
                        if self.memory_type == "persistent":
                            save_state_to_store(node_state, config, self.store)
                        print(f"--- CALL {node_name} node ---\n")
                        for k, v in node_state.items():
                            if v is not None:
                                print(f"------- {k}, {v} -------\n\n")
                                if node_name == "agent":
                                    if v[0].content == "":
                                        tool_names = []
                                        for tool_call in v[0].tool_calls:
                                            tool_names.append(tool_call["name"])
                                        result = {"tool": tool_names}
                                    else:
                                        result = {"content": [v[0].content.replace("\n\n", "\n")]}
                                    # ui needs this format
                                    yield f"data: {json.dumps(result)}\n\n"
                                elif node_name == "tools":
                                    full_content = v[0].content
                                    tool_name = v[0].name
                                    result = {"tool": tool_name, "content": [full_content]}
                                    yield f"data: {json.dumps(result)}\n\n"
                                    if not full_content:
                                        continue

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield str(e)

    async def non_streaming_run(self, query, config):
        # for use as worker agent (tool of supervisor agent)
        # only used in chatcompletion api
        # chat completion api only supports volatile memory
        initial_state = self.prepare_initial_state(query)
        if "tool_choice" in config:
            initial_state["tool_choice"] = config.pop("tool_choice")
        try:
            async for s in self.app.astream(initial_state, config=config, stream_mode="values"):
                message = s["messages"][-1]
                message.pretty_print()
            last_message = s["messages"][-1]
            print("******Response: ", last_message.content)
            return last_message.content
        except Exception as e:
            return str(e)
