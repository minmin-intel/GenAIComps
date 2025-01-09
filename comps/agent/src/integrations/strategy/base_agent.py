# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from ..tools import get_tools_descriptions
from ..utils import adapt_custom_prompt, setup_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


class BaseAgent:
    def __init__(self, args, local_vars=None, **kwargs) -> None:
        self.llm = setup_chat_model(args)
        self.tools_descriptions = get_tools_descriptions(args.tools)
        self.app = None
        self.memory = None
        self.id = f"assistant_{self.__class__.__name__}_{uuid4()}"
        self.args = args
        adapt_custom_prompt(local_vars, kwargs.get("custom_prompt"))
        print(self.tools_descriptions)

    @property
    def is_vllm(self):
        return self.args.llm_engine == "vllm"

    @property
    def is_tgi(self):
        return self.args.llm_engine == "tgi"

    @property
    def is_openai(self):
        return self.args.llm_engine == "openai"

    def compile(self):
        pass

    def execute(self, state: dict):
        pass

    def prepare_initial_state(self, query):
        raise NotImplementedError

    async def stream_generator(self, query, config):
        # rag and react agent can use this streamming method
        # sql agent may need some customization
        initial_state = self.prepare_initial_state(query)
        try:
            async for event in self.app.astream(initial_state, config=config, stream_mode=["updates", "messages"]):
                # each event is like:
                # ('updates', {'agent': {'messages': [AIMessage(content="Hi, how can I help you?")]}})
                # ('messages', (AIMessageChunk(content='Hi'), {'langgraph_step': 3, 'langgraph_node': 'agent', ...}))
                event_type= event[0]
                data = event[1]
                if event_type == "updates": 
                    # this is a complete message - so it is not strictly "streamming" per se.
                    # can have long latency if input or output is long.
                    # instead, langgraph waits for the completed of the message and updates it.
                    # Also, AI message may not have content, but only have tool calls.
                    # as a result, we will miss the thinking process.
                    # But this can be fixed in planner code for react_llama, rag_llama, sql_llama.
                    # TODO: populate AIMessage content with LLM output (raw or processed TBD)
                    for node_name, node_state in data.items():
                        yield f"--- CALL {node_name} node ---\n"
                        for k, v in node_state.items():
                            if k=="messages" and v is not None:
                                if isinstance(v[0], AIMessage):
                                    yield f"AI message:\n{v[0].content}\n\n"
                                    if v[0].tool_calls:
                                        for tc in v[0].tool_calls:
                                            yield f"Tool call:\n{tc}\n\n"
                                elif isinstance(v[0], ToolMessage):
                                    yield f"Tool message:\n{v[0].content}\n\n"
                                elif isinstance(v[0], HumanMessage):
                                    yield f"Human message:\n{v[0].content}\n\n"
                                else:
                                    yield f"Other update:\n{v[0]}\n\n"
                elif event_type == "messages":
                    # this streams tokens of all LLM calls while LLM emits tokens
                    # so lowest latency between LLM output and UI
                    # we can get thinking process from this
                    # but it can be very long and verbose, and we may not want to show all of them (or maybe we do?)
                    # especially in sql_agent_llama, we also used LLM to parse outputs and fix sql queries 
                    # so the streamed tokens also includes the parser/query fixer tokens.
                    tokens = data[0].content
                    yield f"**Streamming tokens: {repr(tokens)}\n"
                else:
                    pass

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield str(e)

    async def non_streaming_run(self, query, config):
        initial_state = self.prepare_initial_state(query)
        print("@@@ Initial State: ", initial_state)
        try:
            async for s in self.app.astream(initial_state, config=config, stream_mode="values"):
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()

            last_message = s["messages"][-1]
            print("******Response: ", last_message.content)
            return last_message.content
        except Exception as e:
            return str(e)
