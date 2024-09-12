from huggingface_hub import ChatCompletionOutputToolCall, ChatCompletionOutputFunctionDefinition
import uuid

def convert_json_to_tool_call(json_str):
    return {'tool_calls': [ChatCompletionOutputToolCall(function=ChatCompletionOutputFunctionDefinition(arguments={'query': json_str["query"]}, name='duckduckgo_search', description=None), id=str(uuid.uuid4()), type='function')]}