import json
from transformers import AutoTokenizer

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

with open("rag_agent_log.json", "r") as f:
    data = f.readlines()

input_data = []
output_data = []

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

for line in data:
    parsed_line = json.loads(line)
    if "input" in parsed_line:
        input_data.append(count_tokens(parsed_line["input"], tokenizer))
    if "output" in parsed_line:
        if len(parsed_line["output"]) > 5:
            output_data.append(count_tokens(parsed_line["output"], tokenizer))

# calculate average len of input and output
input_len = sum(input_data)/len(input_data)
output_len = sum(output_data)/len(output_data)
max_input_len = max(input_data)
max_output_len = max(output_data)

print(f"Input length: avg {input_len}, max {max_input_len}")
print(f"Output length: avg {output_len}, max {max_output_len}")

