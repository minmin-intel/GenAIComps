import json

with open("sql_agent_log.json", "r") as f:
    data = f.readlines()

input_data = []
output_data = []

for line in data:
    parsed_line = json.loads(line)
    if "input" in parsed_line:
        input_data.append(parsed_line["input"])
    if "output" in parsed_line:
        if len(parsed_line["output"]) > 5:
            output_data.append(parsed_line["output"])

# calculate average len of input and output
input_len = sum([len(x) for x in input_data])/len(input_data)
output_len = sum([len(x) for x in output_data])/len(output_data)
max_input_len = max([len(x) for x in input_data])
max_output_len = max([len(x) for x in output_data])
print(f"Input length: avg {input_len}, max {max_input_len}")
print(f"Output length: avg {output_len}, max {max_output_len}")

