llm_endpoint_url="http://${host_ip}:8085"
model="meta-llama/Meta-Llama-3.1-70B-Instruct"

python3 test.py \
--llm_engine vllm \
--llm_endpoint_url $llm_endpoint_url \
--model $model \
--streaming false \
--ut