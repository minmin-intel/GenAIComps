MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
LLMENDPOINT=http://${host_ip}:8085
STRATEGY=rag_agent_llama

python3 test.py \
--model ${MODEL} \
--llm_endpoint_url ${LLMENDPOINT} \
--strategy ${STRATEGY} \
--recursion_limit 15 \
--test_llama \
--stream false
