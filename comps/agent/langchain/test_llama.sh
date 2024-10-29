MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
LLMENDPOINT=http://${host_ip}:8085
STRATEGY=react_llama #rag_agent_llama #
TEMPERATURE=0.01
TOPK=10

FILEDIR=$WORKDIR/datasets/crag_qas/
FILENAME=crag_qa_music_sampled_with_query_time.jsonl #crag_20_answerable_queries.csv #
OUTPUT=$WORKDIR/datasets/crag_results/react_tgi_hfendp_llama3.1-70B-instruct_92queries.csv
TOOLS=tools/supervisor_agent_tools.yaml


export RETRIEVAL_TOOL_URL="http://${host_ip}:8889/v1/retrievaltool"
export CRAG_SERVER=http://${host_ip}:8080
export WORKER_AGENT_URL="http://${host_ip}:9095/v1/chat/completions"


AGENT_ENDPOINT=$WORKER_AGENT_URL
# echo "AGENT_ENDPOINT: $AGENT_ENDPOINT"

python3 benchmark.py \
--model ${MODEL} \
--llm_endpoint_url ${LLMENDPOINT} \
--temperature ${TEMPERATURE} \
--top_k ${TOPK} \
--max_new_tokens 8192 \
--strategy ${STRATEGY} \
--recursion_limit 16 \
--filedir ${FILEDIR} \
--filename ${FILENAME} \
--output ${OUTPUT} \
--tools $TOOLS \
--agent_endpoint_url ${AGENT_ENDPOINT} \
--test_api \
--select_tool true \
--stream false \
--llm_engine tgi \
--llm_api_mode chat_openai
