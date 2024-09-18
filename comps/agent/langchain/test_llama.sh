MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
LLMENDPOINT=http://${host_ip}:8085
STRATEGY=react_llama #rag_agent_llama

FILEDIR=$WORKDIR/datasets/crag_qas/
FILENAME=crag_20_answerable_queries.csv
OUTPUT=$WORKDIR/datasets/crag_results/v1-react-hier-select-tool_llama3.1-70B-instruct_20queries.csv
TOOLS=tools/supervisor_agent_tools.yaml


export RETRIEVAL_TOOL_URL="http://${host_ip}:8889/v1/retrievaltool"
export CRAG_SERVER=http://${host_ip}:8080
export WORKER_AGENT_URL="http://${host_ip}:9095/v1/chat/completions"


AGENT_ENDPOINT=$WORKER_AGENT_URL
# echo "AGENT_ENDPOINT: $AGENT_ENDPOINT"

python3 benchmark.py \
--model ${MODEL} \
--llm_endpoint_url ${LLMENDPOINT} \
--strategy ${STRATEGY} \
--recursion_limit 15 \
--filedir ${FILEDIR} \
--filename ${FILENAME} \
--output ${OUTPUT} \
--tools $TOOLS \
--agent_endpoint_url ${AGENT_ENDPOINT} \
--test_llama \
--select_tool true \
--stream false
