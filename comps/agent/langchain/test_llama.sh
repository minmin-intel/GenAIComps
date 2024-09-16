MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
LLMENDPOINT=http://${host_ip}:8085
STRATEGY=react_llama #rag_agent_llama

FILEDIR=$WORKDIR/datasets/crag_qas/
FILENAME=crag_20_answerable_queries.csv
OUTPUT=$WORKDIR/datasets/crag_results/v1_results_llama3.1-70B-instruct_20queries.csv


export RETRIEVAL_TOOL_URL="http://${host_ip}:8889/v1/retrievaltool"

python3 test.py \
--model ${MODEL} \
--llm_endpoint_url ${LLMENDPOINT} \
--strategy ${STRATEGY} \
--recursion_limit 15 \
--filedir ${FILEDIR} \
--filename ${FILENAME} \
--output ${OUTPUT} \
--test_llama \
--stream false
