MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
LLMENDPOINT=http://${host_ip}:8085
STRATEGY=react_llama #rag_agent_llama #
TEMPERATURE=0.01
TOPK=10

FILEDIR=$WORKDIR/datasets/ragagent_eval/
FILENAME=crag_qa_music.jsonl
OUTPUT=$WORKDIR/datasets/ragagent_eval/react_music_full.jsonl
TOOLS=tools/supervisor_agent_tools.yaml


export RETRIEVAL_TOOL_URL="http://${host_ip}:8889/v1/retrievaltool"
export CRAG_SERVER=http://${host_ip}:8080
export WORKER_AGENT_URL="http://${host_ip}:9095/v1/chat/completions"


AGENT_ENDPOINT="http://${host_ip}:9090/v1/chat/completions" 


python3 benchmark.py \
--model ${MODEL} \
--llm_endpoint_url ${LLMENDPOINT} \
--temperature ${TEMPERATURE} \
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
