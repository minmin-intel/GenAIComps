ip_address=$(hostname -I | awk '{print $1}')

vllm_port=8086

export agent_image="opea/agent:latest"

export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
export ip_address=$(hostname -I | awk '{print $1}')
export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export LLM_MODEL_ID="meta-llama/Meta-Llama-3.1-70B-Instruct"
export LLM_ENDPOINT_URL="http://${ip_address}:${vllm_port}"
export temperature=0.01
export max_new_tokens=4096
export TOOLSET_PATH=$WORKDIR/GenAIComps/comps/agent/src/tools/
echo "TOOLSET_PATH=${TOOLSET_PATH}"
export recursion_limit=15

export db_name=california_schools
export db_path="sqlite:////home/user/TAG-Bench/dev_folder/dev_databases/${db_name}/${db_name}.sqlite"

docker compose -f stream_sql.yaml up -d