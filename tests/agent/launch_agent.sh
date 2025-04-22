export agent_image="opea/agent:comps"
export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
export ip_address=$(hostname -I | awk '{print $1}')
export LLM_MODEL_ID="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" #"meta-llama/Llama-3.3-70B-Instruct"
export LLM_ENDPOINT_URL="https://api.together.ai"
export LLM_API_KEY=${TOGETHER_API_KEY}
export temperature=0.01
export max_new_tokens=4096
export TOOLSET_PATH=$WORKDIR/GenAIComps/comps/agent/src/tools/
echo "TOOLSET_PATH=${TOOLSET_PATH}"
export recursion_limit=5


docker compose -f $WORKDIR/GenAIComps/tests/agent/reactllama.yaml up -d