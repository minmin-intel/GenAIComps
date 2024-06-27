# used by llm
export HF_TOKEN=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
export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
export LLM_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 #"meta-llama/Meta-Llama-3-8B-Instruct"
# export local_model_dir=<YOUR LOCAL DISK TO STORE MODEL>
# used by agent
# export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
# export custom_tool_dir=/localdisk/minminho/GenAIComps/comps/agent/langchain/tools/
# export agent_env=<YOUR CUSTOM AGENT SETTINGS> #./comps/agent/langchain/AGENT_ENV

# if you are testing local, use below cmd to add AGENT env
# set -o allexport; source ${agent_env}; set +o allexport

docker compose -f docker_compose.yaml up -d