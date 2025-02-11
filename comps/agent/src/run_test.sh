# data
filedir=$WORKDIR/financebench/data/
filename=financebench_open_source.jsonl
output=$WORKDIR/datasets/financebench/results/react_agent_v2_all_t0p5.json

# agent cofig
model="meta-llama/Llama-3.3-70B-Instruct"
temperature=0.5
llm_endpoint="http://localhost:8085"
recursion_limit=15
strategy=react_llama
tools=$WORKDIR/GenAIComps/comps/agent/src/tools/doc_retrieval.yaml

python test.py \
    --filedir $filedir \
    --filename $filename \
    --recursion_limit $recursion_limit \
    --strategy $strategy \
    --tools $tools \
    --model $model \
    --llm_endpoint_url $llm_endpoint \
    --max_new_tokens 4096 \
    --temperature $temperature \
    --output $output | tee $WORKDIR/datasets/financebench/results/react_agent_v2_all.log
