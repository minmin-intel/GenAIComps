# data
filedir=$WORKDIR/financebench/data/ #$WORKDIR/datasets/financebench/ #
filename=financebench_open_source.jsonl #difficult_questions.csv #
# filedir=$WORKDIR/datasets/financebench/results/
# filename=finqa_agent_v9_rest_rest_t0p5.csv

output_filename=finqa_agent_v9_llama4MavericFP8_t0p5_test
output=$WORKDIR/datasets/financebench/results/$output_filename.json
logfile=$WORKDIR/datasets/financebench/logs/$output_filename.log


# agent cofig
# vllm-gaudi
# model="meta-llama/Llama-3.3-70B-Instruct"
# llm_endpoint="http://localhost:8086"

# together api
# model="meta-llama/Llama-4-Scout-17B-16E-Instruct"
model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
# model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
api_key=$TOGETHER_API_KEY
llm_endpoint="https://api.together.ai"

temperature=0.5
recursion_limit=15
strategy=finqa #react_llama
tools=$WORKDIR/GenAIComps/comps/agent/src/tools/doc_retrieval.yaml

python test.py \
    --filedir $filedir \
    --filename $filename \
    --recursion_limit $recursion_limit \
    --strategy $strategy \
    --tools $tools \
    --model $model \
    --llm_endpoint_url $llm_endpoint \
    --api_key $api_key \
    --max_new_tokens 4096 \
    --temperature $temperature \
    --timeout 600 \
    --output $output | tee $logfile
    
    # --debug
    
    #| tee $WORKDIR/datasets/financebench/results/$output_filename.log
