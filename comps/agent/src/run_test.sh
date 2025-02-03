LLM_MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
FILEDIR=$WORKDIR/TAG-Bench/query_by_db/
FILENAME=query_california_schools.csv

python test.py \
--llm_endpoint_url "http://localhost:8086" \
--model $LLM_MODEL_ID \
--with_memory true \
--strategy sql_agent_llama \
--db_name "california_schools" \
--db_path "sqlite:////$WORKDIR/TAG-Bench/dev_folder/dev_databases/california_schools/california_schools.sqlite" \
--recursion_limit 15 \
--filedir $FILEDIR \
--filename $FILENAME \
--output $WORKDIR/datasets/sql_agent_california_schools.csv


# --tools $WORKDIR/GenAIComps/comps/agent/src/tools/custom_tools.yaml \