LLM_MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"

### test sql agent llama
# FILEDIR=$WORKDIR/TAG-Bench/query_by_db/
# FILENAME=query_california_schools.csv
# python test.py \
# --llm_endpoint_url "http://localhost:8086" \
# --model $LLM_MODEL_ID \
# --with_memory true \
# --strategy sql_agent_llama \
# --db_name "california_schools" \
# --db_path "sqlite:////$WORKDIR/TAG-Bench/dev_folder/dev_databases/california_schools/california_schools.sqlite" \
# --recursion_limit 15 \
# --filedir $FILEDIR \
# --filename $FILENAME \
# --output $WORKDIR/datasets/sql_agent_california_schools.csv


### test rag agent: react_llama
FILEDIR=$WORKDIR/datasets/crag_qas
FILENAME=crag_qa_music_sampled_with_query_time.jsonl
python test.py \
--llm_endpoint_url "http://localhost:8086" \
--model $LLM_MODEL_ID \
--strategy react_llama \
--tools tools/supervisor_agent_tools.yaml \
--recursion_limit 15 \
--filedir $FILEDIR \
--filename $FILENAME \
--output $WORKDIR/datasets/react_agent_music.csv


