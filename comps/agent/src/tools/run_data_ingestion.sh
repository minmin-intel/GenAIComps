db_name="test_cocacola_v7"
filedir=$WORKDIR/financebench/data/ #$WORKDIR/datasets/financebench/ #
filename=financebench_open_source.jsonl #difficult_questions.csv #
temperature=0.0


python ingest_data.py \
--db_name $db_name \
--filedir $filedir \
--filename $filename \
--temperature $temperature \
--read_processed \
--generate_metadata
