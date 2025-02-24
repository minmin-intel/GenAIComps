db_name="test_3M_section_summary"
filedir=$WORKDIR/financebench/data/ #$WORKDIR/datasets/financebench/ #
filename=financebench_open_source.jsonl #difficult_questions.csv #


python ingest_data.py \
--db_name $db_name \
--filedir $filedir \
--filename $filename \
--read_processed \
--generate_metadata
