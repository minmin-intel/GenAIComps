# data
filedir=$WORKDIR/financebench/data/ #$WORKDIR/datasets/financebench/ #
filename=financebench_open_source.jsonl #difficult_questions.csv #
db_name=test_3M_table_store

python make_table_store.py \
    --filedir $filedir \
    --filename $filename \
    --db_name $db_name