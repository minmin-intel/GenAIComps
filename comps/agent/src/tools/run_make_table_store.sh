# data
filedir=$WORKDIR/financebench/data/ #$WORKDIR/datasets/financebench/ #
filename=financebench_open_source.jsonl #difficult_questions.csv #


python make_table_store.py \
    --filedir $filedir \
    --filename $filename \