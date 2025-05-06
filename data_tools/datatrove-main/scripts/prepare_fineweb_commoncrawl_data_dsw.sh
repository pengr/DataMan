#!/bin/bash
curdir=`dirname $0`
. $curdir/vars
source $CONDA_DIR/bin/activate
conda activate dataman

python $CODE_DIR/examples/prepare_fineweb_slimpajama_data.py \
    --input $DATA_DIR/RedPajama-Data-1T/en_head_0000.json.gz.dedup.classifier.norm.fineweb/base_processing/output/out/*.jsonl \
    --output $DATA_DIR/RedPajama-Data-1T/en_head_0000.json.gz.dedup.classifier.norm.fineweb.jsonl \
    --num_workers 64 \
    --json

python $CODE_DIR/examples/prepare_fineweb_slimpajama_data.py \
    --input $DATA_DIR/RedPajama-Data-1T/en_head_0001.json.gz.dedup.classifier.norm.fineweb/base_processing/output/out/*.jsonl \
    --output $DATA_DIR/RedPajama-Data-1T/en_head_0001.json.gz.dedup.classifier.norm.fineweb.jsonl \
    --num_workers 64 \
    --json