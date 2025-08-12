#!/bin/bash
curdir=`dirname $0`
. $curdir/../../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

for file in $DATA_DIR/Qwen-Org/zh*.jsonl; do
  sourcename=$(basename "$file" .jsonl)
  file1="$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/${sourcename}_request.jsonl"
  python $CODE_DIR/sft/prepare_data.py \
    --input_path "$file" \
    --output_path "$file1" \
    --lang 'zh' \
    --gpt_model "gpt-4-1106-preview" \
    --truncate \
    --trucate_max_length 1896 \
    --sample_size 2000 \
    --seed 1024 \
    --process_type prepare_gpt_request
done