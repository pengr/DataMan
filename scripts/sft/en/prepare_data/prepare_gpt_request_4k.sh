#!/bin/bash
curdir=`dirname $0`
. $curdir/../../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

for file in $DATA_DIR/Qwen-Org/en*.jsonl; do
  sourcename=$(basename "$file" .jsonl)
  file1="$DATA_DIR/DataMan/Qwen-sft/Qwen2-en/${sourcename}_request.jsonl"
  python $CODE_DIR/sft/prepare_data.py \
    --input_path "$file" \
    --output_path "$file1" \
    --lang 'en' \
    --gpt_model "gpt-4-0125-preview" \
    --truncate \
    --seed 1024 \
    --trucate_max_length 1894 \
    --sample_size 4000 \
    --process_type prepare_gpt_request
done