#!/bin/bash
curdir=`dirname $0`
. $curdir/../../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

## first attempt
for file in $DATA_DIR/DataMan/Qwen-sft/Qwen2-en-suppl/*_response.jsonl; do
  sourcename=$(basename "$file" .jsonl)
  file1="$DATA_DIR/DataMan/Qwen-sft/Qwen2-en/${sourcename}_failed.jsonl"
  python $CODE_DIR/sft/prepare_data.py \
    --input_path "$file" \
    --output_path "$file1" \
    --lang 'en' \
    --gpt_model "gpt-4-1106-preview" \
    --seed 1024 \
    --process_type process_gpt_response
done

## second attempt
for file in $DATA_DIR/DataMan/Qwen-sft/Qwen2-en-suppl/*_response.jsonl; do
  sourcename=$(basename "$file" .jsonl)
  file1="$DATA_DIR/DataMan/Qwen-sft/Qwen2-en/${sourcename}_failed.jsonl"
  python $CODE_DIR/sft/prepare_data.py \
    --input_path "$file" \
    --output_path "$file1" \
    --lang 'en' \
    --gpt_model "gpt-4-1106-preview" \
    --seed 1024 \
    --process_type process_gpt_response
done