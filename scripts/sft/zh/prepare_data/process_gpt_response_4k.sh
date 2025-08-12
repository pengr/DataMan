#!/bin/bash
curdir=`dirname $0`
. $curdir/../../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

## first attempt
for file in $DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/zh_*_response.jsonl; do
  sourcename=$(basename "$file" .jsonl)
  # Each input file has a corresponding output file
  file1="$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/${sourcename/request/failed}.jsonl"
  # All input files are unified into one output file
  file1="$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/failed.jsonl"

  python $CODE_DIR/sft/prepare_data.py \
    --input_path "$file" \
    --output_path "$file1" \
    --lang 'zh' \
    --gpt_model "gpt-4-1106-preview" \
    --seed 1024 \
    --process_type process_gpt_response
done

# ## second attempt
# for file in $DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/re_response.jsonl; do
#   sourcename=$(basename "$file" .jsonl)
#   # Each input file has a corresponding output file
#   # file1="$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/${sourcename/request/failed}.jsonl"
#   # All input files are unified into one output file
#   file1="$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/re_response_failed.jsonl"

#   python $CODE_DIR/sft/prepare_data.py \
#     --input_path "$file" \
#     --output_path "$file1" \
#     --lang 'zh' \
#     --gpt_model "gpt-4-1106-preview" \
#     --seed 1024 \
#     --process_type process_gpt_response
# done

# # ## third attempt
# for file in $DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/re_response_re_response.jsonl; do
#   sourcename=$(basename "$file" .jsonl)
#   # Each input file has a corresponding output file
#   # file1="$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/${sourcename/request/failed}.jsonl"
#   # All input files are unified into one output file
#   file1="$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/re_response_re_response_failed.jsonl"

#   python $CODE_DIR/sft/prepare_data.py \
#     --input_path "$file" \
#     --output_path "$file1" \
#     --lang 'zh' \
#     --gpt_model "gpt-4-1106-preview" \
#     --seed 1024 \
#     --process_type process_gpt_response
# done