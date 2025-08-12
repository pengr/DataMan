#!/bin/bash
curdir=`dirname $0`
. $curdir/../../../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

# All input files are unified into one output file
file1="$(cat "$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/success.jsonl" "$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/success_1.jsonl")"
file2="$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/finetune_bal_q.jsonl"

python $CODE_DIR/sft/prepare_data.py \
  --input_path "$file1" \
  --output_path "$file2" \
  --lang 'zh' \
  --qwen_version "qwen2_balance_q" \
  --sample_size 4000 \
  --test_size 100 \
  --valid_size 100 \
  --seed 1024 \
  --quantile 5 \
  --process_type prepare_qwen_request