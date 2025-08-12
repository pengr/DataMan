#!/bin/bash
curdir=`dirname $0`
. $curdir/../../../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

# 可以选择直接设置一个超大的值，只有它就会用切分完验证和测试后的所有行
python $CODE_DIR/sft/prepare_data.py \
  --input_path "$DATA_DIR/DataMan/Qwen-sft/Qwen2-en-suppl/suppl_success.jsonl" \
  --output_path "$DATA_DIR/DataMan/Qwen-sft/Qwen2-en-suppl/finetune_suppl_bal_q.jsonl" \
  --lang 'en' \
  --qwen_version "qwen2_balance_q" \
  --sample_size 100000 \
  --test_size 100 \
  --valid_size 100 \
  --seed 1024 \
  --quantile 5 \
  --process_type prepare_qwen_request