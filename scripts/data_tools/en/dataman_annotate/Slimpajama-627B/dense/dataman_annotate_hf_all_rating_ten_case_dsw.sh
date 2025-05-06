#!/bin/bash
curdir=`dirname $0`
. $curdir/../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

# DSW Command
CUDA_VISIBLE_DEVICES=0 python $CODE_DIR/data_tools/dataman_annotate_hf.py \
    --model_name_or_path $CKPT_DIR/DataMan/sft/en/all_rating/suppl_models/Qwen2-1_5B-4k_suppl_bal_q4-en/checkpoint-900 \
    --input_path $DATA_DIR/DataMan/ten_case/ten_wriqual_texts.json \
    --processed_path $DATA_DIR/DataMan/ten_case/ten_wriqual_texts_preprocessed \
    --inferenced_path $DATA_DIR/DataMan/ten_case/ten_wriqual_texts_annotated \
    --lang 'en' \
    --num_cpu_workers 10 \
    --num_gpu_workers 1 \
    --max_tokens 29 \
    --seed 1024 \
    --temperature 0.0 \
    --truncate_max_length 1894 \
    --char_truncate_max_length 20000 \
    --data_format slimpajama \
    --model_type all_rating \
    --use_fast \
    --batch_size 10000 \
    --inference_shard 1 0 \
    --json