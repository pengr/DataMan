#!/bin/bash
curdir=`dirname $0`
. $curdir/../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

CUDA_VISIBLE_DEVICES=1 python $CODE_DIR/data_tools/dataman_annotate.py \
    --model_name_or_path $CKPT_DIR/DataMan/sft/en/all_rating/suppl_models/Qwen2-1_5B-4k_suppl_bal_q4-en/checkpoint-900 \
    --input_path $DATA_DIR/DataMan/RedPajama-Data-1T/en_head_0000.json.gz.dedup.classifier.norm.slimpajama_filter.jsonl \
    --processed_path $DATA_DIR/DataMan/RedPajama-Data-1T/en_head_0000.json.gz.dedup.classifier.norm.slimpajama_filter.preprocessed.jsonl \
    --inferenced_path $DATA_DIR/DataMan/RedPajama-Data-1T/en_head_0000.json.gz.dedup.classifier.norm.slimpajama_filter.annotated.jsonl \
    --lang 'en' \
    --num_cpu_workers 128 \
    --num_gpu_workers 1 \
    --max_tokens 29 \
    --seed 1024 \
    --temperature 0.0 \
    --truncate_max_length 1894 \
    --char_truncate_max_length 20000 \
    --data_format commoncrawl \
    --model_type all_rating \
    --batch_size 10000

CUDA_VISIBLE_DEVICES=1 python $CODE_DIR/data_tools/dataman_annotate.py \
    --model_name_or_path $CKPT_DIR/DataMan/sft/en/all_rating/suppl_models/Qwen2-1_5B-4k_suppl_bal_q4-en/checkpoint-900 \
    --input_path $DATA_DIR/DataMan/RedPajama-Data-1T/en_head_0001.json.gz.dedup.classifier.norm.slimpajama_filter.jsonl \
    --processed_path $DATA_DIR/DataMan/RedPajama-Data-1T/en_head_0001.json.gz.dedup.classifier.norm.slimpajama_filter.preprocessed.jsonl \
    --inferenced_path $DATA_DIR/DataMan/RedPajama-Data-1T/en_head_0001.json.gz.dedup.classifier.norm.slimpajama_filter.annotated.jsonl \
    --lang 'en' \
    --num_cpu_workers 128 \
    --num_gpu_workers 1 \
    --max_tokens 29 \
    --seed 1024 \
    --temperature 0.0 \
    --truncate_max_length 1894 \
    --char_truncate_max_length 20000 \
    --data_format commoncrawl \
    --model_type all_rating \
    --batch_size 10000