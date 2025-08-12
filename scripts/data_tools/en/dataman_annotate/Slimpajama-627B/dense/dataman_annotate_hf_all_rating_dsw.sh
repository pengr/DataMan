#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# DSW Command
# FILES=("train_chunk1_half1" "train_chunk1_half2" "train_chunk2_half1" "train_chunk2_half2" "train_chunk3_half1"
#        "train_chunk3_half2" "train_chunk4_half1" "train_chunk4_half2" "train_chunk5_half1" "train_chunk5_half2" 
#        "train_chunk6_half1" "train_chunk6_half2" "train_chunk7_half1" "train_chunk7_half2" "train_chunk8_half1" 
#        "train_chunk8_half2" "train_chunk9_half1" "train_chunk9_half2" "train_chunk10_half1" "train_chunk10_half2"
#        "validation"         "test" )
FILES=("validation"         "test")
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')
GPUS_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

for FILE in "${FILES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python $CODE_DIR/data_tools/dataman_annotate_hf.py \
        --model_name_or_path $CKPT_DIR/DataMan/sft/en/all_rating/suppl_models/Qwen2-1_5B-4k_suppl_bal_q4-en/checkpoint-900 \
        --input_path $DATA_DIR/DataMan/SlimPajama-627B/org/${FILE}.jsonl \
        --processed_path $DATA_DIR/DataMan/Slimpajama-627B/preprocessed/${FILE}_preprocessed \
        --inferenced_path $DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/${FILE}_annotated \
        --lang 'en' \
        --num_cpu_workers 128 \
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
        --inference_shard 2 0 & 
    CUDA_VISIBLE_DEVICES=0 python $CODE_DIR/data_tools/dataman_annotate_hf.py \
        --model_name_or_path $CKPT_DIR/DataMan/sft/en/all_rating/suppl_models/Qwen2-1_5B-4k_suppl_bal_q4-en/checkpoint-900 \
        --input_path $DATA_DIR/DataMan/SlimPajama-627B/org/${FILE}.jsonl \
        --processed_path $DATA_DIR/DataMan/Slimpajama-627B/preprocessed/${FILE}_preprocessed \
        --inferenced_path $DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/${FILE}_annotated \
        --lang 'en' \
        --num_cpu_workers 128 \
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
        --inference_shard 2 1 & 
    wait
done