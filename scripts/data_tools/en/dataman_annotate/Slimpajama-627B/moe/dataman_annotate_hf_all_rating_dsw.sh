#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss

# DSW Command
FILES=("test" "validation")
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')
GPUS_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

for FILE in "${FILES[@]}"; do
    CUDA_VISIBLE_DEVICES=$SHARD python $CODE_DIR/data_tools/dataman_annotate_hf.py \
    --model_name_or_path $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q2-en/checkpoint-1400 \
    --input_path $DATA_DIR/DataMan/Slimpajama-627B/org/${FILE}.jsonl \
    --processed_path $DATA_DIR/DataMan/Slimpajama-627B/preprocessed/${FILE}_preprocessed \
    --inferenced_path $DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/moe/${FILE}_annotated \
    --lang 'en' \
    --num_cpu_workers 16 \
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
    --inference_shard 1 0
done
