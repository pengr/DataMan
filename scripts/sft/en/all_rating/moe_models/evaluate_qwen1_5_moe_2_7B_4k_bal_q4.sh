#!/bin/bash
curdir=`dirname $0`
. $curdir/../../../../vars
source $CONDA_DIR/bin/activate
conda activate hss

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0

BEST_STEPS=($(python $CODE_DIR/sft/select_ckpt.py --json_file_path $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en/trainer_state.json))

# Valid Set
# for step in "${BEST_STEPS[@]}"; do
#     echo "Valid Set!" >> $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en.log
#     echo "Step: $step" >> $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en.log
#     python $CODE_DIR/sft/evaluate_qwen2.py --model_name_or_path $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en/checkpoint-$step \
#     --input_path $DATA_DIR/DataMan/Qwen-sft/Qwen2-en-suppl/final_finetune_bal_q4_valid.jsonl \
#     --output_path $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en.log \
#     --max_tokens 29 --seed 1024 --model_type all_rating
# done

# Test Set
for step in 1300; do
    echo "Test Set!" >> $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en.log
    echo "Step: $step" >> $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en.log
    python $CODE_DIR/sft/evaluate_qwen2.py --model_name_or_path $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en/checkpoint-$step \
    --input_path $DATA_DIR/DataMan/Qwen-sft/Qwen2-en-suppl/final_finetune_bal_q4_test.jsonl \
    --output_path $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en.log \
    --max_tokens 29 --seed 1024 --model_type all_rating
done

# Test Set，without VLLM
# for step in 1300; do
#     echo "Test Set!" >> $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en.log
#     echo "Step: $step" >> $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en.log
#     python $CODE_DIR/sft/evaluate_qwen2_hf.py --model_name_or_path $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en/checkpoint-$step \
#     --input_path $DATA_DIR/DataMan/Qwen-sft/Qwen2-en-suppl/final_finetune_bal_q4_test.jsonl \
#     --output_path $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en.log \
#     --max_tokens 29 --seed 1024 --model_type all_rating
# done