#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate base

decoder_name_or_path=$1 # save_dir
dataset_path="$DATA_DIR/DataMan/ShareGPT" # `sft.json` and `test.json` are in this folder.
dataset_name="test"
output_path_to_store_samples="${dataset_name}/temp0.0_num1.json"
num_return_sequences="1"
temperature="0.0"
prompt_dict_path="$DATA_DIR/DataMan/ShareGPT/sft_prompt.json"

python $CODE_DIR/eval/instruction_tuning/inference.py \
    --task "run_inference" \
    --decoder_name_or_path $decoder_name_or_path \
    --num_return_sequences $num_return_sequences \
    --temperature $temperature \
    --per_device_batch_size 16 \
    --mixed_precision "bf16" \
    --tf32 True \
    --flash_attn True \
    --output_path $output_path_to_store_samples \
    --max_new_tokens 512 \
    --dataset_path $dataset_path \
    --dataset_name $dataset_name \
    --prompt_dict_path $prompt_dict_path
