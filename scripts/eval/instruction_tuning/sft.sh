#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate base

export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# An example for 1 nodes, each uses 8 GPUs
output_dir=$1 # save_dir
run_name="sft10K"
model_name_or_path=$2
dataset_path="$DATA_DIR/DataMan/ShareGPT" # `sft.json` and `test.json` are in this folder.
epoch=2
dataset_name="sft"
prompt_dict_path="$DATA_DIR/DataMan/ShareGPT/sft_prompt.json"

export WANDB_MODE=offline
torchrun $DISTRIBUTED_ARGS $CODE_DIR/eval/instruction_tuning/supervised.py \
  --model_name_or_path "${model_name_or_path}" \
  --eval_size 10 \
  --fp16 False \
  --bf16 True \
  --seed 42 \
  --output_dir "${output_dir}" \
  --dataset_path "${dataset_path}" \
  --dataset_name "${dataset_name}" \
  --prompt_dict_path "${prompt_dict_path}" \
  --num_train_epochs "${epoch}" \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --eval_steps 100 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "ShareGPT" \
  --run_name "${run_name}" \
  --tf32 True \
  --flash_attn True \
  --model_max_length 512 \
  --ddp_timeout 1800 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"