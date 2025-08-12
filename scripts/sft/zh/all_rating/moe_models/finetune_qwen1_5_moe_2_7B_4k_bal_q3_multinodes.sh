#!/bin/bash
export NCCL_IB_TC=16
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_bond
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_MIN_NCHANNELS=4
export NCCL_NET_PLUGIN=none
export ACCL_C4_STATS_MODE=CONN
export ACCL_IB_SPLIT_DATA_NUM=4
export ACCL_IB_QPS_LOAD_BALANCE=1
export ACCL_IB_GID_INDEX_FIX=1
export ACCL_LOG_TIME=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========== Setup accl ========== #
rm -rf /usr/lib/x86_64-linux-gnu/libnccl.so*
cp -rf /mnt/nas/pengru.pr/data/libnccl.so.2 /root/anaconda3/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2
cp /mnt/nas/pengru.pr/data/libnccl.so.2 /usr/lib/x86_64-linux-gnu/
sudo ldconfig

#!/bin/bash
curdir=`dirname $0`
. $curdir/../../../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

set +e
set -x

# # Guide:
# # This script supports distributed training on multi-gpu workers (as well as single-worker training).
# # Please set the options below according to the comments.
# # For multi-gpu workers training, these options should be manually set for each worker.
# # After setting the options, please run the script on each worker.

# # Number of GPUs per GPU worker
# GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# # Number of GPU workers, for single-worker training, please set to 1
# NNODES=${NNODES:-1}

# # The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
# NODE_RANK=${NODE_RANK:-0}

# # The ip address of the rank-0 worker, for single-worker training, please set to localhost
# MASTER_ADDR=${MASTER_ADDR:-localhost}

# # The port for communication
# MASTER_PORT=${MASTER_PORT:-6001}

MODEL="$CKPT_DIR/base/Qwen1_5-MoE-A2_7B-Chat" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/finetune_bal_q3_train.jsonl"
EVAL_DATA="$DATA_DIR/DataMan/Qwen-sft/Qwen2-zh/finetune_bal_q3_valid.jsonl"
OUTPUT_DIR="$CKPT_DIR/DataMan/sft/zh/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q3-zh"
DS_CONFIG_PATH="$CODE_DIR/sft/ds_config_zero2.json"
USE_LORA=False
Q_LORA=False

function usage() {
    echo '
Usage: bash finetune/finetune_ds.sh [-m MODEL_PATH] [-d DATA_PATH] [--deepspeed DS_CONFIG_PATH] [--use_lora USE_LORA] [--q_lora Q_LORA]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        --deepspeed )
            shift
            DS_CONFIG_PATH=$1
            ;;
        --use_lora  )
            shift
            USE_LORA=$1
            ;;
        --q_lora    )
            shift
            Q_LORA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "
# Determine available number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then                   # 若CUDA_VISIBLE_DEVICES未设置或为空存在，使用nvidia-smi命令获取系统中GPU的数量
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")  # 否则根据CUDA_VISIBLE_DEVICES的值计算GPU数量
fi
num_gpus=${NUM_GPUS:-$num_gpus}
# Determine number of nodes
num_nodes=${WORLD_SIZE}
DISTRIBUTED_ARGS="
    --nproc_per_node $num_gpus \
    --nnodes $num_nodes \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
echo $DISTRIBUTED_ARGS

torchrun $DISTRIBUTED_ARGS $CODE_DIR/sft/finetune_qwen2.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to none \
    --model_max_length 2068 \
    --lazy_preprocess True \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --gradient_checkpointing \
    --seed 1024 \
    --deepspeed ${DS_CONFIG_PATH}
    # --run_name "Qwen1_5_moe_2_7B-4k_bal_q3-en" \
    # --save_total_limit 24 \