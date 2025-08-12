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

# ========== Bash========== #
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss

# > Default arguments - can be overriden by environment variables:
# architecture to train, must be compatible with the Llama architecture
arch=${ARCH:-Qwen/Qwen2.5-1.5B}
# total batch size across all devices with gradient accumulation, <修改代码>，单节点跑不下来bsz16
bsz=${BSZ:-2048}
# number of sequences per device 
seq=${SEQ:-8}
# peak learning rate
lr=${LR:-5e-4}
# number of epochs
epochs=${EPOCHS:-1}
# warmup ratio
warmup=${WARMUP:-0.05}
# save model every n steps
save_steps=${SAVE:-1000}
# path to dataset to train on
dataset=${DATASET:-"$DATA_DIR/DataMan/ChineseWebText2.0/data_filtering/moe/style_consistency/train_filtered/0 $DATA_DIR/DataMan/ChineseWebText2.0/data_filtering/moe/style_consistency/train_filtered/1 $DATA_DIR/DataMan/ChineseWebText2.0/data_filtering/moe/style_consistency/train_filtered/2 $DATA_DIR/DataMan/ChineseWebText2.0/data_filtering/moe/style_consistency/train_filtered/3 $DATA_DIR/DataMan/ChineseWebText2.0/data_filtering/moe/style_consistency/train_filtered/4 $DATA_DIR/DataMan/ChineseWebText2.0/data_filtering/moe/style_consistency/train_filtered/5 $DATA_DIR/DataMan/ChineseWebText2.0/data_filtering/moe/style_consistency/train_filtered/6 $DATA_DIR/DataMan/ChineseWebText2.0/data_filtering/moe/style_consistency/train_filtered/7 $DATA_DIR/DataMan/ChineseWebText2.0/data_filtering/moe/style_consistency/train_filtered/8 $DATA_DIR/DataMan/ChineseWebText2.0/data_filtering/moe/style_consistency/train_filtered/9"}
# suffix to append to run name
suffix=${SUFFIX:-"style_consistency-top_k-sampling"}
# name of training session
run_name="lm-$(basename $arch)-select_30B_tokens_by-${suffix}"
out_dir="$CKPT_DIR/DataMan/pretrain/zh/moe/$run_name"
mkdir -p $out_dir

# Determine available number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then                   # 若CUDA_VISIBLE_DEVICES未设置或为空存在，使用nvidia-smi命令获取系统中GPU的数量
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")  # 否则根据CUDA_VISIBLE_DEVICES的值计算GPU数量
fi
num_gpus=${NUM_GPUS:-$num_gpus}
# Determine number of nodes
num_nodes=${WORLD_SIZE}

# <修改代码>，设置多/单（通用）节点的运行命令header，调用torchrun进行训练, PyTorch环境变量：$MASTER_ADDR，$MASTER_PORT，$WORLD_SIZE，$RANK 
DISTRIBUTED_ARGS="
    --nproc_per_node $num_gpus \
    --nnodes $num_nodes \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

export OMP_NUM_THREADS=$num_gpus  # 用途：这个变量指定了由OpenMP使用的线程数量。
export WANDB_PROJECT="dataman"    # 用途：这个变量指定了Weights & Biases（W&B）项目的名称为"dataman"
export WANDB_DIR=$out_dir         # 用途：这个变量指定了W&B运行的目录。
export WANDB_MODE="offline"       # 用途：这个变量控制W&B的运行模式。
export FSDP_SHARDING_STRATEGY="5"  # 用途：指定Fully Sharded Data Parallel（FSDP）的分片策略。选择分片策略，5对应于_hybrid_shard_zero2，这是一种特定的分片策略，结合了设备内和跨设备的张量切片，提升了内存和计算效率。
export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"  # 用途：决定FSDP状态字典的类型。作用：FULL_STATE_DICT指定保存和加载时使用完整状态字典。这对于确保模型的一致性和可恢复性尤其重要。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

base_arguments=(
    --report_to wandb                    # 设置将训练报告发送到wandb
    --do_train                           # 设置进行训练过程
    --model_name_or_path "/cpfs01/user/pengru.pr/base/qwen/Qwen2___5-1___5B" # <修改代码>，设置模型路径
    # --resume_from_checkpoint "$CKPT_DIR/DataMan/pretrain/zh/moe/lm-Qwen2.5-1.5B-select_30B_tokens_by-style_consistency-top_k-sampling"
    # --resume_from_checkpoint "$CKPT_DIR/DataMan/pretrain/zh/moe/lm-Qwen2.5-1.5B-select_30B_tokens_by-grammatical_diversity-top_k-sampling"
    --config_name "/cpfs01/user/pengru.pr/base/qwen/Qwen2___5-1___5B"                # <修改代码>，注释掉arch变量指定的模型配置名称，因为无法访问hf下载
    --config_overrides ""                # 没有配置覆盖
    # --tokenizer_name $arch             # <修改代码>，注释掉tokenizer变量，因为无法访问hf下载
    --tokenizer_name "/cpfs01/user/pengru.pr/base/qwen/Qwen2___5-1___5B"    # <修改代码>，设置tokenizer的路径
    --run_name $run_name                 # 设置运行名称
    --output_dir $out_dir                # 指定输出目录为$out_dir
    --log_level info                     # 设置日志级别为info
    --logging_steps 1                    # 设置日志记录的步数为1
    --disable_tqdm true                  # 禁用tqdm进度条
    --save_steps $save_steps             # 设置保存模型的步数间隔
    --cache_dir .cache                   # 设置缓存目录
    --overwrite_output_dir               # 允许覆盖输出目录
    --dataloader_num_workers 8           # 设置数据加载器的工作线程数为8
    --num_train_epochs $epochs           # 设置训练的总轮数
    --per_device_train_batch_size $seq   # 设置每个设备处理的序列数
    --gradient_accumulation_steps $(($bsz / $seq / $num_gpus / $num_nodes))  # 设置梯度累积步数
    --learning_rate $lr                  # 设置学习率
    --lr_scheduler_type cosine           # 设置学习率调度类型为cosine
    --min_lr_ratio 0.1                   # 设置最小学习率比例
    --max_grad_norm 1.0                  # 设置最大梯度范数
    --adam_beta1 0.9                     # 设置Adam优化器的beta1参数
    --adam_beta2 0.95                    # 设置Adam优化器的beta2参数
    --weight_decay 0.1                   # 设置权重衰减
    --warmup_ratio $warmup               # 设置预热比例
    #--optim adamw_torch_fused           # 未设置优化器为adamw_torch_fused
    --bf16                               # 使用bfloat16精度
    --bf16_full_eval                     # 在评估时使用bfloat16精度
    --fsdp auto_wrap                     # 设置Fully Sharded Data Parallel为auto_wrap模式
    # --fsdp "auto_wrap full_shard"  # 明确指定sharding策略
    # --fsdp_config "/mnt/nas/pengru.pr/DataMan/scripts/pretrain/en/moe/fsdp_config.json"  # 添加FSDP配置文件
    --ddp_find_unused_parameters false   # 设置DDP不寻找未使用的参数
    # Depending on model size and sequence length, gradient checkpointing might result in higher throughput
    # --gradient_checkpointing_kwargs '{"use_reentrant": false}'  # 添加这个参数
    #--gradient_checkpointing            # 未根据模型大小和序列长度，梯度检查点可能会提高吞吐量
    --tokenized_train_dataset $dataset   # 指定训练数据集
    $@                                   # 传递剩余其他命令行参数
)

echo command: "${DISTRIBUTED_ARGS} ${base_arguments[@]}"  # 打印最终运行命令，用于调试
torchrun $DISTRIBUTED_ARGS $CODE_DIR/pretrain/train_language_model.py \
"${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out     # 执行生成的命令，并将标准输出和错误输出同时重定向到日志文件