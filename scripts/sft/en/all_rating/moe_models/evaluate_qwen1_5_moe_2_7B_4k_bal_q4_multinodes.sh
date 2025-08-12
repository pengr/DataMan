#!/bin/bash
curdir=`dirname $0`
. $curdir/../../../../vars
source $CONDA_DIR/bin/activate
conda activate hss

export CUDA_DEVICE_MAX_CONNECTIONS=1

BEST_STEPS=($(python $CODE_DIR/sft/select_ckpt.py --json_file_path $CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en/trainer_state.json))

main_log="$CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en.log"
temp_log_dir=$(mktemp -d)  # 创建临时日志目录

max_parallel=8  # 最大并行任务数
current_jobs=0

i=0
for step in "${BEST_STEPS[@]}"; do
    # 为每个step分配GPU（轮询0-7）
    gpu_id=$((i % 8))
    temp_log="$temp_log_dir/step_${step}.log"
    
    # 启动后台任务
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        echo "Test Set!" >> "$temp_log"
        echo "Step: $step" >> "$temp_log"
        python $CODE_DIR/sft/evaluate_qwen2.py \
            --model_name_or_path "$CKPT_DIR/DataMan/sft/en/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q4-en/checkpoint-$step" \
            --input_path "$DATA_DIR/DataMan/Qwen-sft/Qwen2-en-suppl/final_finetune_bal_q4_test.jsonl" \
            --output_path "$temp_log" \
            --max_tokens 29 --seed 1024 --model_type all_rating 2>&1  # 合并stderr到输出
    ) &
    
    # 控制并行度
    ((i++))
    ((current_jobs++))
    if (( current_jobs >= max_parallel )); then
        wait -n  # 等待任意一个任务完成
        ((current_jobs--))
    fi
done

wait  # 等待剩余任务完成

# 按原始顺序合并临时日志到主日志
for step in "${BEST_STEPS[@]}"; do
    cat "$temp_log_dir/step_${step}.log" >> "$main_log"
    rm "$temp_log_dir/step_${step}.log"
done

rmdir "$temp_log_dir"  # 清理临时目录
