#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# DSW Command
ARCHS=(
"lm-1.3B-select_60B_tokens_by-educational_value-sample_with_temperature2.0"
"lm-1.3B-select_60B_tokens_by-uniform-sampling"
)

# 获取 GPU 数量
num_gpus=${#ARCHS[@]}

# 创建一个函数来处理每个 arch
process_arch() {
    arch="$1"
    ckpt_dir="$CKPT_DIR/DataMan/pretrain/princeton-nlp/${arch}"
    linear_yml="$CODE_DIR/scripts/pretrain/linear.yml"
    temp_yml="$CODE_DIR/scripts/pretrain/tmp_${arch}.yml"
    
    ## 合并 checkpoint-merge 模型, 首先获取 step 数最大的 3 个 checkpoint
    top3_checkpoints=$(ls -d "$ckpt_dir/checkpoint-"* | grep -o '[0-9]\+' | sort -nr | head -n 3)
    IFS=$'\n' read -r -d '' -a steps <<< "$top3_checkpoints"
    merge_output="$ckpt_dir/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}"
    MODEL1="$ckpt_dir/checkpoint-${steps[0]}"
    MODEL2="$ckpt_dir/checkpoint-${steps[1]}"
    MODEL3="$ckpt_dir/checkpoint-${steps[2]}"
    
    # 生成填充后的配置文件
    sed -e "s#{{MODEL1}}#$MODEL1#g" \
        -e "s#{{MODEL2}}#$MODEL2#g" \
        -e "s#{{MODEL3}}#$MODEL3#g" \
        "$linear_yml" > "$temp_yml"
    # 运行 mergekit-yaml 命令, 合并模型, 然后删除临时创建的 yml
    mergekit-yaml "$temp_yml" "$merge_output"
    rm -f "$temp_yml"

    # ICL: lm-evaluation-harness
    icl_evaluation_harness() {
        local ckpt="$1"
        local out="$2"
        local bsz="$3"

        bash "$CODE_DIR/scripts/eval/lm-evaluation-harness/lm_eval_harness.sh" "$ckpt" "$out" "$bsz"
    }
    icl_evaluation_harness "$merge_output" "$CKPT_DIR/DataMan/eval/en/lm-evaluation-harness/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}" 16

    # PPL: validation and test
    eval_ppl() {
        local input="$1"
        local output="$2"
        local model="$3"
        local bsz="$4"
        local max_tokens="$5"

        bash "$CODE_DIR/scripts/eval/ppl/ppl.sh" "$input" "$output" "$model" "$bsz" "$max_tokens"
    }
    mkdir -p "$CKPT_DIR/DataMan/eval/en/validation_ppl/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}"
    eval_ppl "$DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/validation_annotated_success" \
    "$CKPT_DIR/DataMan/eval/en/validation_ppl/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}" \
    "$merge_output" 80 1000000 >> "$CKPT_DIR/DataMan/eval/en/validation_ppl/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}/result.log"

    mkdir -p "$CKPT_DIR/DataMan/eval/en/test_ppl/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}"
    eval_ppl "$DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/test_annotated_success" \
    "$CKPT_DIR/DataMan/eval/en/test_ppl/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}" \
    "$merge_output" 80 1000000 >> "$CKPT_DIR/DataMan/eval/en/test_ppl/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}/result.log"
}

# 主循环
for (( i=0; i<${#ARCHS[@]}; i++ )); do
    arch="${ARCHS[$i]}"
    
    # 使每个arch在特定的GPU上运行
    CUDA_VISIBLE_DEVICES=$((i % num_gpus)) process_arch "$arch" &
done

# 等待所有后台进程完成
wait