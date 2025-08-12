#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# DSW Command
ARCHS=(
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-accuracy-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-coherence-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-creativity-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-grammatical_diversity-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-knowledge_novelty-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-language_consistency-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-originality-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-professionalism-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-semantic_density-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-sensitivity-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-structural_standardization-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-style_consistency-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-topic_focus-top_k-sampling"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-uniform-sample_with_overall_score1.0"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-uniform-sample_with_overall_score2.0"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-uniform-sample_with_overall_score3.0"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-uniform-sample_with_overall_score4.0"
"lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-uniform-sample_with_overall_score5.0"
)

# 获取 GPU 数量
num_gpus=${#ARCHS[@]}

# 创建一个函数来处理每个 arch
process_arch() {
    arch="$1"
    ckpt_dir="$CKPT_DIR/DataMan/pretrain/en/${arch}"
    linear_yml="$CODE_DIR/scripts/pretrain/linear.yml"
    temp_yml="$CODE_DIR/scripts/pretrain/tmp_${arch}.yml"
    
    ## 合并 checkpoint-merge 模型, 自动获取step数最大的3个checkpoint
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

    # Instruction Tuning: sft
    instruction_tuning_sft() {
        local output_dir="$1"
        local model_path="$2"
        bash "$CODE_DIR/scripts/eval/instruction_tuning/sft.sh" "$output_dir" "$model_path"
    }
    instruction_tuning_sft "$CKPT_DIR/DataMan/eval/en/instruction_tuning_ShareGPT/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}" "$merge_output"

    # Instruction Tuning: inference
    instruction_tuning_inference() {
        local decoder_name_or_path="$1"

        bash "$CODE_DIR/scripts/eval/instruction_tuning/inference.sh" "$decoder_name_or_path"
    }
    instruction_tuning_inference "$CKPT_DIR/DataMan/eval/en/instruction_tuning_ShareGPT/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}"
}

# 主循环
for (( i=0; i<${#ARCHS[@]}; i++ )); do
    arch="${ARCHS[$i]}"
    
    # 使每个arch在特定的GPU上运行
    CUDA_VISIBLE_DEVICES=$((i % num_gpus)) process_arch "$arch" &
done

# 等待所有后台进程完成
wait