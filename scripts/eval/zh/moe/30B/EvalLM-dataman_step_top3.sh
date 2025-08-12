#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# 配置路径和文件名
arch=$1
ckpt_dir="$CKPT_DIR/DataMan/pretrain/zh/moe/${arch}"
linear_yml="$CODE_DIR/scripts/pretrain/linear.yml"
temp_yml="$CODE_DIR/scripts/pretrain/tmp_${arch}.yml"

# 合并 checkpoint-merge 模型, 自动获取step数最大的3个checkpoint
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

    bash "$CODE_DIR/scripts/eval/lm-evaluation-harness/lm_eval_harness_zh.sh" "$ckpt" "$out" "$bsz"
}
icl_evaluation_harness "$merge_output" "$CKPT_DIR/DataMan/eval/zh/lm-evaluation-harness/moe/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}" 

# PPL: validation and test
eval_ppl() {
    local input="$1"
    local output="$2"
    local model="$3"
    local bsz="$4"
    local max_tokens="$5"
    local domain_field="$6"
    
    bash "$CODE_DIR/scripts/eval/ppl/ppl.sh" "$input" "$output" "$model" "$bsz" "$max_tokens" "$domain_field"
}
mkdir -p "$CKPT_DIR/DataMan/eval/zh/test_ppl/moe/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}"
log_file="$CKPT_DIR/DataMan/eval/zh/test_ppl/moe/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}/result.log"
eval_ppl "$DATA_DIR/DataMan/ChineseWebText2.0/annotated/all_rating/moe/test_annotated" \
"$CKPT_DIR/DataMan/eval/zh/test_ppl/moe/${arch}/checkpoint-merge-${steps[0]}-${steps[1]}-${steps[2]}" \
"$merge_output" 40 1000000 "single_label" > >(tee -a "$log_file") 