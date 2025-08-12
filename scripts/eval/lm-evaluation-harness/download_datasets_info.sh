#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# 数据集名称数组
datasets=(
    "allenai/ai2_arc"
    "allenai/sciq"
    "EleutherAI/logiqa"
    "google/boolq"
    "Rowan/hellaswag"
    "ybisk/piqa"
    "allenai/winogrande"
    "google-research-datasets/natural_questions"
    "lighteval/mmlu"
)

# 遍历每个模型
for dataset in "${datasets[@]}"; do
    local_dir="$CODE_DIR/eval/lm-evaluation-harness/lm_eval/datasets/$dataset"
    until huggingface-cli download --repo-type dataset --resume-download $dataset --local-dir $local_dir --local-dir-use-symlinks False --revision refs/convert/parquet; do
        echo "下载 $dataset 失败，正在重试..."
        sleep 5 # 等待5秒再次尝试
    done
    echo "下载 $dataset 完成"
done