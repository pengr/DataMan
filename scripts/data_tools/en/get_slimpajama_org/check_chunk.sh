#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# 定义输入文件的数组
input_files=(
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/test.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/validation.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk1_half1.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk1_half2.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk2_half1.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk2_half2.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk3_half1.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk3_half2.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk4_half1.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk4_half2.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk5_half1.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk5_half2.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk6_half1.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk6_half2.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk7_half1.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk7_half2.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk8_half1.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk8_half2.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk9_half1.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk9_half2.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk10_half1.jsonl"
    "${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk10_half2.jsonl"
)

# 循环检查每个文件
for INPUT_FILE in "${input_files[@]}"; do
    python ${CODE_DIR}/data_tools/data_utils.py --input "$INPUT_FILE" --strategy 'check_json'
    echo "${INPUT_FILE} passes the check"
done