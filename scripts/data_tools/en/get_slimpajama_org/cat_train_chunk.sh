#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# cascading chunks 1-10 of slimpajama's training set with a half-and-half way 
for CHUNK in {1..10}; do
    start=$(date +%s) 
    CHUNK_DIR="${DATA_DIR}/SlimPajama-627B/train/chunk${CHUNK}"
    
    # 查找文件中最大数字，确保找出范围内的最大文件
    MAX_NUM=$(ls ${CHUNK_DIR}/example_train_*.jsonl | sed -n 's/.*example_train_\([0-9]\+\)\.jsonl/\1/p' | sort -n | tail -1)
    
    # 第一部分合并文件
    FILES=$(seq 0 2999)
    OUTPUT_FILE="${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk${CHUNK}_half1.jsonl"
    # 遍历并合并文件
    for i in ${FILES}; do
        FILE="${CHUNK_DIR}/example_train_${i}.jsonl"
        if [[ -f "${FILE}" ]]; then
            cat "${FILE}" >> "${OUTPUT_FILE}"
        else
            echo "Warning: ${FILE} not found."
        fi
    done
    echo "Created ${OUTPUT_FILE}"

    # 第二部分合并文件
    FILES=$(seq 3000 ${MAX_NUM})
    OUTPUT_FILE="${DATA_DIR}/DataMan/Slimpajama-627B/org/train_chunk${CHUNK}_half2.jsonl"
    # 遍历并合并文件
    for i in ${FILES}; do
        FILE="${CHUNK_DIR}/example_train_${i}.jsonl"
        if [[ -f "${FILE}" ]]; then
            cat "${FILE}" >> "${OUTPUT_FILE}"
        else
            echo "Warning: ${FILE} not found."
        fi
    done
    echo "Created ${OUTPUT_FILE}"
    
    end=$(date +%s)
    duration=$((end - start))
    echo "Running time for chunk${CHUNK}: ${duration} seconds."
done
