#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# cating chunks 1-5 of slimpajama's test set 
OUTPUT_FILE="${DATA_DIR}/DataMan/Slimpajama-627B/org/test.jsonl"
for CHUNK in $(seq 1 5); do
    start=$(date +%s) 
    CHUNK_DIR="${DATA_DIR}/SlimPajama-627B/test/chunk${CHUNK}"

    # 使用 find 查找文件并用 sort -V 自然排序
    for FILE in $(find "$CHUNK_DIR" -name "example_holdout_*.jsonl" | sort -V); do
        cat "$FILE" >> "$OUTPUT_FILE"
    done

    end=$(date +%s)
    duration=$((end - start))
    echo "Running time for chunk${CHUNK}: ${duration} seconds."
done
echo "Created ${OUTPUT_FILE}"
