#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# DSW Command, test_demo和test是一样的文件
FILES=("test" )  
# Number of CPUs (physical cores）in all nodes
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')

for FILE in "${FILES[@]}"; do 
    python $CODE_DIR/data_tools/tokenize_dataset.py \
    --input /mnt/nas/zhangjinyang/database/cpt/ChineseWebText2.0/ChineseWebText2.0/${FILE}.jsonl \
    --output $DATA_DIR/DataMan/ChineseWebText2.0/preprocessed/${FILE}_preprocessed \
    --tokenizer $CKPT_DIR/base/Qwen2-1_5B \
    --use_fast \
    --input_field "text" \
    --input_tokens_field "input_ids" \
    --tokens_field "input_ids" \
    --length_field "length" \
    --max_length 1024 \
    --min_length 1024 \
    --num_workers 128 \
    --batch_size 4096 \
    --overlap 0  \
    --json
done