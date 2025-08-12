#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# DSW Command
FILES=("validation"         "test" )
# Number of CPUs (physical cores）in all nodes
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')

for FILE in "${FILES[@]}"; do 
    python $CODE_DIR/data_tools/tokenize_dataset.py \
    --input $DATA_DIR/DataMan/Slimpajama-627B/org/${FILE}.jsonl \
    --output $DATA_DIR/DataMan/Slimpajama-627B/preprocessed/${FILE}_preprocessed \
    --tokenizer $CKPT_DIR/base/Sheared-LLaMA-1.3B \
    --use_fast \
    --input_field "text" \
    --input_tokens_field "input_ids" \
    --tokens_field "input_ids" \
    --length_field "length" \
    --max_length 1024 \
    --min_length 1024 \
    --num_workers 32 \
    --batch_size 128 \
    --overlap 0  \
    --json
done