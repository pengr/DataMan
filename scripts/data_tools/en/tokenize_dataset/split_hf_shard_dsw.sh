#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# DSW Command
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')
FILES=("train_chunk1_half1" "train_chunk1_half2" "train_chunk2_half1" "train_chunk2_half2" "train_chunk3_half1"
       "train_chunk3_half2" "train_chunk4_half1" "train_chunk4_half2" "train_chunk5_half1" "train_chunk5_half2" 
       "train_chunk6_half1" "train_chunk6_half2" "train_chunk7_half1" "train_chunk7_half2" "train_chunk8_half1" 
       "train_chunk8_half2" "train_chunk9_half1" "train_chunk9_half2" "train_chunk10_half1" "train_chunk10_half2"
       "validation"         "test" )

for FILE in "${FILES[@]}"; do
    python $CODE_DIR/data_tools/split_hf_shard.py \
    --input $DATA_DIR/DataMan/Slimpajama-627B/preprocessed/${FILE}_preprocessed \
    --output $DATA_DIR/DataMan/Slimpajama-627B/preprocessed \
    --num_workers 64 \
    --num_shards 2
done