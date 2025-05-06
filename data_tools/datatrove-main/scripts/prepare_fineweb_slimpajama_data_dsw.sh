#!/bin/bash
curdir=`dirname $0`
. $curdir/vars
source $CONDA_DIR/bin/activate
conda activate dataman

FILES=(
    "train_chunk1_half1" "train_chunk1_half2" "train_chunk2_half1" "train_chunk2_half2" "train_chunk3_half1"
    "train_chunk3_half2" "train_chunk4_half1" "train_chunk4_half2" "train_chunk5_half1" "train_chunk5_half2" 
    "train_chunk6_half1" "train_chunk6_half2" "train_chunk7_half1" "train_chunk7_half2" "train_chunk8_half1" 
    "train_chunk8_half2" "train_chunk9_half1" "train_chunk9_half2" "train_chunk10_half1" "train_chunk10_half2"
)

python $CODE_DIR/examples/prepare_fineweb_slimpajama_data.py \
    --input $DATA_DIR/SlimPajama-627B/fineweb_slimpajama_top3step/train_chunk1_half1/minhash/out/deduped_output/*.jsonl \
    --output $DATA_DIR/SlimPajama-627B/fineweb_slimpajama_top3step/train_chunk1_half1/huggingface \
    --num_workers 64 \
    --columns_to_remove 'id' 'dump' 'dataset' 'file_path' 'token_count'
