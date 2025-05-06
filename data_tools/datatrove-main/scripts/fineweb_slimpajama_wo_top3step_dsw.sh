#!/bin/bash
curdir=`dirname $0`
. $curdir/vars
source $CONDA_DIR/bin/activate
conda activate dataman
export NLTK_DATA=xxxxxxxxxxxxxxx/nltk_data

FILES=(
    #"train_chunk1_half1" 
    #     "train_chunk1_half2" "train_chunk2_half1" "train_chunk2_half2" "train_chunk3_half1"
    #    "train_chunk3_half2" "train_chunk4_half1" "train_chunk4_half2" "train_chunk5_half1" "train_chunk5_half2" 
       "train_chunk6_half1" "train_chunk6_half2"
       # "train_chunk7_half1" "train_chunk7_half2" "train_chunk8_half1" 
    #    "train_chunk8_half2" "train_chunk9_half1" "train_chunk9_half2" "train_chunk10_half1" "train_chunk10_half2"
    #  "validation" "test"
)

for FILE in "${FILES[@]}"; do
    python $CODE_DIR/examples/fineweb_slimpajama_top3step.py \
    --input $DATA_DIR/SlimPajama-627B/qurating_tokenized_chunk/analyse/${FILE}_annotated_success \
    --output $DATA_DIR/SlimPajama-627B/fineweb_slimpajama_top3step/${FILE}
done