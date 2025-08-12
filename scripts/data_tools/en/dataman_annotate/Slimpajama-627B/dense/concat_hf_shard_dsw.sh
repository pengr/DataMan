#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# DSW Command
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')
FILES=("test" "validation")
for FILE in "${FILES[@]}"; do
    input_files=""
    for shard in {0..1}; do
        input_files+="$DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/${FILE}_annotated/${shard} "
    done
    
    python $CODE_DIR/data_tools/concat_hf_shard.py \
    --inputs $input_files \
    --output $DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/${FILE}_annotated \
    --num_workers 64
done