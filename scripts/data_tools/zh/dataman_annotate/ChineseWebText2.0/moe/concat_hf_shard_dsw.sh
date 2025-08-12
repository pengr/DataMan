#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss

# DSW Command, test_demo和test是一样的文件
FILES=("test")
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')
GPUS_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

for FILE in "${FILES[@]}"; do
    input_files=""
    for shard in 0; do
        input_files+="$DATA_DIR/DataMan/ChineseWebText2.0/annotated/all_rating/moe/${FILE}_annotated/${shard} "
    done

    python $CODE_DIR/data_tools/concat_hf_shard.py \
    --inputs $input_files \
    --output $DATA_DIR/DataMan/ChineseWebText2.0/annotated/all_rating/moe/${FILE}_annotated \
    --num_workers 64
done