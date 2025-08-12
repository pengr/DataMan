#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss

# DSW Command, test_demo和test是一样的文件
FILES=("test")
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')
GPUS_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

for FILE in "${FILES[@]}"; do
    python $CODE_DIR/data_tools/ppl.py \
        --input $DATA_DIR/DataMan/ChineseWebText2.0/preprocessed/${FILE}_preprocessed \
        --output $DATA_DIR/DataMan/ChineseWebText2.0/ppl/${FILE}_ppl \
        --model /cpfs01/user/fengxiao.zjy/Qwen2.5-3B/Qwen/Qwen2___5-3B \
        --batch_size 32 \
        --save_workers 32 \
        --bf16 \
        --max_tokens 1000000 \
        --text_field 'text' \
        --tokens_field 'input_ids' \
        --field 'ppl' \
        --shard 1 0
done