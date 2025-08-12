#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss

# DSW Command, test_demo和test是一样的文件
FILES=("test")
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')
GPUS_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

for FILE in "${FILES[@]}"; do
    python $CODE_DIR/data_tools/data_utils.py \
    --input $DATA_DIR/DataMan/ChineseWebText2.0/annotated/all_rating/moe/${FILE}_annotated \
    --output $DATA_DIR/DataMan/ChineseWebText2.0/annotated/all_rating/moe \
    --columns accuracy \
              coherence \
              language_consistency \
              semantic_density \
              knowledge_novelty \
              topic_focus \
              creativity \
              professionalism \
              style_consistency \
              grammatical_diversity \
              structural_standardization \
              originality \
              sensitivity \
              overall_score \
    --num_workers 64 \
    --strategy check_hf_columns
done
