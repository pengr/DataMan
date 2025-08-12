#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# DSW Command
FILES=("validation"         "test" )
for FILE in "${FILES[@]}"; do
    python $CODE_DIR/data_tools/data_utils.py \
    --input $DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/${FILE}_annotated \
    --output $DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating \
    --columns accuracy coherence language_consistency semantic_density \
             knowledge_novelty topic_focus creativity professionalism \
             style_consistency grammatical_diversity structural_standardization \
             originality sensitivity overall_score \
    --num_workers 128 \
    --strategy check_hf_columns
done
