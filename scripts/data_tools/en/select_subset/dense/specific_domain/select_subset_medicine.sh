#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

input_files=""
for chunk in {1..10}; do
    for half in {1..2}; do
        input_files+="$DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk${chunk}_half${half}_annotated_success "
    done
done

# DSW Command: 新版本：随机采样按照source_domain占比来采样
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')
# 由于无shard 选项，若想将30B数据存储为单个shard时，因此tokens表面是30B，实际由于margin会更大一些。
# 所以需要将 tokens_per_shard 设置大于 tokens 才可以一次性覆盖这些额外的 tokens。
python $CODE_DIR/data_tools/select_subset.py \
--inputs $input_files \
--output $DATA_DIR/DataMan/Slimpajama-627B/data_filtering/medicine/train_filtered \
--metric_field overall_score \
--seq_len_field length \
--domain_field application_domain \
--tokens 10_000_000_000_000 \
--tokens_per_shard 10_000_000_000_000 \
--margin 0.1 \
--seed 42 \
--num_workers 128 \
--target_scores "4" "5" \
--target_domains "A" \
--strategy "None"