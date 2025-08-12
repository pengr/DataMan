#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss

input_files=""
for chunk in {1..151}; do
    formatted_chunk=$(printf "%04d" $chunk)
    if [[ $formatted_chunk == "0047" || $formatted_chunk == "0144" ]]; then
        continue
    fi
    input_files+="$DATA_DIR/DataMan/ChineseWebText2.0/annotated/all_rating/moe/${formatted_chunk}_annotated_success "
done

# DSW Command: 新版本：随机采样按照source_domain占比来采样
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')
# 由于无shard 选项，若想将30B数据存储为单个shard时，因此tokens表面是30B，实际由于margin会更大一些。
# 所以需要将 tokens_per_shard 设置大于 tokens 才可以一次性覆盖这些额外的 tokens。
python $CODE_DIR/data_tools/select_subset.py \
--inputs $input_files \
--output $DATA_DIR/DataMan/ChineseWebText2.0/data_filtering/moe/style_consistency/train_filtered \
--metric_field style_consistency \
--seq_len_field length \
--domain_field single_label application_domain \
--tokens 30_000_000_000 \
--tokens_per_shard 3_000_000_000 \
--margin 0.1 \
--seed 42 \
--num_workers 32 \
--strategy "multi_domain"