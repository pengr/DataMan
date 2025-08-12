#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss

input_files=""
for chunk in {1..10}; do
    for half in {1..2}; do
        input_files+="$DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/moe/train_chunk${chunk}_half${half}_annotated_success "
    done
done

# DSW Command: 新版本：随机采样按照source_domain占比来采样
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')
# tokens选择一个1万亿是为了把所有"4","5"分的领域数据全覆盖，但是tokens_per_shard还是设置30亿就好了
python $CODE_DIR/data_tools/select_subset.py \
--inputs $input_files \
--output $DATA_DIR/DataMan/Slimpajama-627B/data_filtering/moe/medicine/train_filtered \
--metric_field overall_score \
--seq_len_field length \
--domain_field application_domain \
--tokens 10_000_000_000_000 \
--tokens_per_shard 3_000_000_000 \
--margin 0.1 \
--seed 42 \
--num_workers 64 \
--target_scores "5" \
--target_domains "A" \
--strategy "None"