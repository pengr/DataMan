#!/bin/bash
curdir=`dirname $0`
. $curdir/../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

input_files=""
for chunk in {0..19}; do
    input_files+="$DATA_DIR/DataMan/princeton-nlp/QuRatedPajama-260B/huggingface/dataset_${chunk} "
done

# DSW Command: 新版本：随机采样按照source_domain占比来采样
CPUS_NODE=$(python -c 'import os; print(os.cpu_count())')
# 由于无shard 选项，若想将30B数据存储为单个shard时，因此tokens表面是30B，实际由于margin会更大一些。
# 所以需要将 tokens_per_shard 设置大于 tokens 才可以一次性覆盖这些额外的 tokens。
python $CODE_DIR/data_tools/select_subset_qurating.py \
--inputs $input_files \
--output $DATA_DIR/DataMan/princeton-nlp/QuRatedPajama-260B/data_filtering/educational_value-sample_with_temperature2.0/train_filtered \
--seq_len_field length \
--metric_field educational_value_average \
--domain_field source_domain \
--tokens 60_000_000_000 \
--temperature 2.0 \
--sample \
--normalize \
--tokens_per_shard 3_000_000_000 \
--margin 0.1 \
--seed 42 \
--num_workers 180