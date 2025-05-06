#!/bin/bash
curdir=`dirname $0`
. $curdir/../../vars
source $CONDA_DIR/bin/activate
conda activate hss

# 外部传入参数
INPUT=$1
OUTPUT=$2
MODEL=$3
BSZ=$4
MAX_TOKENS=$5
DOMAIN_FIELD=$6

# 执行当前checkpoint和当前数据集的ppl
python $CODE_DIR/eval/loglikelihood.py \
    --input $INPUT \
    --output $OUTPUT \
    --model $MODEL \
    --batch_size $BSZ \
    --save_workers 4 \
    --bf16 \
    --max_tokens $MAX_TOKENS \
    --text_field 'text' \
    --tokens_field 'input_ids' \
    --field 'avg_loglikelihood' \
    --domain_field $DOMAIN_FIELD