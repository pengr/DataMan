#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss 
# 必须是hss, dataman跑中文任务会报错:Exception: data did not match any variant of untagged enum ModelWrapper at line 757452 column 3
# 因为 tokenizer 文件格式不兼容或损坏

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