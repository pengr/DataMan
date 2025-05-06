#!/bin/bash
curdir=`dirname $0`
. $curdir/../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

python $CODE_DIR/eval/loglikelihood.py \
--input $DATA_DIR/DataMan/Slimpajama-627B/data_filtering/uniform/train_filtered/0 \
--output $CKPT_DIR/DataMan/analysis/en/dense/Llama-2-7b-hf-ppl \
--model $CKPT_DIR/base/Llama-2-7b-hf \
--batch_size 40 \
--num_workers 1 \
--save_workers 4 \
--fp16 \
--max_tokens 10000000 \
--text_field 'text' \
--tokens_field 'input_ids' \
--field 'avg_loglikelihood' \
--domain_field 'source_domain' \
--shard 30 1