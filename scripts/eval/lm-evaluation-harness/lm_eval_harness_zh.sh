#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss

export HF_ENDPOINT=https://hf-mirror.com
export MKL_THREADING_LAYER=GNU

# 外部传入参数
CKPT=$1
OUT=$2
BSZ=$3

# 执行当前checkpoint的lm-evaluation-harness
cd $CODE_DIR/eval/lm-evaluation-harness
if [ ! -n "$BSZ" ] ; then BSZ="auto:4" ; fi
echo Eval Model: $CKPT
echo "Running checkpoint $CKPT with batch size $BSZ to output_path $OUT"

## num_fewshot最大是5
echo Task: ceval-valid
OUTFILES=${OUT}/ceval-valid
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks ceval-valid \
    --num_fewshot 5 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out 

# num_fewshot最大是5
# echo Task: cmmlu
# OUTFILES=${OUT}/cmmlu
# python lm_eval --model hf \
#     --model_args pretrained=$CKPT,parallelize=True \
#     --tasks cmmlu \
#     --num_fewshot 5 \
#     --output_path "$OUTFILES" \
#     --batch_size $BSZ \
#     --write_out 

## num_fewshot可以增大到10，但batch_size要同步减少8
echo Task: agieval_cn
OUTFILES=${OUT}/agieval_cn
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks agieval_cn \
    --num_fewshot 5 \
    --output_path "$OUTFILES" \
    --batch_size 1 \
    --write_out 

