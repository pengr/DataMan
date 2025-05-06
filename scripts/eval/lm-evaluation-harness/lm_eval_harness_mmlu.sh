#!/bin/bash
curdir=`dirname $0`
. $curdir/../../vars
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

echo Task: mmlu
OUTFILES=${OUT}/mmlu
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks mmlu \
    --num_fewshot 5 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out 