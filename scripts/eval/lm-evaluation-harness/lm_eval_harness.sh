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

echo Task: arc_easy
OUTFILES=${OUT}/arc_easy
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks arc_easy \
    --num_fewshot 15 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out

echo Task: arc_challenge
OUTFILES=${OUT}/arc_challenge
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks arc_challenge \
    --num_fewshot 15 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out

echo Task: sciq
OUTFILES=${OUT}/sciq
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks sciq \
    --num_fewshot 2 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out

echo Task: logiqa
OUTFILES=${OUT}/logiqa
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks logiqa \
    --num_fewshot 2 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out

echo Task: boolq
OUTFILES=${OUT}/boolq
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks boolq \
    --num_fewshot 0 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out

echo Task: hellaswag
OUTFILES=${OUT}/hellaswag
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks hellaswag \
    --num_fewshot 6 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out \


echo Task: piqa
OUTFILES=${OUT}/piqa
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks piqa \
    --num_fewshot 6 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out

echo Task: winogrande
OUTFILES=${OUT}/winogrande
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks winogrande \
    --num_fewshot 15 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out

echo Task: nq_open
OUTFILES=${OUT}/nq_open
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks nq_open \
    --num_fewshot 10 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out

echo Task: mmlu
OUTFILES=${OUT}/mmlu
python lm_eval --model hf \
    --model_args pretrained=$CKPT,parallelize=True \
    --tasks mmlu \
    --num_fewshot 5 \
    --output_path "$OUTFILES" \
    --batch_size $BSZ \
    --write_out