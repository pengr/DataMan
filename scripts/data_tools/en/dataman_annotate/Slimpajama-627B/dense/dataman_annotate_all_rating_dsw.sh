#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
DLC_DIR=$CODE_DIR/scripts
cd ${DLC_DIR}

FILES=("train_chunk1_half1" "train_chunk1_half2" "train_chunk2_half1" "train_chunk2_half2" "train_chunk3_half1"
       "train_chunk3_half2" "train_chunk4_half1" "train_chunk4_half2" "train_chunk5_half1" "train_chunk5_half2" 
       "train_chunk6_half1" "train_chunk6_half2" "train_chunk7_half1" "train_chunk7_half2" "train_chunk8_half1" 
       "train_chunk8_half2" "train_chunk9_half1" "train_chunk9_half2" "train_chunk10_half1" "train_chunk10_half2"
       "validation"         "test")
# 总文件数，分配给每个集群节点处理的文件数，计算所需节点数（若不能整除则额外加一个节点）
total_files=${#FILES[@]}
FILES_PER_NODE=4
num_nodes=$((total_files / FILES_PER_NODE))
if (( total_files % FILES_PER_NODE != 0 )); then
  num_nodes=$((num_nodes + 1))
fi
    
for ((i=0; i<num_nodes; i++)); do
    start_index=$((i * FILES_PER_NODE))
    FILES_TO_PROCESS=("${FILES[@]:start_index:FILES_PER_NODE}")
    
    # 构建要传递给 --command 参数的脚本内容
    command_script=""
    for FILE in "${FILES_TO_PROCESS[@]}"; do
        command_script+=". /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

python $CODE_DIR/data_tools/dataman_annotate.py \
    --model_name_or_path $CKPT_DIR/DataMan/sft/en/all_rating/suppl_models/Qwen2-1_5B-4k_suppl_bal_q4-en/checkpoint-900 \
    --input_path $DATA_DIR/DataMan/SlimPajama-627B/org/${FILE}.jsonl \
    --processed_path $DATA_DIR/DataMan/Slimpajama-627B/preprocessed/${FILE}_preprocessed.jsonl \
    --inferenced_path $DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/${FILE}_annotated.jsonl \
    --llama_tokenizer_path $CKPT_DIR/base/Sheared-LLaMA-1.3B \
    --lang en \
    --num_cpu_workers $(python -c 'import os; print(os.cpu_count())') \
    --max_tokens 29 \
    --seed 1024 \
    --temperature 0.0 \
    --truncate_max_length 1894 \
    --char_truncate_max_length 20000 \
    --num_gpu_workers $(python -c 'import torch; print(torch.cuda.device_count())') \
    --data_format slimpajama \
    --model_type all_rating
"
    done
done