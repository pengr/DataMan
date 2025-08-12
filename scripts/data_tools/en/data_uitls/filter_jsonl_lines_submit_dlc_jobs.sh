#!/bin/bash
curdir=`dirname $0`
. $curdir/../../../vars
DLC_DIR=$CODE_DIR/scripts
cd ${DLC_DIR}

FILES=("train_chunk1_half1" "train_chunk1_half2" "train_chunk2_half1" "train_chunk2_half2" "train_chunk3_half1"
       "train_chunk3_half2" "train_chunk4_half1" "train_chunk4_half2" "train_chunk5_half1" "train_chunk5_half2" 
       "train_chunk6_half1" "train_chunk6_half2" "train_chunk7_half1" "train_chunk7_half2" "train_chunk8_half1" 
       "train_chunk8_half2" "train_chunk9_half1" "train_chunk9_half2" "train_chunk10_half1" "train_chunk10_half2"
       "validation"         "test" )
# 总文件数，分配给每个集群节点处理的文件数，计算所需节点数（若不能整除则额外加一个节点）
total_files=${#FILES[@]}
FILES_PER_NODE=2
num_nodes=$((total_files / FILES_PER_NODE))
if (( total_files % FILES_PER_NODE != 0 )); then
  num_nodes=$((num_nodes + 1))
fi

for ((i=0; i<num_nodes; i++)); do
    start_index=$((i * FILES_PER_NODE))
    FILES_TO_PROCESS=("${FILES[@]:start_index:FILES_PER_NODE}")
   
    # 构建要传递给 --command 参数的脚本内容
    command_script="curdir=`dirname $0`
. $curdir/../../../vars
DLC_DIR=$CODE_DIR/scripts
cd ${DLC_DIR}"
    for FILE in "${FILES_TO_PROCESS[@]}"; do
        command_script+="
python $CODE_DIR/data_tools/data_utils.py \
    --input $DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/${FILE}_annotated.jsonl \
    --output $DATA_DIR/DataMan/Slimpajama-627B/annotated/all_rating/${FILE}_annotated_score5.jsonl \
    --columns overall_score \
    --strategy "filter_jsonl_lines"
"
    done
    
    # 单租提交作业 
    ./dlc create job -c /root/.dlc/config \
    --kind PyTorchJob \
    --name filter_jsonl_lines_dlc_node${i} \
    --worker_count 1 \
    --worker_cpu 128 \
    --worker_gpu 0 \
    --worker_memory 1024Gi \
    --worker_shared_memory 1024Gi \
    --data_sources data128herqudrh8 \
    --worker_image m6-docker-registry-vpc.cn-shanghai.cr.aliyuncs.com/eflops/pengru-pr:pr-py311-cuda123-torch212-qwen2-dataman \
    --workspace_id ws1i51fratbjvoxp \
    --priority 4 \
    --command "${command_script}" 
done