#!/bin/bash
DLC_DIR=/mnt/nas/zhangjinyang/codebase/problem
cd ${DLC_DIR}

FILES=(
0126
0127
0128
0129
0130
0131
0132
0133
0134
0135
0136
0137
0138
0139
0140
0141
0142
0143
0144
0145
0146
0147
0148
0149
0150
0151
)
# 总文件数，分配给每个集群节点处理的文件数，计算所需节点数（若不能整除则额外加一个节点）
total_files=${#FILES[@]}
FILES_PER_NODE=2
num_nodes=$((total_files / FILES_PER_NODE))
if (( total_files % FILES_PER_NODE != 0 )); then
  num_nodes=$((num_nodes + 1))
fi

# num_nodes=1
for ((i=0; i<num_nodes; i++)); do
    start_index=$((i * FILES_PER_NODE))
    FILES_TO_PROCESS=("${FILES[@]:start_index:FILES_PER_NODE}")
    
    # 构建要传递给 --command 参数的脚本内容
    command_script="
. /mnt/nas/zhangjinyang/vars
source $CONDA_DIR/bin/activate
conda activate dataman"
    for FILE in "${FILES_TO_PROCESS[@]}"; do
        # 动态构建执行命令
        command_script+="
gunzip /mnt/nas/zhangjinyang/database/cpt/ChineseWebText2.0/ChineseWebText2.0/part-${FILE}.jsonl.gz
"
    done

    # 单租提交作业 
    ./dlc -c /root/.dlc/config create job \
    --kind PyTorchJob \
    --name gzip_node${i} \
    --worker_count 1 \
    --worker_cpu 32 \
    --worker_gpu 0 \
    --worker_memory 256Gi \
    --worker_shared_memory 256Gi \
    --data_sources dataci26mfk0d7zn,data12t5laf1hr1a \
    --worker_image m6-docker-registry-vpc.cn-shanghai.cr.aliyuncs.com/eflops/zjy-vllm-syn:0 \
    --workspace_id ws1barvuvxl5j6qh \
    --priority 4 \
    --command "${command_script}" 
done