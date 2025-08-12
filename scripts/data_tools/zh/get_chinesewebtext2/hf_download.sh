#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
DLC_DIR=$CODE_DIR/scripts
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
FILES_PER_NODE=1
num_nodes=$((total_files / FILES_PER_NODE))
if (( total_files % FILES_PER_NODE != 0 )); then
  num_nodes=$((num_nodes + 1))
fi
# num_nodes=1
for ((i=1; i<num_nodes; i++)); do
    start_index=$((i * FILES_PER_NODE))
    FILES_TO_PROCESS=("${FILES[@]:start_index:FILES_PER_NODE}")

    command_script=""". /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss
"""

    for FILE in "${FILES_TO_PROCESS[@]}"; do
        echo ${FILE}
        command_script+="
export HF_ENDPOINT=https://hf-mirror.com/
huggingface-cli download --repo-type dataset --resume-download CASIA-LM/ChineseWebText2.0 --include "ChineseWebText2.0/part-${FILE}.jsonl.gz" --local-dir /mnt/nas/zhangjinyang/database/cpt/ChineseWebText2.0
"
    done
done
