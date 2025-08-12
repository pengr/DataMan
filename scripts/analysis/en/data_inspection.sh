#!/bin/bash
curdir=`dirname $0`
. $curdir/../../vars
DLC_DIR=$CODE_DIR/scripts
cd ${DLC_DIR}

FILES=(
'accuracy'
'coherence'
'language_consistency'
'semantic_density'
'knowledge_novelty'
'topic_focus'
'creativity'
'professionalism'
'style_consistency'
'grammatical_diversity'
'structural_standardization'
'originality'
'sensitivity'
       )
# 总文件数，分配给每个集群节点处理的文件数，计算所需节点数（若不能整除则额外加一个节点）
total_files=${#FILES[@]}
FILES_PER_NODE=1
num_nodes=$((total_files / FILES_PER_NODE))
if (( total_files % FILES_PER_NODE != 0 )); then
  num_nodes=$((num_nodes + 1))
fi

for ((i=0; i<num_nodes; i++)); do
    start_index=$((i * FILES_PER_NODE))
    FILES_TO_PROCESS=("${FILES[@]:start_index:FILES_PER_NODE}")
   
    # 构建要传递给 --command 参数的脚本内容
    command_script="curdir=`dirname $0`
. $curdir/../../vars
source $CONDA_DIR/bin/activate
conda activate dataman"
    for FILE in "${FILES_TO_PROCESS[@]}"; do
        command_script+="
python $CODE_DIR/analysis/data_inspection.py \
--input_paths "$DATA_DIR/Slimpajama-627B/data_filtering/uniform/train_filtered/0" \
--criteria_list ${FILE} \
--output_path $CKPT_DIR/DataMan/analysis/data_inspection
"
    done
done