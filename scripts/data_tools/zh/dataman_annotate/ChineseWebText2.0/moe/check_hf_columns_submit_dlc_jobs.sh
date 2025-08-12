#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
DLC_DIR=$CODE_DIR/scripts
cd ${DLC_DIR}

# 0001 0002 0003 0004 0005 0006 0007 0008 
# 0009 0010 0011 0012 0013 0014 0015 0016 
# 0017 0018 0019 0020 0021 0022 0023 0024 
# 0025 0026 0027 0028 0029 0030 0031 0032
# 0033 0034 0035 0036 0037 0038 0039 0040   
# test

# 0047 0144 没有
FILES=(
0041  0042  0043  0044  0045  0046        0048  
0049  0050  0051  0052  0053  0054  0055  0056  
0057  0058  0059  0060  0061  0062  0063  0064  
0065  0066  0067  0068  0069  0070  0071  0072  
0073  0074  0075  0076  0077  0078  0079  0080  
0081  0082  0083  0084  0085  0086  0087  0088  
0089  0090  0091  0092  0093  0094  0095  0096  
0097  0098  0099  0100  0101  0102  0103  0104  
0105  0106  0107  0108  0109  0110  0111  0112  
0113  0114  0115  0116  0117  0118  0119  0120  
0121  0122  0123  0124  0125  0126  0127  0128  
0129  0130  0131  0132  0133  0134  0135  0136  
0137  0138  0139  0140  0141  0142  0143    
0145  0146  0147  0148  0149  0150  0151  
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
    command_script=". /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss"
    for FILE in "${FILES_TO_PROCESS[@]}"; do
        command_script+="
python $CODE_DIR/data_tools/data_utils.py \
    --input $DATA_DIR/DataMan/ChineseWebText2.0/annotated/all_rating/moe/${FILE}_annotated \
    --output $DATA_DIR/DataMan/ChineseWebText2.0/annotated/all_rating/moe \
    --columns accuracy \
              coherence \
              language_consistency \
              semantic_density \
              knowledge_novelty \
              topic_focus \
              creativity \
              professionalism \
              style_consistency \
              grammatical_diversity \
              structural_standardization \
              originality \
              sensitivity \
              overall_score \
    --num_workers 8 \
    --strategy check_hf_columns
"
    done
    
    # 单租提交作业 
    ./dlc create job -c /root/.dlc/config \
    --kind PyTorchJob \
    --name check_hf_columns_dlc_moe_zh_${FILE}_${i} \
    --worker_count 1 \
    --worker_cpu 8 \
    --worker_gpu 0 \
    --worker_memory 128Gi \
    --worker_shared_memory 128Gi \
    --data_sources data2joc1zmqc6ql,datagz8u87nwts56 \
    --worker_image m6-docker-registry-vpc.cn-shanghai.cr.aliyuncs.com/eflops/pengru-pr:pr-py311-cuda123-torch212-qwen2-dataman \
    --workspace_id ws1barvuvxl5j6qh \
    --priority 4 \
    --command "${command_script}" 
    # \
    # --aimaster_args "--job-execution-mode Sync --enable-job-restart True --max-num-of-job-restart 200 --job-restart-timeout 1000000 --fault-tolerant-policy OnFailure --enable-local-detection True --enable-job-hang-detection True --job-hang-interval 1800 --enable-c4d-hang-detection True" \
    # --enable_preemptible_job --aimaster_enable
done