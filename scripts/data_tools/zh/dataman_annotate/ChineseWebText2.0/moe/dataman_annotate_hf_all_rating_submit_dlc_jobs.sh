#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
DLC_DIR=$CODE_DIR/scripts
cd ${DLC_DIR}

# 已运行完
# 0001 0002 0003 0004 0005 0006 0007 0008 
# 0009 0010 0011 0012 0013 0014 0015 0016 
# 0017 0018 0019 0020 0021 0022 0023 0024 
# 0025 0026 0027 0028 0029 0030 0031 0032
# 0033 0034 0035 0036 0037 0038 0039 0040   
# 0041 0044 0065  0071 0084 0089 0125
# 0051 0060      0042  0043    0045  0046    0048  
# 0049  0050    0052  0053  0054  0055  0056  
# 0057  0058  0059    0061  0062  0063   
#     0067  0068  0069     0072  
#     0075   0077    0085  0087  0088  
#     0090  0091  0092  0093  0094  0095  0096  
# 0097  0098  0099     0102  0103   0112  
#     0118  0119   
# 0121   0123  0124    0126  0127  0128  
# 0129  0130  0131  0132  0133  0134  0135  0136  
# 0137  0138  0139  0140   0143  0146   0151 
# 0064 0066 0070 0082 0100 0101 0104  
# 0105  0106  0107  0108  0109  0110  0111 
# 0113  0114  0115  0116  0117 0120 0122
# 0076 0074          0078  0079
#  0141 0142 
# 0145  0147 0149  0150 0071 0041
# 0070 0130  0129 0126 0087 0060 
# 0051 0089  0044 0052 0053 
# 0055 0058 0043 0135 0097
# 0085 0151 0086 0143 0081
# 0090 0148 0080 0083   0073 

# 正在跑
FILES=(
      0102
       )
# 注：0047和0144本来就没有
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

    command_script=""". /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate hss
"""

    for FILE in "${FILES_TO_PROCESS[@]}"; do
        for (( SHARD=0; SHARD<8; SHARD++ )); do
            command_script+="
CUDA_VISIBLE_DEVICES=$SHARD python $CODE_DIR/data_tools/dataman_annotate_hf.py \
    --model_name_or_path $CKPT_DIR/DataMan/sft/zh/all_rating/moe_models/Qwen1_5_moe_2_7B-4k_bal_q3-zh/checkpoint-700 \
    --input_path /mnt/nas/zhangjinyang/database/cpt/ChineseWebText2.0/ChineseWebText2.0/part-${FILE}.jsonl \
    --processed_path $DATA_DIR/DataMan/ChineseWebText2.0/preprocessed/${FILE}_preprocessed \
    --inferenced_path $DATA_DIR/DataMan/ChineseWebText2.0/annotated/all_rating/moe/${FILE}_annotated/$SHARD \
    --lang 'zh' \
    --num_cpu_workers 16 \
    --num_gpu_workers 1 \
    --max_tokens 29 \
    --seed 1024 \
    --temperature 0.0 \
    --truncate_max_length 1896 \
    --char_truncate_max_length 60000 \
    --data_format ChineseWebText2 \
    --model_type all_rating \
    --use_fast \
    --batch_size 10000 \
    --inference_shard 8 $SHARD & "
        done
        command_script+="
        sleep 5h
        "
    done
    
    # 单租group_m6提交作业         --worker_gpu_type NVIDIA-L20Y \
    ./dlc create job -c /root/.dlc/config \
        --kind PyTorchJob \
        --name dataman_annotate_hf_all_rating_zh_${FILE}_node${i} \
        --worker_count 1 \
        --worker_cpu 32 \
        --worker_gpu 8 \
        --worker_memory 256Gi \
        --worker_shared_memory 256Gi \
        --data_sources data2joc1zmqc6ql,datagz8u87nwts56 \
        --worker_image m6-docker-registry-vpc.cn-shanghai.cr.aliyuncs.com/eflops/pengru-pr:pr-py311-cuda123-torch212-qwen2-dataman \
        --workspace_id ws1barvuvxl5j6qh \
        --priority 4 \
        --command "${command_script}" \
        --aimaster_args "--job-execution-mode Sync --enable-job-restart True --max-num-of-job-restart 200 --job-restart-timeout 1000000 --fault-tolerant-policy OnFailure --enable-local-detection True --enable-job-hang-detection True --job-hang-interval 1800 --enable-c4d-hang-detection True" \
        --enable_preemptible_job --aimaster_enable
done
