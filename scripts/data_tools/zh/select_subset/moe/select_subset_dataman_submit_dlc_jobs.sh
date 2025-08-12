#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
DLC_DIR=$CODE_DIR/scripts
cd ${DLC_DIR}

FILES=(
    #    "baselines/select_subset_bert_probability"
       "baselines/select_subset_perplexity_highest"
       "baselines/select_subset_perplexity_lowest"
    #    "baselines/select_subset_uniform"
    #    "individual_quality_criteria/select_subset_accuracy"         
    #    "individual_quality_criteria/select_subset_coherence"  
    #    "individual_quality_criteria/select_subset_creativity" 
    #    "individual_quality_criteria/select_subset_grammatical_diversity" 
    #    "individual_quality_criteria/select_subset_knowledge_novelty" 
    #    "individual_quality_criteria/select_subset_language_consistency" 
    #    "individual_quality_criteria/select_subset_originality"    
    #    "individual_quality_criteria/select_subset_professionalism" 
    #    "individual_quality_criteria/select_subset_semantic_density" 
    #    "individual_quality_criteria/select_subset_sensitivity" 
    #    "individual_quality_criteria/select_subset_structural_standardization" 
    #    "individual_quality_criteria/select_subset_style_consistency" 
    #    "individual_quality_criteria/select_subset_topic_focus"  
    #    "overall_score/select_subset_overall_score1" 
    #    "overall_score/select_subset_overall_score2" 
    #    "overall_score/select_subset_overall_score3"
    #    "overall_score/select_subset_overall_score4" 
    #    "overall_score/select_subset_overall_score5" 
    #    "quality_criteria_mix/select_subset_quality_criteria_mix"
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
        # 动态构建执行命令
        command_script+="
bash $CODE_DIR/scripts/data_tools/zh/select_subset/moe/${FILE}.sh
"
    done

    # 单租提交作业, name不能有反斜杠/，字符数也不能超过63
    ./dlc create job -c /root/.dlc/config \
        --kind PyTorchJob \
        --name moe_${FILE##*/}_node${i} \
        --worker_count 1 \
        --worker_cpu 32 \
        --worker_gpu 0 \
        --worker_memory 300Gi \
        --worker_shared_memory 300Gi \
        --data_sources data2joc1zmqc6ql,datagz8u87nwts56 \
        --worker_image m6-docker-registry-vpc.cn-shanghai.cr.aliyuncs.com/eflops/pengru-pr:pr-py311-cuda123-torch212-qwen2-dataman \
        --workspace_id ws1barvuvxl5j6qh \
        --priority 4 \
        --command "${command_script}" 
        # \
        # --aimaster_args "--job-execution-mode Sync --enable-job-restart True --max-num-of-job-restart 200 --job-restart-timeout 1000000 --fault-tolerant-policy OnFailure --enable-local-detection True --enable-job-hang-detection True --job-hang-interval 1800 --enable-c4d-hang-detection True" \
        # --enable_preemptible_job --aimaster_enable
done