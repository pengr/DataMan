#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# 在instruct following abilities对比时，无需放"uniform-sample_with_overall_score1.0/2.0/3.0/4.0"
FILES=(
   "accuracy-top_k-sampling" 
   "coherence-top_k-sampling" 
   "creativity-top_k-sampling"
   "grammatical_diversity-top_k-sampling" 
   "knowledge_novelty-top_k-sampling"
   "language_consistency-top_k-sampling"  
   "originality-top_k-sampling" 
   "professionalism-top_k-sampling"
   "semantic_density-top_k-sampling" 
   "sensitivity-top_k-sampling" 
   "structural_standardization-top_k-sampling"
   "style_consistency-top_k-sampling" 
   "topic_focus-top_k-sampling" 
   "uniform-sample_with_overall_score5.0")

# 执行命令 DataMan模型  V.S. educational_value-sample_with_temperature2.0
for FILE in "${FILES[@]}"; do
    bash $CODE_DIR/scripts/eval/instruction_tuning/get_winrate.sh \
    $CKPT_DIR/DataMan/eval/en/instruction_tuning_ShareGPT/lm-Sheared-LLaMA-1.3B-select_30B_tokens_by-${FILE}/${CHECKPOINT}/test/temp0.0_num1.json \
    $CKPT_DIR/DataMan/eval/en/instruction_tuning_ShareGPT/lm-1.3B-select_30B_tokens_by-educational_value-sample_with_temperature2.0/checkpoint-merge/test/temp0.0_num1.json \
    alpaca_eval_gpt4o
done