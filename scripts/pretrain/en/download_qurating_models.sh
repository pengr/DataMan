#!/bin/bash
curdir=`dirname $0`
. $curdir/../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

# 模型名称数组
models=(
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-uniform-sampling"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-dsir_wikipedia_en"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-dsir_book"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-perplexity-bottom_k"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-perplexity-top_k"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-uniform-sampling-curriculum-high_to_low-required_expertise"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-uniform-sampling-curriculum-low_to_high-required_expertise"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-writing_style-top_k"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-writing_style-sample_with_temperature1.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-writing_style-sample_with_temperature2.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-facts_and_trivia-top_k"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-facts_and_trivia-sample_with_temperature1.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-facts_and_trivia-sample_with_temperature2.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-educational_value-top_k"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-educational_value-sample_with_temperature1.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-educational_value-sample_with_temperature2.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-required_expertise-top_k"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-required_expertise-sample_with_temperature1.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-required_expertise-sample_with_temperature2.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-mix_of_criteria-sample_with_temperature2.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_writing_style-top_k"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_writing_style-sample_with_temperature1.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_writing_style-sample_with_temperature2.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_facts_and_trivia-top_k"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_facts_and_trivia-sample_with_temperature1.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_facts_and_trivia-sample_with_temperature2.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_educational_value-top_k"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_educational_value-sample_with_temperature1.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_educational_value-sample_with_temperature2.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_required_expertise-top_k"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_required_expertise-sample_with_temperature1.0"
    "princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_required_expertise-sample_with_temperature2.0"
    "princeton-nlp/lm-1.3B-select_45B_tokens_by-uniform-sampling"
)

# 遍历每个模型
for model in "${models[@]}"; do
    local_dir="$CKPT_DIR/DataMan/pretrain/$model"
    until huggingface-cli download --resume-download $model --local-dir $local_dir --local-dir-use-symlinks False; do
        echo "下载 $model 失败，正在重试..."
        sleep 5 # 等待5秒再次尝试
    done
    echo "下载 $model 完成"
done