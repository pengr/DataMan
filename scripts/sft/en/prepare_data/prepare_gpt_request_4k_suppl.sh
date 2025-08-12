#!/bin/bash
curdir=`dirname $0`
. $curdir/../../../vars
source $CONDA_DIR/bin/activate
conda activate dataman

# gpt_model can be "gpt-4-0125-preview" or "gpt-4o" 
FILES=("code_slim_github" "code_slim_stackexchange")
for FILE in "${FILES[@]}"; do
  python $CODE_DIR/sft/prepare_data.py \
    --input_path "$DATA_DIR/Qwen-Org/${FILE}.jsonl" \
    --output_path "$DATA_DIR/DataMan/Qwen-sft/Qwen2-en-suppl/${FILE}_request.jsonl" \
    --lang 'en' \
    --gpt_model "gpt-4o" \
    --truncate \
    --seed 1024 \
    --trucate_max_length 1894 \
    --sample_size 10500 \
    --process_type prepare_gpt_request
done

python $CODE_DIR/sft/prepare_data.py \
  --input_path "$DATA_DIR/Qwen-Org/en_slim_arxiv.jsonl" \
  --output_path "$DATA_DIR/DataMan/Qwen-sft/Qwen2-en-suppl/en_slim_arxiv_request_2.jsonl" \
  --exist_output_path "$DATA_DIR/DataMan/Qwen-sft/Qwen2-en/en_slim_arxiv_request.jsonl" "$DATA_DIR/DataMan/Qwen-sft/Qwen2-en/en_slim_arxiv_request_1.jsonl" \
  --lang 'en' \
  --gpt_model "gpt-4o" \
  --truncate \
  --seed 1024 \
  --trucate_max_length 1894 \
  --sample_size 2200 \
  --process_type prepare_gpt_request

python $CODE_DIR/sft/prepare_data.py \
  --input_path "$DATA_DIR/Qwen-Org/en_wiki.jsonl" \
  --output_path "$DATA_DIR/DataMan/Qwen-sft/Qwen2-en-suppl/en_wiki_request_2.jsonl" \
  --exist_output_path "$DATA_DIR/DataMan/Qwen-sft/Qwen2-en/en_wiki_request.jsonl" "$DATA_DIR/DataMan/Qwen-sft/Qwen2-en/en_wiki_request_1.jsonl" \
  --lang 'en' \
  --gpt_model "gpt-4o" \
  --truncate \
  --seed 1024 \
  --trucate_max_length 1894 \
  --sample_size 6300 \
  --process_type prepare_gpt_request