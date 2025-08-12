#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# DASH OpenAI Call
# export CALL_URL=xxxxx
# export DASHSCOPE_API_KEY=xxxxxx

OUTPUTS_A=${1}
OUTPUTS_B=${2}
ANNOTATOR=${3}

python $CODE_DIR/eval/instruction_tuning/get_winrate.py \
--outputs_A $OUTPUTS_A \
--outputs_B $OUTPUTS_B \
--annotators_config $ANNOTATOR