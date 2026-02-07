#!/bin/bash

# Set mirror source and proxy (optional)
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"

## create conda virtual environment, python 3.8, CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
cd "$PWD/DataMan"
bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -p "$PWD/anaconda3"
source "$PWD/anaconda3/bin/activate"

## Or, you can activate an existing dataman virtual environment
# source /miniconda3/bin/activate dataman

#pytorch 1.12 and above, 2.0 and above are recommended
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install jsonlines
pip install gevent
#(for Qwen transformers>=4.32, for Qwen1.5/2 transformers>= 4.37.0)
pip install transformers
pip install wandb
pip install accelerate
pip install tiktoken
pip install einops
pip install transformers_stream_generator
pip install peft 
pip install deepspeed
#if pip install failed, see https://github.com/mpi4py/mpi4py/issues/335, https://blog.csdn.net/liuliqun520/article/details/125416284
pip install mpi4py
#if pip install failed, see https://github.com/oobabooga/text-generation-webui/issues/4182
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation
#Below are optional. Installing them might be slow.
git clone -b v2.5.6 --depth 1 https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install csrc/layer_norm --no-build-isolation
#If the version of flash-attn is higher than 2.1.1, the following is not needed.
pip install csrc/rotary
pip install vllm==0.3.3
pip install ray
# huggingface
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
# gguf,llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
# lm_eval
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
# fire, alpaca_eval, need to modify the underlying code
pip install fire
pip install alpaca_eval
# alpaca-farm
pip install alpaca-farm
pip install mergekit
pip install datatrove
