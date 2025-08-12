#!/bin/bash
. /mnt/nas/pengru.pr/DataMan/scripts/vars
source $CONDA_DIR/bin/activate
conda activate dataman

# 解压缩所有.zst文件
find $DATA_DIR/SlimPajama-627B -type f -name "*.zst" -exec zstd -d {}

# 删除所有文件夹内的.zst文件
find $DATA_DIR/SlimPajama-627B -type f -name "*.zst" -delete