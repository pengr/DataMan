#!/bin/bash
python /mnt/nas/pengru.pr/DataMan/data_filtering/compress_datasets.py \
    --input "/mnt/nas/pengru.pr/checkpoints/data_filtering/en/score/test_processed.jsonl" \
    --output "/mnt/nas/pengru.pr/checkpoints/data_filtering/en/score/test_compressed.parquet" \
    --columns "text", "id" \
    --json \
    --single \
    --parquet \
    --compression "zstd" \
    --compression_level" "3"


# 解压parquet文件的方法
# import pandas as pd
# # 读取parquet文件
# df = pd.read_parquet('/mnt/nas/pengru.pr/checkpoints/data_filtering/en/score/test_compressed.parquet')
# # 将数据帧写入jsonl
# df.to_json('/mnt/nas/pengru.pr/checkpoints/data_filtering/en/score/test_compressed.jsonl', orient='records', lines=True)