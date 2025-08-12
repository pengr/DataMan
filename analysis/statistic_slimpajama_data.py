import gc
import json
import os
import random
import re
import time
from collections import Counter
from functools import partial, reduce
from math import ceil
from multiprocessing import Manager, Pool
from typing import List, Optional, Tuple, Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm

def process_batch(batch):
    source_counts = Counter(batch['source_domain'])
    application_counts = Counter(batch['application_domain'])
    overall_score_counts = Counter(batch['overall_score'])
    return source_counts, application_counts, overall_score_counts

def process_dataset(input_paths):
    dataset = concatenate_datasets([load_from_disk(path) for path in tqdm(input_paths)]) # 合并所有数据集
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['source_domain', 'application_domain', 'overall_score']])
    
    total_batches = (len(dataset) + 99999) // 100000  # 计算总批次数
    with Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(process_batch, dataset.iter(batch_size=100000)), 
                            total=total_batches, 
                            desc="Processing batches"))  
    
    total_source_counts = Counter()
    total_application_counts = Counter()
    total_overall_score_counts = Counter()
    for source_count, application_count, overall_score_count in results:
        total_source_counts.update(source_count)
        total_application_counts.update(application_count)
        total_overall_score_counts.update(overall_score_count)
    
    return total_source_counts, total_application_counts, total_overall_score_counts

# 训练集
input_paths = [
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk1_half1_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk1_half2_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk2_half1_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk2_half2_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk3_half1_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk3_half2_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk4_half1_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk4_half2_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk5_half1_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk5_half2_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk6_half1_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk6_half2_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk7_half1_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk7_half2_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk8_half1_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk8_half2_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk9_half1_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk9_half2_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk10_half1_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/train_chunk10_half2_annotated_success",
]

total_source_counts, total_application_counts, total_overall_score_counts = process_dataset(input_paths)

# 计算占比
source_domain_total = sum(total_source_counts.values())
application_domain_total = sum(total_application_counts.values())
overall_score_total = sum(total_overall_score_counts.values())

# 打印占比
print("\nSource Domain Proportions:")
for domain, count in total_source_counts.most_common():
    proportion = (count / source_domain_total) * 100
    print(f"{domain}: {count}, {proportion:.2f}%")

print("\nApplication Domain Proportions:")
for domain, count in total_application_counts.most_common():
    proportion = (count / application_domain_total) * 100
    print(f"{domain}: {count}, {proportion:.2f}%")

print("\nOverall Score Proportions:")
for overall_score, count in total_overall_score_counts.most_common():
    proportion = (count / overall_score_total) * 100
    print(f"{overall_score}: {count}, {proportion:.2f}%")

#!/bin/bash
## EN Pre-training Data
# python /mnt/nas/pengru.pr/DataMan/analysis/en/statistic_slimpajama_data.py >> /mnt/nas/pengru.pr/checkpoints/DataMan/analysis/en/statistic_slimpajama_data.log