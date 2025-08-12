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
    source_counts = Counter(domain['single_label'] for domain in batch['domain'])
    application_counts = Counter(batch['application_domain'])
    overall_score_counts = Counter(batch['overall_score'])
    return source_counts, application_counts, overall_score_counts

def process_dataset(input_paths):
    dataset = concatenate_datasets([load_from_disk(path) for path in tqdm(input_paths)]) # 合并所有数据集
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['domain', 'application_domain', 'overall_score']])
    
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

# 训练集，注意没有0047，0144!
input_paths = [
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0001_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0002_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0003_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0004_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0005_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0006_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0007_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0008_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0009_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0010_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0011_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0012_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0013_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0014_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0015_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0016_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0017_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0018_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0019_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0020_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0021_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0022_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0023_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0024_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0025_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0026_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0027_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0028_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0029_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0030_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0031_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0032_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0033_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0034_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0035_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0036_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0037_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0038_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0039_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0040_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0041_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0042_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0043_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0044_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0045_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0046_annotated_success",
# "/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0047_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0048_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0049_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0050_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0051_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0052_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0053_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0054_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0055_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0056_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0057_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0058_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0059_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0060_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0061_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0062_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0063_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0064_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0065_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0066_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0067_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0068_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0069_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0070_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0071_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0072_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0073_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0074_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0075_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0076_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0077_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0078_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0079_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0080_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0081_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0082_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0083_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0084_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0085_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0086_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0087_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0088_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0089_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0090_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0091_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0092_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0093_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0094_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0095_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0096_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0097_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0098_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0099_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0100_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0101_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0102_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0103_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0104_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0105_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0106_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0107_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0108_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0109_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0110_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0111_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0112_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0113_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0114_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0115_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0116_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0117_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0118_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0119_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0120_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0121_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0122_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0123_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0124_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0125_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0126_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0127_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0128_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0129_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0130_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0131_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0132_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0133_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0134_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0135_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0136_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0137_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0138_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0139_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0140_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0141_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0142_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0143_annotated_success",
# "/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0144_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0145_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0146_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0147_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0148_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0149_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0150_annotated_success",
"/mnt/nas/pengru.pr/data/DataMan/ChineseWebText2.0/annotated/all_rating/moe/0151_annotated_success",
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