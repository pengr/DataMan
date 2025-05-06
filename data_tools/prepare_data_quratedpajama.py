import os
import gc
import time
import random
import argparse
import numpy as np
import multiprocessing
from multiprocessing import Manager, Pool
from math import ceil
from tqdm import tqdm
import numpy.typing as npt
from functools import partial, reduce
from typing import List, Optional, Tuple, Iterable
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from itertools import islice
import pandas as pd
import json
import re
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import torch
import glob

# 定义文件目录和分配数量
directory = 'xxxxxxxxxxxxxxxdata/DataMan/princeton-nlp/QuRatedPajama-260B/org'
output_dir = 'xxxxxxxxxxxxxxxdata/DataMan/princeton-nlp/QuRatedPajama-260B/huggingface'
# 获取所有的 tsv 文件路径
tsv_files = glob.glob(os.path.join(directory, '*.tsv'))
total_files = len(tsv_files)
num_datasets = 20
files_per_dataset = total_files // num_datasets
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def process_dataset(index):
    start = index * files_per_dataset
    end = min((index + 1) * files_per_dataset, total_files)
    subset_files = tsv_files[start:end]
    
    # 加载和合并数据集
    dataset = concatenate_datasets([
        load_dataset('json', data_files=path, split='train')
        for path in tqdm(subset_files, desc=f"Processing dataset {index}")
    ])
    
    # 保存到Arrow文件
    arrow_file = os.path.join(output_dir, f'dataset_{index}')
    dataset.save_to_disk(arrow_file, num_proc=32)
    print(f'Saved dataset {index} to {arrow_file}')

if __name__ == "__main__":
    # 创建进程池
    with Pool(num_datasets) as pool:
        # 使用map方法并行处理
        pool.map(process_dataset, range(num_datasets))