# 脚本主要功能：加载一个或多个数据集，对特定列进行保留或移除，然后将处理后的数据集保存到指定的输出路径。
# 如果多个数据集需要合并，可以选择将它们合并为一个数据集。
import argparse
from datasets import load_from_disk, load_dataset, concatenate_datasets
from tqdm import tqdm
import os
import zstd
import json

# 用于对多个数据集进行连接（concat），将它们合并为一个数据集。数据集必须具有相同的结构
def concatenated_datasets(args):
    datasets = dict(load_datasets(args))
    yield args.output, concatenate_datasets(list(datasets.values()))


def load_datasets(args):
    for path in tqdm(args.input, desc="Loading datasets"):
        if args.json:
            # 用于加载 Hugging Face 提供的标准数据集，或从其他自定义来源（如 CSV、JSON 文件等）加载数据集
            dataset = load_dataset("json", data_files=path, split="train")
        else:
            # 用于从磁盘加载已经存在的数据集。这个数据集必须是先前保存的 datasets 格式
            dataset = load_from_disk(path)  
        yield (args.output + "/" + os.path.basename(path)), dataset


def main(args):
    if args.single:
        datasets = concatenated_datasets(args)
    else:
        datasets = load_datasets(args)

    for output_path, dataset in datasets:
        column_names = dataset.column_names
        print(f"column names: {column_names}")

        columns_to_remove = set(column_names) - set(args.columns)
        dataset = dataset.remove_columns(columns_to_remove)

        print(f"Saving to '{output_path}'...")
        if args.parquet:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # compression: 压缩算法, 可以是 None, gzip, brotli, lz4, zstd, snappy 等
            dataset.to_parquet(output_path, compression=args.compression, compression_level=args.compression_level)
        else:
            # 使用的进程数目。当保存大规模数据集时，可以通过并行化（多进程）加速保存过程。
            dataset.save_to_disk(output_path, num_proc=28)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for tokenizing a dataset.")
    parser.add_argument("--input", type=str, nargs="+", help="Path list to the input datasets.")
    parser.add_argument("--output", type=str, help="Path to the output tokenized dataset.")
    parser.add_argument("--columns", type=str, nargs="+", help="Columns to keep.", default=["input_ids", "length"])
    parser.add_argument("--json", action="store_true", help="Input is json dataset.")
    parser.add_argument("--single", action="store_true", help="Concatenate into single input/output dataset.")
    parser.add_argument("--parquet", action="store_true", help="Store as parquet")
    parser.add_argument("--compression", type=str, default="zstd", help="Compression algorithm for parquet format.")
    parser.add_argument("--compression_level", type=int, default=3, help="Compression level for parquet format.")

    args = parser.parse_args()
    main(args)
