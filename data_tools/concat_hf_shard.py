#合并指定路径上的数据集
import os
import argparse
from datasets import load_from_disk, load_dataset, concatenate_datasets
from tqdm import tqdm

def main(input_paths, output_path, num_workers):
    merged_dataset = concatenate_datasets([load_from_disk(path) for path in tqdm(input_paths)]) # 合并所有数据集
    merged_dataset.save_to_disk(output_path, num_proc=num_workers)   # 保存合并后的数据集到指定文件夹
    print(f"Successfully merged and saved dataset to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge datasets from multiple folders.")
    parser.add_argument("--inputs", type=str, nargs="+", help="Path to the input datasets.")
    parser.add_argument("--output", type=str, help="Path to the output dataset.")
    parser.add_argument("--num_workers", type=int, default=128, help="Workers for saving.")
    args = parser.parse_args()
    main(args.inputs, args.output, args.num_workers)