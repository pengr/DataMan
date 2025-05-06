import os
import argparse
from datasets import load_from_disk
from tqdm import tqdm

def main(input_path, output_path, num_workers, num_shards):
    # 创建输出路径（如果不存在的话）
    os.makedirs(output_path, exist_ok=True)
    
    # 加载数据集
    dataset = load_from_disk(input_path)

    # 循环处理每一个 shard
    for index in range(num_shards):
        split_dataset = dataset.shard(num_shards=num_shards, index=index)

        # 生成输出文件的路径
        base_name = os.path.basename(input_path)
        shard_filename = os.path.join(output_path, f"{base_name}_sub{index + 1}")

        # 保存切分后的数据集
        split_dataset.save_to_disk(shard_filename, num_proc=num_workers)
        print(f"Successfully saved dataset to {shard_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split datasets from multiple folders into two parts using dataset.shard.")
    parser.add_argument("--input", type=str, help="Path to the input datasets.")
    parser.add_argument("--output", type=str, help="Path to the output datasets.")
    parser.add_argument("--num_workers", type=int, default=128, help="Number of workers for saving.")
    parser.add_argument("--num_shards", type=int, default=2, help="Number of shards")
    args = parser.parse_args()
    main(args.input, args.output, args.num_workers, args.num_shards)
