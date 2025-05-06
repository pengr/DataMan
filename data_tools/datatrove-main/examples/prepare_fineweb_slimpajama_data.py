import os
import argparse
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

def main(args):
    # 确保输出目录存在
    if not args.json:
        os.makedirs(args.output, exist_ok=True)

    # 加载并合并数据集
    dataset = concatenate_datasets([
        load_dataset('json', data_files=path, split='train')
        for path in tqdm(args.input)
    ])

    # 将 metadata 展开为独立列
    def flatten_metadata(example):
        metadata = example.pop("metadata", {})
        example.update(metadata)
        return example
    
     # 通过 map 操作展平 metadata，并立即保存结果
    dataset = dataset.map(flatten_metadata, remove_columns=["metadata"], num_proc=args.num_workers)

    # 删除指定的顶级列
    dataset = dataset.remove_columns(args.columns_to_remove)
    
    # 根据选择保存为 jsonl 文件或其他格式
    if args.json:
        dataset.to_json(args.output, orient='records', lines=True)
    else:
        dataset.save_to_disk(args.output, num_proc=args.num_workers)
    print(f'Saved dataset to {args.output}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", help="Path to the input datasets.")
    parser.add_argument("--output", type=str, help="Path to the output dataset.")
    parser.add_argument("--num_workers", type=int, default=64, help="Workers for saving.")
    parser.add_argument("--columns_to_remove", type=str, nargs="+", default=[], help="Columns to remove or rename.")
    parser.add_argument("--json", action="store_true", help="save as json format.")
    args = parser.parse_args()
    main(args)
