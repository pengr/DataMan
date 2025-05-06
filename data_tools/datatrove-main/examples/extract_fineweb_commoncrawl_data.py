import json
import argparse
from datasets import load_dataset, Dataset

def main(args):
    # 加载数据集
    dataset_a = load_dataset('json', data_files=args.path_a, split='train')
    dataset_b = load_dataset('json', data_files=args.path_b, split='train')

    # 提取source字段到set中
    sources_b = set(dataset_b['source'])

    # 使用filter找出在A中而不在B中的记录
    filter_dataset = dataset_a.filter(lambda example: example['source'] not in sources_b)

    # 根据用户选择保存格式
    if args.json:
        filter_dataset.to_json(args.output, orient='records', lines=True)
    else:
        filter_dataset.save_to_disk(args.output, num_proc=args.num_workers)
    print(f'Saved dataset to {args.output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find unique entries in A not in B.')
    parser.add_argument('--path_a', type=str, help='Path or name of the dataset A')
    parser.add_argument('--path_b', type=str, help='Path or name of the dataset B')
    parser.add_argument("--json", action="store_true", help="save as json format.")
    parser.add_argument('--output', type=str, help='Path to save the output dataset')
    args = parser.parse_args()
    main(args)