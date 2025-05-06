import json
import argparse
from datasets import load_from_disk, load_dataset, concatenate_datasets
from multiprocessing import Process, Lock
from tqdm import tqdm
import os

def check_jsonl_file(input_file):
    """验证JSONL文件中的ID是否唯一且连续."""
    id_set = set()
    with open(input_file, 'r', encoding='utf-8') as file:
        for lineno, line in enumerate(file, 1):
            line = line.strip()
            try:
                data = json.loads(line)
                if 'id' in data:
                    # 提取并添加 id 值
                    id_value = int(data['id'].split('_')[-1])
                    id_set.add(id_value)
            except json.JSONDecodeError as e:
                print(f"Error in line {lineno}: {e}")
    if len(id_set) == lineno and max(id_set) == lineno:
        print(f"All ids from 1 to {lineno} are present and unique.")
    else:
        print(f"Error: IDs are not consecutive or unique.")
    print(f"{input_file}, Number of lines: {lineno}")

def remove_jsonl_columns(input_file, output_file_path, columns):
    dataset = load_dataset('json', data_files=input_file, split="train")  # 加载 JSONL 文件为 Dataset 对象
    dataset = dataset.remove_columns(columns)   # 移除指定的列
    dataset.to_json(output_file_path, lines=True)   # 以JSONL格式保存更新的数据集, lines=True代表每个数据条目将作为独立的行写入文件
    print(f"Removed dataset saved to {output_file_path}")

def rename_jsonl_columns(input_file, output_file_path, old_columns, new_columns):
    dataset = load_dataset('json', data_files=input_file, split="train")  # 加载 JSONL 文件为 Dataset 对象
    for old_col, new_col in zip(old_columns, new_columns):
        dataset = dataset.rename_column(old_col, new_col)   # 移除指定的列
    dataset.to_json(output_file_path, lines=True)           # 以JSONL格式保存更新的数据集, lines=True代表每个数据条目将作为独立的行写入文件
    print(f"Renamed dataset saved to {output_file_path}")

def filter_jsonl_lines(input_file, output_file_path, columns):
    assert len(columns) == 1, "Only support single column"
    dataset = load_dataset('json', data_files=input_file, split="train")  # 加载 JSONL 文件为 Dataset 对象
    dataset = dataset.filter(lambda example: example[columns[0]] == "5") # 过滤出质量总分为5的行
    print(f"Filtered dataset size: {len(dataset)}")
    dataset.to_json(output_file_path, lines=True)  # 以JSONL格式保存更新的数据集
    print(f"Filtered dataset saved to {output_file_path}")

def check_jsonl_columns(input_file, output_dir, columns):
    dataset = load_dataset('json', data_files=input_file, split="train")   # 加载 JSONL 文件为 Dataset 对象
    source, extension = os.path.splitext(os.path.basename(input_file))     # 获取输入文件名和扩展名                                      
    success_file = os.path.join(output_dir, f"{source}_success.jsonl")     
    failed_file = os.path.join(output_dir, f"{source}_failed.jsonl")
   
    is_success = lambda example: all(example[col] and example[col].isdigit() and 1 <= int(example[col]) <= 5 for col in columns) # 使用 lambda 函数过滤数据
    failed_dataset = dataset.filter(lambda example: not is_success(example)) # 过滤失败的数据集

    if len(failed_dataset) > 0:                             # 仅当失败数据集不为空, 才执行如下操作： 
        success_dataset = dataset.filter(is_success)        # 过滤成功的数据集
        success_dataset.to_json(success_file, lines=True)   # 保存成功数据集
        failed_dataset.to_json(failed_file, lines=True)     # 保存失败数据集
        print(f"Success dataset saved to {success_file} with {len(success_dataset)} records.")
        print(f"Failed dataset saved to {failed_file} with {len(failed_dataset)} records.")
    else:
        print(f"{input_file} columns check is valid. No failed records found.")

def check_hf_columns(input_file, output_dir, columns, num_workers):
    dataset = load_from_disk(input_file)
    source, extension = os.path.splitext(os.path.basename(input_file))  # 获取输入文件名和扩展名
    success_file = os.path.join(output_dir, f"{source}_success")
    failed_file = os.path.join(output_dir, f"{source}_failed")
    
    def is_success(example):
        success = all(example[col] and example[col].isdigit() and 1 <= int(example[col]) <= 5 for col in columns)
        return {"success": success}

    # Use map with multiprocessing to determine success or failure
    results = dataset.map(is_success, num_proc=num_workers)
    # Get the indices of success and failure
    failed_indices = [i for i, success in enumerate(results['success']) if not success]

    # If there are failed data, save success and failed datasets
    if failed_indices:
        success_indices = [i for i, success in enumerate(results['success']) if success]
        success_dataset = dataset.select(success_indices)
        failed_dataset = dataset.select(failed_indices)
        success_dataset.save_to_disk(success_file, num_proc=num_workers)
        failed_dataset.save_to_disk(failed_file, num_proc=1)  # 防止出现个位数样本的failed_dataset对象,报错：IndexError: Index 3 out of range for dataset of size 3.
        print(f"Success dataset saved to {success_file} with {len(success_dataset)} records.")
        print(f"Failed dataset saved to {failed_file} with {len(failed_dataset)} records.")
    else:
        print(f"{input_file} columns check is valid. No failed records found.")
        dataset.save_to_disk(success_file, num_proc=num_workers)
        print(f"Success dataset saved to {success_file} with {len(dataset)} records.")

def json_to_dataset(input_file, output_dir, num_workers):
    datasets = []
    dataset = load_dataset('json', data_files=input_file, split="train")  # 加载 JSONL 文件为 Dataset 对象
    dataset.save_to_disk(output_dir, num_proc=num_workers)
    print(f"Combined dataset saved to {output_dir}")

def write_row_to_jsonl(row):
    """将单行数据格式化为JSON字符串."""
    return json.dumps(row, ensure_ascii=False)

def shard_to_jsonl(shard, output_file_path, batch_size, lock, id):
    """将数据集的一个片段逐批保存到一个输出文件中."""
    # 遍历数据集片段，分块处理
    for start in tqdm(range(0, len(shard), batch_size), desc=f"Processing Shard {id}"):
        end = min(start + batch_size, len(shard))
        batch = shard.select(range(start, end))            # 使用 select 方法读取当前批次数据
        json_lines = list(map(write_row_to_jsonl, batch))  # 将行数据转换为JSON字符串
        with lock:   # 使用锁来安全地写入文件
            with open(output_file_path, 'a', encoding='utf-8') as output_file:  # 以追加模式打开文件
                output_file.write('\n'.join(json_lines) + '\n')

def dataset_to_jsonl(input_file, output_file_path, num_workers, batch_size):
    dataset = load_from_disk(input_file)  # 从指定路径加载Dataset对象
    lock = Lock()   # 创建一个锁
    processes = []  # 创建进程列表
    # 创建进程, 启动进程
    for id in range(num_workers):
        shard = dataset.shard(num_shards=num_workers, index=id, contiguous=True)
        process = Process(target=shard_to_jsonl, args=(shard, output_file_path, batch_size, lock, id))
        processes.append(process)
        process.start()  
    # 等待所有进程完成
    for process in processes:
        process.join()   


def main():
    parser = argparse.ArgumentParser(description='Processing between JSONL files and Huggingface Dataset files.')
    parser.add_argument('--input', type=str, help='Paths to the input JSONL file or saved dataset.')
    parser.add_argument('--output', type=str, help='Path to the output JSONL file or the output directory for datasets.')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of processes.')
    parser.add_argument('--batch_size', type=int, default=10000, help='Number of rows to process in each batch.')
    parser.add_argument("--columns", type=str, nargs="+", default=None, help="Columns to remove or rename.")
    parser.add_argument("--new_columns", type=str, nargs="+", default=None, help="New Columns name.")
    parser.add_argument('--strategy', type=str, choices=['check_json', 'remove_columns', 'rename_columns', 'filter_jsonl_lines',
                         'check_jsonl_columns', 'check_hf_columns', 'json_to_dataset', 'dataset_to_json'], 
                        required=True, help="Process strategy")
    args = parser.parse_args()
    
    if args.strategy == "check_json":             # 检查JSONL文件，并验证文件中的ID是否唯一且连续
        check_jsonl_file(args.input)
    elif args.strategy == 'remove_columns':       # 移除JSONL文件的某些列，重新存为新的JSONL文件
        remove_jsonl_columns(args.input, args.output, args.columns)
    elif args.strategy == 'rename_columns':       # 更名JSONL文件的某些列，重新存为新的JSONL文件
        rename_jsonl_columns(args.input, args.output, args.columns, args.new_columns)
    elif args.strategy == 'filter_jsonl_lines':         # 筛选JSONL文件的某些行，重新存为新的JSONL文件
        filter_jsonl_lines(args.input, args.output, args.columns)
    elif args.strategy == 'check_jsonl_columns':  # 检查JSONL文件的某些列，成功则存为xx_success.jsonl,失败则存为xx_failed.jsonl
        check_jsonl_columns(args.input, args.output, args.columns)
    elif args.strategy == 'check_hf_columns':  # 检查HF Arrow文件的某些列，成功则存为xx_success,失败则存为xx_failed
        check_hf_columns(args.input, args.output, args.columns, args.num_workers)
    elif args.strategy == "json_to_dataset":      # 从JSONL文件生成Datasets对象并存储
        json_to_dataset(args.input, args.output, args.num_workers)
    elif args.strategy == "dataset_to_json":      # 将存储的Datasets对象转为JSONL文件
        dataset_to_jsonl(args.input, args.output, args.num_workers, args.batch_size)
    else:
        raise ValueError("Unsupported strategy")

if __name__ == '__main__':
    main()
