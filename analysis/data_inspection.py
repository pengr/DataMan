import os
import argparse
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm

def process_datasets(input_paths, criteria_list, output_path):
    # 合并所有数据集, 移除不需要的列
    dataset = concatenate_datasets([load_from_disk(path) for path in tqdm(input_paths)])  
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in  \
        ['text', 'source_domain', 'accuracy', 'coherence', 'language_consistency', 'semantic_density', 
        'knowledge_novelty', 'topic_focus', 'creativity', 'professionalism', 'style_consistency', 
        'grammatical_diversity', 'structural_standardization', 'originality', 'sensitivity', 'overall_score']])  

    source_domain_list = ["CommonCrawl", "C4", "ArXiv", "Book", "Github", "Wikipedia", "StackExchange"]

    # 准备所有的过滤条件
    conditions = []
    for criteria in criteria_list:
        for source_domain in source_domain_list:
            for criteria_value in range(1, 6):
                conditions.append({
                    "criteria": criteria,
                    "domain": source_domain,
                    "criteria_value": str(criteria_value),
                    "output": os.path.join(output_path, f'{criteria}_{source_domain}_{criteria_value}')
                })

    # 利用多线程处理过滤和保存
    for cond in tqdm(conditions):
        filtered_dataset = dataset.filter(
            lambda example: (
                (example["source_domain"] == cond["domain"] and example[cond["criteria"]] == cond["criteria_value"])
            ),
            num_proc=32  # 使用多进程
        )

        print(f"Filtered dataset size for {cond['output']}: {len(filtered_dataset)}")
        if len(filtered_dataset) > 0:
            filtered_dataset.save_to_disk(cond["output"], num_proc=4)
            print(f"Filtered dataset saved to {cond['output']}")
        else:
            print(f"No data to save for {cond['output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets")
    parser.add_argument("--input_paths", type=str, nargs='+', help="List of input dataset paths")
    parser.add_argument("--criteria_list", type=str, nargs='+', help="List of criteria to filter datasets")
    parser.add_argument("--output_path", type=str, help="Path to the output.")
    args = parser.parse_args()
    process_datasets(args.input_paths, args.criteria_list)