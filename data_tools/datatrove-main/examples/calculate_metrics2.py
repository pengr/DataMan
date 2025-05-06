from datasets import load_dataset, concatenate_datasets
import pandas as pd
import warnings
import json

# 忽略不必要的警告
warnings.filterwarnings("ignore")

def load_and_label(paths, filtered_label):
    """
    加载 JSONL 文件并添加 'filtered' 标签。
    :param paths: JSONL 文件路径列表
    :param filtered_label: 真实标签，True 表示被过滤，False 表示被保留
    :return: 带有 'filtered' 标签的数据集
    """
    dataset = load_dataset("json", data_files=paths, split="train")
    
    def add_labels(example):
        example['filtered'] = filtered_label
        try:
            example['overall_score'] = float(example.get('overall_score', None))
        except (ValueError, TypeError):
            example['overall_score'] = None
        return example
    
    return dataset.map(add_labels)

def compute_metrics_for_type(filter_paths, keep_paths, data_type):
    """
    计算特定数据类型（fineweb 或 slimpajama）的精确率和召回率。
    
    :param filter_paths: 被过滤的文件路径列表
    :param keep_paths: 被保留的文件路径列表
    :param data_type: 数据类型名称（'fineweb' 或 'slimpajama'）
    """
    print(f"Processing data type: {data_type}\n{'='*50}")
    
    # 加载并标注过滤和保留的数据集
    filter_dataset = load_and_label(filter_paths, True)
    keep_dataset = load_and_label(keep_paths, False)
    
    # 合并数据集
    all_dataset = concatenate_datasets([filter_dataset, keep_dataset])
    
    # 过滤掉 'overall_score' 为 None 的记录
    all_dataset = all_dataset.filter(lambda x: x['overall_score'] is not None)
    
    # 计算预测标签：overall_score < 4.0 表示预测为低质量（filtered_pred=True）
    def compute_filtered_pred(example):
        example['filtered_pred'] = example['overall_score'] < 4.0
        return example
    
    all_dataset = all_dataset.map(compute_filtered_pred)
    
    # 将数据集转换为 Pandas DataFrame 以方便计算
    df = all_dataset.to_pandas()
    
    # 计算混淆矩阵
    TP = len(df[(df['filtered'] == True) & (df['filtered_pred'] == True)])
    FP = df[(df['filtered'] == False) & (df['filtered_pred'] == True)]
    FN = df[(df['filtered'] == True) & (df['filtered_pred'] == False)]
    TN = len(df[(df['filtered'] == False) & (df['filtered_pred'] == False)])
    
    # 保存 FP 和 FN 数据到 JSONL 文件
    fp_filename = f"{data_type}_FP.jsonl"
    fn_filename = f"{data_type}_FN.jsonl"
    
    with open(fp_filename, 'w') as fp_file:
        for record in FP.to_dict(orient='records'):
            fp_file.write(json.dumps(record) + '\n')
    
    with open(fn_filename, 'w') as fn_file:
        for record in FN.to_dict(orient='records'):
            fn_file.write(json.dumps(record) + '\n')
    
    print(f"FP cases saved to: {fp_filename}")
    print(f"FN cases saved to: {fn_filename}")
    
    # 计算精确率和召回率
    accuracy = (TP + TN) / (TP + TN + len(FP) + len(FN))
    precision = TP / (TP + len(FP)) if (TP + len(FP)) > 0 else 0
    recall = TP / (TP + len(FN)) if (TP + len(FN)) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 输出结果
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {len(FP)}")
    print(f"False Negatives (FN): {len(FN)}")
    print(f"True Negatives (TN): {TN}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print("-"*50 + "\n")
    
    return accuracy, precision, recall, f1_score

def main():
    # 定义 fineweb 和 slimpajama 的过滤和保留文件路径
    fineweb_filter_paths = [
        "/mnt/nas/pengru.pr/data/RedPajama-Data-1T/en_head_0000.json.gz.dedup.classifier.norm.fineweb_filter.annotated.jsonl",
        "/mnt/nas/pengru.pr/data/RedPajama-Data-1T/en_head_0001.json.gz.dedup.classifier.norm.fineweb_filter.annotated.jsonl"
    ]
    
    fineweb_keep_paths = [
        "/mnt/nas/pengru.pr/data/RedPajama-Data-1T/en_head_0000.json.gz.dedup.classifier.norm.fineweb_keep.annotated.jsonl",
        "/mnt/nas/pengru.pr/data/RedPajama-Data-1T/en_head_0001.json.gz.dedup.classifier.norm.fineweb_keep.annotated.jsonl"
    ]
    
    slimpajama_filter_paths = [
        "/mnt/nas/pengru.pr/data/RedPajama-Data-1T/en_head_0000.json.gz.dedup.classifier.norm.slimpajama_filter.annotated.jsonl",
        "/mnt/nas/pengru.pr/data/RedPajama-Data-1T/en_head_0001.json.gz.dedup.classifier.norm.slimpajama_filter.annotated.jsonl"
    ]
    
    slimpajama_keep_paths = [
        "/mnt/nas/pengru.pr/data/RedPajama-Data-1T/en_head_0000.json.gz.dedup.classifier.norm.slimpajama_keep.annotated.jsonl",
        "/mnt/nas/pengru.pr/data/RedPajama-Data-1T/en_head_0001.json.gz.dedup.classifier.norm.slimpajama_keep.annotated.jsonl"
    ]
    
    # 计算 fineweb 的精确率和召回率
    fineweb_accuracy, fineweb_precision, fineweb_recall, fineweb_f1_score = compute_metrics_for_type(
        fineweb_filter_paths,
        fineweb_keep_paths,
        "fineweb"
    )
    
    # 计算 slimpajama 的精确率和召回率
    slimpajama_accuracy, slimpajama_precision, slimpajama_recall, slimpajama_f1_score = compute_metrics_for_type(
        slimpajama_filter_paths,
        slimpajama_keep_paths,
        "slimpajama"
    )
    
    # 汇总结果
    print("### 汇总结果 ###")
    print(f"Fineweb Accuracy: {fineweb_accuracy:.2f}")
    print(f"Fineweb Precision: {fineweb_precision:.2f}")
    print(f"Fineweb Recall: {fineweb_recall:.2f}")
    print(f"Fineweb F1 Score: {fineweb_f1_score:.2f}")
    print(f"Slimpajama Accuracy: {slimpajama_accuracy:.2f}")
    print(f"Slimpajama Precision: {slimpajama_precision:.2f}")
    print(f"Slimpajama Recall: {slimpajama_recall:.2f}")
    print(f"Slimpajama F1 Score: {slimpajama_f1_score:.2f}")

if __name__ == "__main__":
    main()
