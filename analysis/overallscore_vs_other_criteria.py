import argparse
from datasets import load_from_disk, load_dataset
from scipy.stats import pearsonr, spearmanr
import numpy as np

def analyse_corr(args):
    if args.json:
        dataset = load_dataset('json', data_files=args.input, split="train")
    else:
        dataset = load_from_disk(args.input)
        
    results = []
    main_data = np.array(dataset[args.main_field]).astype(float)
    main_invalid_indices = np.where(np.isnan(main_data) | np.isinf(main_data))[0].tolist()        # 找到主数据中的无效索引
    assert np.unique(main_data).size > 1, f"Cannot calculate correlation for '{args.main_field}'. All values are the same."

    for other_field in args.other_fields:
        other_data = np.array(dataset[other_field]).astype(float)
        other_invalid_indices = np.where(np.isnan(other_data) | np.isinf(other_data))[0].tolist()  # 找到其他数据中的无效索引
        assert np.unique(other_data).size > 1, f"Cannot calculate correlation for '{other_field}'. All values are the same."

        # 合并无效索引并去重, 基于无效索引去删除无效数据
        all_invalid_indices = set(main_invalid_indices) | set(other_invalid_indices)
        main_data_cleaned = np.delete(main_data, list(all_invalid_indices))
        other_data_cleaned = np.delete(other_data, list(all_invalid_indices))

        # 计算相关性, 忽略p值
        pearson_corr = pearsonr(other_data_cleaned, main_data_cleaned)[0]
        spearman_corr = spearmanr(other_data_cleaned, main_data_cleaned)[0]
        results.append((other_field, pearson_corr, spearman_corr))

    # 按相关性绝对值排序
    sorted_results = sorted(results, key=lambda x: (abs(x[1]), abs(x[2])), reverse=True)
    
    with open(args.output, 'w', encoding="utf-8") as out_f:
        for other_field, pearson_corr, spearman_corr in sorted_results:
            out_f.write(f"{other_field}'s pearson_corr: {pearson_corr:.3f}, spearman_corr: {spearman_corr:.3f}\n")
    print(f"Correlation results saved to {args.output}")

def main():
    parser = argparse.ArgumentParser(description='Calculate correlations between columns in a dataset.')
    parser.add_argument("--input", type=str, help="Path to the input datasets.")
    parser.add_argument("--output", type=str, help="Path to the output tokenized dataset.")
    parser.add_argument("--main_field", type=str, help='The main column to correlate against.')
    parser.add_argument('--other_fields', nargs='+', type=str, help='Other columns to calculate correlations with.')
    parser.add_argument("--json", action="store_true", help="Input is json dataset.")
    args = parser.parse_args()
    analyse_corr(args)

if __name__ == "__main__":
    main()

# #!/bin/bash
# 用来统计预训练数据集中哪些质量指标与Overall Score相关性高

## Dense架构
# python /mnt/nas/pengru.pr/DataMan/analysis/en/overallscore_vs_other_criteria.py \
# --input /mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/validation_annotated_success \
# --output /mnt/nas/pengru.pr/checkpoints/DataMan/analysis/en/dense/overallscore_vs_other_criteria.log \
# --main_field overall_score \
# --other_fields accuracy coherence language_consistency semantic_density knowledge_novelty topic_focus creativity professionalism style_consistency grammatical_diversity structural_standardization originality sensitivity

## MoE架构
# python /mnt/nas/pengru.pr/DataMan/analysis/en/overallscore_vs_other_criteria.py \
# --input /mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/annotated/all_rating/moe/validation_annotated_success \
# --output /mnt/nas/pengru.pr/checkpoints/DataMan/analysis/en/moe/overallscore_vs_other_criteria.log \
# --main_field overall_score \
# --other_fields accuracy coherence language_consistency semantic_density knowledge_novelty topic_focus creativity professionalism style_consistency grammatical_diversity structural_standardization originality sensitivity
