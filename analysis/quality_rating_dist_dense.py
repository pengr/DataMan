import json
import os
from collections import Counter
from multiprocessing import Pool
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm

def process_batch(batch):
    # 统计所有域内的分数
    source_accuracy_counts = Counter()
    source_coherence_counts = Counter()
    source_language_consistency_counts = Counter()
    source_semantic_density_counts = Counter()
    source_knowledge_novelty_counts = Counter()
    source_topic_focus_counts = Counter()
    source_creativity_counts = Counter()
    source_professionalism_counts = Counter()
    source_style_consistency_counts = Counter()
    source_grammatical_diversity_counts = Counter()
    source_structural_standardization_counts = Counter()
    source_originality_counts = Counter()
    source_sensitivity_counts = Counter()
    
    for source_domain, accuracy, coherence, language_consistency, semantic_density, knowledge_novelty, \
    topic_focus, creativity, professionalism, style_consistency, grammatical_diversity, structural_standardization, \
    originality, sensitivity in zip(batch['source_domain'], 
                                        batch['accuracy'],
                                        batch['coherence'],
                                        batch['language_consistency'],
                                        batch['semantic_density'],
                                        batch['knowledge_novelty'],
                                        batch['topic_focus'],
                                        batch['creativity'],
                                        batch['professionalism'],
                                        batch['style_consistency'],
                                        batch['grammatical_diversity'],
                                        batch['structural_standardization'],
                                        batch['originality'],
                                        batch['sensitivity']):
        source_accuracy_counts[(source_domain, 'accuracy', accuracy)] += 1
        source_coherence_counts[(source_domain, 'coherence', coherence)] += 1
        source_language_consistency_counts[(source_domain, 'language_consistency', language_consistency)] += 1
        source_semantic_density_counts[(source_domain, 'semantic_density', semantic_density)] += 1
        source_knowledge_novelty_counts[(source_domain, 'knowledge_novelty', knowledge_novelty)] += 1
        source_topic_focus_counts[(source_domain, 'topic_focus', topic_focus)] += 1
        source_creativity_counts[(source_domain, 'creativity', creativity)] += 1
        source_professionalism_counts[(source_domain, 'professionalism', professionalism)] += 1
        source_style_consistency_counts[(source_domain, 'style_consistency', style_consistency)] += 1
        source_grammatical_diversity_counts[(source_domain, 'grammatical_diversity', grammatical_diversity)] += 1
        source_structural_standardization_counts[(source_domain, 'structural_standardization', structural_standardization)] += 1
        source_originality_counts[(source_domain, 'originality', originality)] += 1
        source_sensitivity_counts[(source_domain, 'sensitivity', sensitivity)] += 1

    return source_accuracy_counts, source_coherence_counts, source_language_consistency_counts, source_semantic_density_counts, \
        source_knowledge_novelty_counts, source_topic_focus_counts, source_creativity_counts, source_professionalism_counts, \
        source_style_consistency_counts, source_grammatical_diversity_counts, source_structural_standardization_counts, \
        source_originality_counts, source_sensitivity_counts

def process_dataset(input_paths):
    dataset = concatenate_datasets([load_from_disk(path) for path in tqdm(input_paths)])  # 合并所有数据集
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in  \
    ['text', 'source_domain', 'accuracy', 'coherence', 'language_consistency', 'semantic_density', 
    'knowledge_novelty', 'topic_focus', 'creativity', 'professionalism', 'style_consistency', 
    'grammatical_diversity', 'structural_standardization', 'originality', 'sensitivity', 'overall_score']])

    total_batches = (len(dataset) + 99999) // 100000
    with Pool(processes=128) as pool:
        results = list(tqdm(pool.imap(process_batch, dataset.iter(batch_size=100000)), 
                            total=total_batches, 
                            desc="Processing batches"))  

    total_source_accuracy_counts, total_source_coherence_counts, total_source_language_consistency_counts, total_source_semantic_density_counts, \
    total_source_knowledge_novelty_counts, total_source_topic_focus_counts, total_source_creativity_counts, total_source_professionalism_counts, \
    total_source_style_consistency_counts, total_source_grammatical_diversity_counts, total_source_structural_standardization_counts, \
    total_source_originality_counts, total_source_sensitivity_counts = Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), \
        Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter(),

    for source_accuracy_counts, source_coherence_counts, source_language_consistency_counts, source_semantic_density_counts, \
        source_knowledge_novelty_counts, source_topic_focus_counts, source_creativity_counts, source_professionalism_counts, \
        source_style_consistency_counts, source_grammatical_diversity_counts, source_structural_standardization_counts, \
        source_originality_counts, source_sensitivity_counts in results:
        total_source_accuracy_counts.update(source_accuracy_counts)
        total_source_coherence_counts.update(source_coherence_counts)
        total_source_language_consistency_counts.update(source_language_consistency_counts)
        total_source_semantic_density_counts.update(source_semantic_density_counts)
        total_source_knowledge_novelty_counts.update(source_knowledge_novelty_counts)
        total_source_topic_focus_counts.update(source_topic_focus_counts)
        total_source_creativity_counts.update(source_creativity_counts)
        total_source_professionalism_counts.update(source_professionalism_counts)
        total_source_style_consistency_counts.update(source_style_consistency_counts)
        total_source_grammatical_diversity_counts.update(source_grammatical_diversity_counts)
        total_source_structural_standardization_counts.update(source_structural_standardization_counts)
        total_source_originality_counts.update(source_originality_counts)
        total_source_sensitivity_counts.update(source_sensitivity_counts)

    return total_source_accuracy_counts, total_source_coherence_counts, total_source_language_consistency_counts, total_source_semantic_density_counts, \
    total_source_knowledge_novelty_counts, total_source_topic_focus_counts, total_source_creativity_counts, total_source_professionalism_counts, \
    total_source_style_consistency_counts, total_source_grammatical_diversity_counts, total_source_structural_standardization_counts, \
    total_source_originality_counts, total_source_sensitivity_counts

# 数据集路径, 是从slimpajama uniform 30B里统计的
input_paths = ["/mnt/nas/pengru.pr/data/DataMan/Slimpajama-627B/data_filtering/uniform/train_filtered/0"]
total_source_accuracy_counts, total_source_coherence_counts, total_source_language_consistency_counts, total_source_semantic_density_counts, \
    total_source_knowledge_novelty_counts, total_source_topic_focus_counts, total_source_creativity_counts, total_source_professionalism_counts, \
    total_source_style_consistency_counts, total_source_grammatical_diversity_counts, total_source_structural_standardization_counts, \
    total_source_originality_counts, total_source_sensitivity_counts = process_dataset(input_paths)

# 任意挑选一个指标，都可以统计source_domian的占比值
total_source_counts = Counter()
for (source_domain, _, _), count in total_source_accuracy_counts.items():
    total_source_counts[source_domain] += count

# 在代码开头添加输出文件的设置
output_file = "/mnt/nas/pengru.pr/checkpoints/DataMan/analysis/en/dense/quality_rating_dist.log"

with open(output_file, "w") as f:
    # 计算 domain 总数
    for criteria_name, total_source_criteria_counts in zip(['accuracy', 'coherence', 'language_consistency', 'semantic_density', 
        'knowledge_novelty', 'topic_focus', 'creativity', 'professionalism', 'style_consistency', 
        'grammatical_diversity', 'structural_standardization', 'originality', 'sensitivity'],
        [total_source_accuracy_counts, total_source_coherence_counts, total_source_language_consistency_counts, total_source_semantic_density_counts, \
        total_source_knowledge_novelty_counts, total_source_topic_focus_counts, total_source_creativity_counts, total_source_professionalism_counts, \
        total_source_style_consistency_counts, total_source_grammatical_diversity_counts, total_source_structural_standardization_counts, \
        total_source_originality_counts, total_source_sensitivity_counts]
        ):

        # 打印每个 source_domain 内，不同的criteria的 score 占比
        f.write(f"\nSource Domain, criteria {criteria_name} Score Proportions:\n")
        for (source_domain, criteria_name, criteria_value), count in total_source_criteria_counts.items():
            proportion = (count / total_source_counts[source_domain]) * 100
            f.write(f"{criteria_name} - {source_domain} - {criteria_value}: {count}, {proportion:.2f}%\n")