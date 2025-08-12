import sys
import os
import re
import json
import argparse
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ENGLISH
# Regular expression: matches ellipses surrounded by double quotes, matches URL items starting with ![](//) characters, score, domain type, raw text, N/A
CATEGORY_DICT_EN = {"A":"Medicine","B":"Finance","C":"Law","D":"Education","E":"Technology","F":"Entertainment","G":"Mathematics","H":"Coding",
                    "I":"Government","J":"Culture","K":"Transportation","L":"Retail E-commerce","M":"Telecommunication","N":"Agriculture","O":"Other"}
## 适用于gpt-4-turbo
# SCORE_PATTERN_EN = r"(?:\[[1-9][0-4]?\]|\b[1-9][0-4]?\.|\b[1-9][0-4]?\))[^:]+:\**\s\**(\d+(?:\.\d+)?)/5|\b[1-9][0-4]?\.[^:]+\s\((\d+(?:\.\d+)?)/5\)"
## 适用于gpt-4o
SCORE_PATTERN_EN = r"(?:\[[1-9][0-4]?\]|\b[1-9][0-4]?\.|\b[1-9][0-4]?\)).*(?:\*+|\s+|\(+)(\d+(\.\d+)?)/5"
CATEGORY_PATTERN_EN = r'\[\s*([A-Z])\s*\]|(?<=\s)([A-Z])(?=\])(?!\]\])|\bDomain Type:\**\s*\**([A-Z])\s*(?=[\(\)\*\-])'
METRIC_LIST = ["Accuracy", "Coherence", "Language Consistency", "Semantic Density", "Knowledge Novelty", 
"Topic Focus", "Creativity", "Professionalism", "Style Consistency", "Grammatical Diversity", 
"Structural Standardization", "Originality", "Sensitivity", "Overall Score"]

# CHINESE
# 正则表达式：评分、领域类型、原始文本、N/A、双引号包围的单个或多个省略号、以 http:// 或 https:// 开头并以合法URL字符结束的项
CATEGORY_DICT_ZH = {"A":"医疗","B":"金融","C":"法律","D":"教育","E":"科技","F":"娱乐","G":"数学","H":"代码",
                    "I":"政务","J":"文化","K":"交通","L":"零售电商","M":"电信","N":"农业","O":"其他"}
SCORE_PATTERN_ZH = r"\[[1-9][0-4]?][^：]+：(\d+(?:\.\d+)?)/5"
CATEGORY_PATTERN_ZH = '领域类型.*?[\[\]]?([A-Z])[\]\[]?'

# 设置Pandas显示格式，确保在控制台打印时不会省略任何行和列，并且小数点保留两位
pd.set_option('display.max_rows', None)  # 设置显示的最大行数
pd.set_option('display.max_columns', None)  # 设置显示的最大列数
pd.set_option('display.width', None)  # 设置打印宽度
pd.set_option('display.precision', 2)  # 设置小数点保留两位
pd.set_option('display.colheader_justify', 'center')  # 设置列标题居中

def fix_seed(seed=1024):
    assert isinstance(seed, int)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_request_template_and_pattern(lang):
    global CATEGORY_DICT, SCORE_PATTERN, CATEGORY_PATTERN, METRIC_LIST
    if lang == 'en':
        CATEGORY_DICT = CATEGORY_DICT_EN
        SCORE_PATTERN = SCORE_PATTERN_EN
        CATEGORY_PATTERN = CATEGORY_PATTERN_EN
    elif lang == 'zh':
        CATEGORY_DICT = CATEGORY_DICT_ZH
        SCORE_PATTERN = SCORE_PATTERN_ZH
        CATEGORY_PATTERN = CATEGORY_PATTERN_ZH
    else:
        raise ValueError(f"Unsupported language: {lang}")

def three_classes(score):
    if score < 3:
        return '差'  # Poor
    elif score > 3:
        return '好'  # Good
    else:
        return '中'  # Average

def two_classes(score):
    return '好' if score >= 3 else '差'  # Good or Poor

def analyse_sft_data(input_path, sample_size, seed, output_path):
    output_file = open(output_path, 'w', encoding='utf-8')
    data_rows = []  
    with open(input_path, 'r', encoding='utf-8') as rf:
        for id, line in enumerate(rf):
            data = json.loads(line)
            source = data["source"].rpartition('_')[0]
            gpt_request = data['messages'][1]['content']
            gpt_response = data['messages'][2]['content']
            scores = gpt_response.split("\n")[1:]
            category = CATEGORY_DICT[gpt_response.split("\n")[0]]
            row = [source, category] + list(map(float, scores))
            if len(row) != 16:  # final_finetune_bal_q1_train中的'en_cc_v5_2017bot_89917'多了一个overall_score分数
                row = row[:16]
            assert len(row) == 16, "只有16列"
            data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=['source', 'category'] + METRIC_LIST)
    def custom_sample(group, n, seed):
        return group.sample(n=min(n, len(group)), random_state=seed)
    df = df.groupby('source').apply(lambda x: custom_sample(x, sample_size, seed)).reset_index(drop=True)
    total_samples = df.shape[0]
    output_file.write(f"随机种子 {seed}, 每一个源采样 {sample_size} 条数据, 总样本量 {total_samples}\n")

    # 计算 Pearson 相关矩阵
    correlation_matrix = df[METRIC_LIST].corr(method='pearson')
    # 将相关性矩阵中的数值转换为小数形式
    correlation_matrix = correlation_matrix.round(2)
    output_file.write(str(correlation_matrix) + "\n")

    # 绘制热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Blues', 
        square=True, cbar_kws={"shrink": .8}, linewidths=.5, 
        annot_kws={"size": 10, "color": "black"}, 
        cbar=False)
    # 设置横坐标刻度的角度为45度
    plt.xticks(rotation=45, ha='right')  # ha='right' 确保文本对齐

    # 保存热图
    plt.savefig(os.path.join(os.path.dirname(output_path), 'quality_rating_corr.pdf'), bbox_inches='tight', pad_inches=0)
    # 显示热图
    plt.show()

    output_file.write("\n\n##################### 从整体的角度 #####################\n")
    ## 1
    output_file.write("1. 所有评价指标(含总评分)的平均分及方差，从高到低排名：\n")
    metric_avg_scores = df[METRIC_LIST].mean().sort_values(ascending=False)
    metric_variances = df[METRIC_LIST].var().reindex(metric_avg_scores.index)  # 计算方差并按照平均分的索引排序
    output_file.write(pd.concat([metric_avg_scores, metric_variances], axis=1, keys=['Avg Score', 'Variance']).to_string() + "\n")

    ## 2
    output_file.write("\n\n2. 所有分数段(1极差,2差,3一般,4好,5极好)的样本量及占比，从高到低排名：\n")
    output_file.write(f"Score scale   Sample Number   Sample Ratio\n")
    score_counts = df[METRIC_LIST[-1]].value_counts(normalize=True).sort_index() * 100
    for score, percent in score_counts.sort_values(ascending=False).items():
        output_file.write(f"{score}           {int(percent * total_samples / 100)}           {percent:.2f}%\n")

    output_file.write("\n\n##################### 从source的角度 #####################\n")
    ## 3
    output_file.write("3. 每个源中的样本数：\n")
    source_sample_counts = df.groupby('source').size()
    for source, count in source_sample_counts.items():
        output_file.write(f"{source}  {count}\n")

    ## 3
    output_file.write("\n\n3. 所有源的所有评价指标(含总评分)的平均分：\n")
    source_metric_avg_scores = df.groupby('source')[METRIC_LIST].mean().sort_values(by=METRIC_LIST[-1], ascending=False)
    output_file.write(source_metric_avg_scores.to_string() + "\n")

    ## 4
    source_metric_avg_scores = df.groupby('source')[METRIC_LIST].mean()
    for metric in METRIC_LIST:
        output_file.write(f"\n\n4. 每一个源的评价指标“{metric}”的平均分，从高到低排名：\n")
        sorted_scores = source_metric_avg_scores[metric].sort_values(ascending=False)
        output_file.write(sorted_scores.to_frame().to_string() + "\n")

    ## 5
    sources = df['source'].unique()
    for source in sources:
        output_file.write(f'\n\n5. 源“{source}”的每一项评价指标的平均分，从高到低排名：\n')
        output_file.write(f"                 {source}\n")
        output_file.write("Metric                    \n")
        source_df = df[df['source'] == source]
        metric_avg_scores = source_df[METRIC_LIST].mean().sort_values(ascending=False)
        output_file.write(metric_avg_scores.to_string() + "\n")

    ## 6
    output_file.write("\n\n6. 所有源的所有领域的样本占比，从高到低排名：\n")
    source_category_counts = df.groupby('source')['category'].value_counts(normalize=True) * 100
    source_category_counts = source_category_counts.sort_values(ascending=False).reset_index(name='Sample Ratio')
    output_file.write(source_category_counts.to_string(index=False) + "\n")

    ## 7
    output_file.write("\n\n7. 同一个源的所有领域的样本量及占比，从高到低排名：\n")
    source_category_counts = df.groupby(['source', 'category']).size().reset_index(name='Sample Number')
    source_total_samples = df.groupby(['source']).size().reset_index(name='Total')
    source_category_stats = pd.merge(source_category_counts, source_total_samples, on='source')
    source_category_stats['Sample Ratio'] = source_category_stats['Sample Number'] / source_category_stats['Total']
    sorted_stats = source_category_stats.sort_values(by=['source', 'Sample Number'], ascending=[True, False])
    sorted_stats['Sample Ratio'] = sorted_stats['Sample Ratio'].apply(lambda x: f'{x:.2%}')
    output_file.write(sorted_stats[['source', 'category', 'Sample Number', 'Sample Ratio']].to_string(index=False) + "\n")

    output_file.write("\n\n##################### 从domain的角度 #####################\n")
    ## 8
    domain_metric_avg_scores = df.groupby('category')[METRIC_LIST].mean()
    for metric in METRIC_LIST:
        output_file.write(f"\n\n8. 每一个领域的评价指标“{metric}”的平均分，从高到低排名：\n")
        sorted_scores = domain_metric_avg_scores[metric].sort_values(ascending=False)
        output_file.write(sorted_scores.to_frame().to_string() + "\n")

    ## 9
    categories = df['category'].unique()
    for category in categories:
        output_file.write(f'\n\n9. 领域“{category}”的每一项评价指标的平均分，从高到低排名：\n')
        output_file.write(f"                 {category}\n")
        output_file.write("Metric                    \n")
        category_df = df[df['category'] == category]
        metric_avg_scores = category_df[METRIC_LIST].mean().sort_values(ascending=False)
        output_file.write(metric_avg_scores.to_string() + "\n")

    ## 10
    output_file.write("\n\n10. 所有领域的样本量及其占比，从高到低排名：\n")
    category_counts = df['category'].value_counts().reset_index(name='Sample Number')     # 计算每个领域的样本数量
    category_counts.columns = ['Category', 'Sample Number']  # 重命名列以避免冲突
    category_counts['Sample Ratio'] = (category_counts['Sample Number'] / df.shape[0]).round(4) # 计算样本比例
    category_counts.sort_values(by='Sample Number', ascending=False, inplace=True) # 排序结果
    category_counts['Sample Ratio'] = category_counts['Sample Ratio'].apply(lambda x: f'{x:.2%}') # 格式化样本比例为百分比形式
    output_file.write(category_counts.to_string(index=False) + "\n")

    ## 11 分析每个'source'内各个'Overall Score'的分布和占比 
    output_file.write("\n\n11. 分析每个'source'内各个'Overall Score'的分布和占比：\n")
    scores_distribution = df.groupby('source')[METRIC_LIST[-1]].value_counts(normalize=True)
    for source, distribution in scores_distribution.groupby(level=0):
        output_file.write(f"'source': {source}\n")
        for score, percentage in distribution.items():
            output_file.write(f"Score: {score}, Percentage: {percentage:.2%}\n")
        output_file.write("\n")

    ## 12 分析每个'source'内'Overall Score'二分类的分布和占比
    # 转换分数，并计算分类频次和占比
    output_file.write("\n\n12. 二分类中 '好' 和 '差' 的百分比差值：\n")
    df['Two_Class_Score'] = df[METRIC_LIST[-1]].apply(two_classes)
    two_class_distribution = df.groupby('source')['Two_Class_Score'].value_counts(normalize=True)

    # 在二分类中计算'好'和'差'的占比差值并按差值排序
    difference_dict = {}
    for source, distribution in two_class_distribution.groupby(level=0):
        df_distribution = distribution.reset_index(level=0, drop=True)  # 重置索引，丢弃source级别，只保留分类标签
        good_percentage, poor_percentage = 0.0, 0.0
        # 使用索引检查分类是否存在，并赋值
        if '好' in df_distribution.index:
            good_percentage = df_distribution['好']
        if '差' in df_distribution.index:
            poor_percentage = df_distribution['差']
        difference = good_percentage - poor_percentage
        difference_dict[source] = {
            '好': good_percentage,
            '差': poor_percentage,
            '差值': difference
        }

    # 根据 '差值' 由高到低排序，获取已排序的 'source'
    sorted_sources = sorted(difference_dict.items(), key=lambda item: item[1]['差值'], reverse=True)

    # 打印出排序后的结果
    for source, values in sorted_sources:
        output_file.write(f"'source': {source} - 二分类\n")
        output_file.write(f"Class Good Percentage: {values['好']:.2%}\n")
        output_file.write(f"Class Poor Percentage: {values['差']:.2%}\n")
        output_file.write(f"Difference: {values['差值']:.2%}\n\n")

def analyse_sft_data_discrepancy(input_path, sample_size1, sample_size2, seed, output_path):
    output_file = open(output_path, 'w', encoding='utf-8')
    data_rows = []  
    with open(input_path, 'r', encoding='utf-8') as rf:
        for id, line in enumerate(rf):
            data = json.loads(line)
            source = data["source"].rpartition('_')[0]
            gpt_request = data['messages'][1]['content']
            gpt_response = data['messages'][2]['content']
            scores = gpt_response.split("\n")[1:]
            category = CATEGORY_DICT[gpt_response.split("\n")[0]]
            row = [source, category] + list(map(float, scores))
            if len(row) != 16:  # final_finetune_bal_q1_train中的'en_cc_v5_2017bot_89917'多了一个overall_score分数
                row = row[:16]
            assert len(row) == 16, "只有16列"
            data_rows.append(row)

    # 制作成pandas的DataFrame格式
    df = pd.DataFrame(data_rows, columns=['source', 'category'] + METRIC_LIST)
    def custom_sample(group, n, seed):
        return group.sample(n=min(n, len(group)), random_state=seed)
    df1 = df.groupby('source').apply(lambda x: custom_sample(x, sample_size1, seed)).reset_index(drop=True)
    df2 = df.groupby('source').apply(lambda x: custom_sample(x, sample_size2, seed)).reset_index(drop=True)
    total_samples1, total_samples2 = df1.shape[0], df2.shape[0]
    output_file.write(f"随机种子 {seed}, 每一个源采样 {sample_size1} 和 {sample_size2} 条数据, 总样本量 {total_samples1} 和 {total_samples2}\n")

    output_file.write("\n\n##################### 从整体的角度 #####################\n")
    ## 1
    output_file.write("1. 所有评价指标(含总评分)的平均分，从高到低排名：\n")
    metric_avg_scores1 = df1[METRIC_LIST].mean()
    metric_avg_scores2 = df2[METRIC_LIST].mean()
    metric_avg_scores_disc = metric_avg_scores1 - metric_avg_scores2
    result = pd.DataFrame({
        'Metric': METRIC_LIST,
        f'Sampling{sample_size1}': metric_avg_scores1,
        f'Sampling{sample_size2}': metric_avg_scores2,
        'Discrepancy': metric_avg_scores_disc
    })  # 创建输出 DataFrame
    result.sort_values(by=f'Sampling{sample_size1}', ascending=False, inplace=True)  # 根据第一个样本大小的平均分进行排序
    output_file.write(result.to_string(index=False) + "\n")

    ## 2
    output_file.write("\n\n2. 所有分数段(1极差,2差,3一般,4好,5极好)的样本量及占比，从高到低排名：\n")
    output_file.write(f"Score scale   Sampling{sample_size1}    Sampling{sample_size2}   Discrepancy\n")
    score_counts1 = df1[METRIC_LIST[-1]].value_counts(normalize=True).sort_index() * 100
    score_counts2 = df2[METRIC_LIST[-1]].value_counts(normalize=True).sort_index() * 100
    for score, percent1 in score_counts1.sort_values(ascending=False).items():
        percent2 = score_counts2[score] if score in score_counts2 else 0
        output_file.write(f"{score}\t\t\t\t{int(percent1 * total_samples1 / 100)}, {percent1:.2f}%\t\t\t\t{int(percent2 * total_samples2 / 100)}, {percent2:.2f}%\t\t\t\t{int(percent1 * total_samples1 / 100)-int(percent2 * total_samples2 / 100)}, {(percent1-percent2):.2f}%\n")

    output_file.write("\n\n##################### 从source的角度 #####################\n")
    ## 3
    output_file.write("3. 所有源的所有评价指标(含总评分)的平均分：\n")
    output_file.write(f'Sampling{sample_size1}\n')
    source_metric_avg_scores1 = df1.groupby('source')[METRIC_LIST].mean().sort_values(by=METRIC_LIST[-1], ascending=False)
    output_file.write(source_metric_avg_scores1.to_string() + "\n")

    output_file.write(f'Sampling{sample_size2}\n')
    source_metric_avg_scores2 = df2.groupby('source')[METRIC_LIST].mean().sort_values(by=METRIC_LIST[-1], ascending=False)
    output_file.write(source_metric_avg_scores2.to_string() + "\n")

    output_file.write('Discrepancy\n')
    source_metric_avg_scores_disc = source_metric_avg_scores1 - source_metric_avg_scores2
    output_file.write(source_metric_avg_scores_disc.to_string() + "\n")

    ## 4
    source_metric_avg_scores1 = df1.groupby('source')[METRIC_LIST].mean()
    source_metric_avg_scores2 = df2.groupby('source')[METRIC_LIST].mean()
    discrepancy = source_metric_avg_scores1 - source_metric_avg_scores2
    for metric in METRIC_LIST:
        output_file.write(f"\n\n4. 每一个源的评价指标“{metric}”的平均分，从高到低排名：\n")
        combined_df = pd.concat([
            source_metric_avg_scores1[metric].rename(f"Sampling{sample_size1}"),
            source_metric_avg_scores2[metric].rename(f"Sampling{sample_size2}"),
            discrepancy[metric].rename("Discrepancy")], axis=1)
        sorted_combined_df = combined_df.sort_values(by=f"Sampling{sample_size1}", ascending=False)
        output_file.write(sorted_combined_df.to_string() + "\n")

    ## 5
    sources = df['source'].unique() # 获取所有的源
    for source in sources:
        output_file.write(f'\n\n5. 源“{source}”的每一项评价指标的平均分，从高到低排名：\n')
        combined_df = pd.DataFrame({
            f"Sampling{sample_size1}": source_metric_avg_scores1.loc[source],
            f"Sampling{sample_size2}": source_metric_avg_scores2.loc[source],
            'Discrepancy': discrepancy.loc[source]
        })
        # 按照第一种采样方式（1000条文本）的评价指标平均分从高到低排序
        sorted_combined_df = combined_df.sort_values(by=f"Sampling{sample_size1}", ascending=False)
        output_file.write(pd.DataFrame(sorted_combined_df).to_string(header=True, index=True) + "\n")

    ## 6
    output_file.write("\n\n6. 所有源的所有领域的样本占比，从高到低排名：\n")
    source_category_proportion1 = (df1.groupby('source')['category'].value_counts(normalize=True) * 100).reset_index(name=f"Sampling{sample_size1}")
    source_category_proportion2 = (df2.groupby('source')['category'].value_counts(normalize=True) * 100).reset_index(name=f"Sampling{sample_size2}")
    combined_proportions = pd.merge(source_category_proportion1, source_category_proportion2, on=['source', 'category'], how='outer') # 合并两个采样占比 DataFrame
    combined_proportions['Discrepancy'] = combined_proportions[f"Sampling{sample_size1}"] - combined_proportions[f"Sampling{sample_size2}"] # 计算样本占比之差
    combined_proportions.sort_values(f"Sampling{sample_size1}", ascending=False, inplace=True) # 按照第一种采样方式（1000条文本）的样本占比从高到低排序
    output_file.write(combined_proportions[['source', 'category', f"Sampling{sample_size1}", f"Sampling{sample_size2}", 'Discrepancy']].to_string(index=False) + "\n")

    ## 7
    output_file.write("\n\n7. 同一个源的所有领域的样本量及占比，从高到低排名：\n")
    def calculate_ratio_stats(df, total_name):  # 计算样本量及占比
        category_counts = df.groupby(['source', 'category']).size().reset_index(name='Sample Number')
        total_samples = df.groupby(['source']).size().reset_index(name=total_name)
        stats = pd.merge(category_counts, total_samples, on='source')
        stats['Sample Ratio'] = stats['Sample Number'] / stats[total_name]
        return stats[['source', 'category', 'Sample Number', 'Sample Ratio']]
    
    stats1 = calculate_ratio_stats(df1, 'Total1')
    stats2 = calculate_ratio_stats(df2, 'Total2')
    combined_stats = pd.merge(stats1, stats2, on=['source', 'category'], suffixes=[f"_Sampling{sample_size1}", f"_Sampling{sample_size2}"])
    combined_stats['Discrepancy Number'] = combined_stats[f"Sample Number_Sampling{sample_size1}"] - combined_stats[f"Sample Number_Sampling{sample_size2}"]    # 计算差异
    combined_stats['Discrepancy Ratio'] = (combined_stats[f"Sample Ratio_Sampling{sample_size1}"] - combined_stats[f"Sample Ratio_Sampling{sample_size2}"]) * 100 # 计算差异
    combined_stats.sort_values(by=['source', f"Sample Number_Sampling{sample_size1}"], ascending=[True, False], inplace=True) # 按照源的名字排序，然后在每个源内按照第一种采样方式的样本量从高到低排序

    combined_stats[f"Sampling{sample_size1}"] = combined_stats.apply(lambda row: f"{row[f'Sample Number_Sampling{sample_size1}']}, {row[f'Sample Ratio_Sampling{sample_size1}']:.2%}", axis=1)
    combined_stats[f"Sampling{sample_size2}"] = combined_stats.apply(lambda row: f"{row[f'Sample Number_Sampling{sample_size2}']}, {row[f'Sample Ratio_Sampling{sample_size2}']:.2%}", axis=1)
    combined_stats['Discrepancy'] = combined_stats.apply(lambda row: f"{row['Discrepancy Number']}, {row['Discrepancy Ratio']:.2f}%", axis=1)
    output_file.write(combined_stats[['source', 'category', f"Sampling{sample_size1}", f"Sampling{sample_size2}", 'Discrepancy']].to_string(index=False) + "\n")

    output_file.write("\n\n##################### 从domain的角度 #####################\n")
    ## 8
    domain_metric_avg_scores1 = df1.groupby('category')[METRIC_LIST].mean()
    domain_metric_avg_scores2 = df2.groupby('category')[METRIC_LIST].mean()
    discrepancy = domain_metric_avg_scores1 - domain_metric_avg_scores2
    for metric in METRIC_LIST:
        output_file.write(f"\n\n8. 每一个领域的评价指标“{metric}”的平均分，从高到低排名：\n")
        combined_df = pd.concat([
            domain_metric_avg_scores1[metric].rename(f"Sampling{sample_size1}"),
            domain_metric_avg_scores2[metric].rename(f"Sampling{sample_size2}"),
            discrepancy[metric].rename("Discrepancy")], axis=1)
        sorted_combined_df = combined_df.sort_values(by=f"Sampling{sample_size1}", ascending=False)
        output_file.write(sorted_combined_df.to_string() + "\n")

    ## 9
    categories = df['category'].unique()
    for category in categories:
        output_file.write(f'\n\n9. 领域“{category}”的每一项评价指标的平均分，从高到低排名：\n')
        combined_df = pd.DataFrame({
            f"Sampling{sample_size1}": domain_metric_avg_scores1.loc[category],
            f"Sampling{sample_size2}": domain_metric_avg_scores2.loc[category],
            'Discrepancy': discrepancy.loc[category]
        })
        # 按照第一种采样方式（1000条文本）的评价指标平均分从高到低排序
        sorted_combined_df = combined_df.sort_values(by=f"Sampling{sample_size1}", ascending=False)
        output_file.write(pd.DataFrame(sorted_combined_df).to_string(header=True, index=True) + "\n")

    ## 10
    output_file.write("\n\n10. 所有领域的样本量及其占比，从高到低排名：\n")
    def calculate_category_stats(df, sample_size):   # 计算每个领域在两种采样下的样本量及其占比
        category_stats = df['category'].value_counts().rename_axis('Category').reset_index(name='Sample Number')
        category_stats['Sampling'] = (category_stats['Sample Number'] / (sample_size * len(df['source'].unique()))).round(4)
        return category_stats.set_index('Category')

    stats1 = calculate_category_stats(df1, sample_size1)
    stats2 = calculate_category_stats(df2, sample_size2)
   
    discrepancy_stats = stats1.join(stats2, lsuffix=f"_Sampling{sample_size1}", rsuffix=f"_Sampling{sample_size2}")      # 计算样本量之差及占比之差
    discrepancy_stats['Discrepancy Number'] = discrepancy_stats[f'Sample Number_Sampling{sample_size1}'] - discrepancy_stats[f'Sample Number_Sampling{sample_size2}']
    discrepancy_stats['Discrepancy Ratio'] = (discrepancy_stats[f'Sampling_Sampling{sample_size1}'] - discrepancy_stats[f'Sampling_Sampling{sample_size2}']) * 100
    
    discrepancy_stats.sort_values(by=f'Sample Number_Sampling{sample_size1}', ascending=False, inplace=True)  # 按照第一种采样方式的样本量从高到低排序
    discrepancy_stats[f"Sampling{sample_size1}"] = discrepancy_stats.apply(lambda row: f"{int(row[f'Sample Number_Sampling{sample_size1}'])}, {row[f'Sampling_Sampling{sample_size1}']:.2%}", axis=1)
    discrepancy_stats[f"Sampling{sample_size2}"] = discrepancy_stats.apply(lambda row: f"{int(row[f'Sample Number_Sampling{sample_size2}'])}, {row[f'Sampling_Sampling{sample_size2}']:.2%}", axis=1)
    discrepancy_stats['Discrepancy'] = discrepancy_stats.apply(lambda row: f"{int(row['Discrepancy Number'])}, {row['Discrepancy Ratio']:.2f}%", axis=1)
    output_file.write(discrepancy_stats[[f"Sampling{sample_size1}", f"Sampling{sample_size2}", 'Discrepancy']].to_string() + "\n")


def main(args): 
    fix_seed(args.seed)
    set_request_template_and_pattern(args.lang)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.analyse_type == "data":
        analyse_sft_data(args.input_path, args.sample_size1, args.seed, args.output_path)
    elif args.analyse_type == "discrepancy":
        assert args.sample_size2, "--sample_size2 is required when --analyse_type is 'discrepancy'"
        analyse_discrepancy(args.input_path, args.sample_size1, args.sample_size2, args.seed, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='input jsonlines file.')
    parser.add_argument('--output_path', type=str, required=True, help='output text file.')
    parser.add_argument('--lang', default='en', help='language name', choices=['en', 'zh'])
    parser.add_argument('--sample_size1', default=1000, type=int, help='number of texts sampled from a single source')
    parser.add_argument('--sample_size2', type=int, help='number of another texts sampled from a single source')
    parser.add_argument('--seed', default=1024, type=int, help='seed size')
    parser.add_argument('--analyse_type', default=None, type=str, help='analyse type', choices=['data', 'discrepancy'])
    args = parser.parse_args()
    main(args)

#!/bin/bash
## --analyse_type data 用于生成单个数据集的统计数据，--analyse_type discrepancy 用来生成一个数据集在不同sample_size scale的差值统计数据

## ZH SFT Data, 要用未上采样的数据集
# python /mnt/nas/pengru.pr/DataMan/analysis/statistic_sft_data.py \
# --input_path /mnt/nas/pengru.pr/data/DataMan/Qwen-sft/Qwen2-zh/finetune_train.jsonl \
# --output_path /mnt/nas/pengru.pr/checkpoints/DataMan/analysis/zh/statistic_sft_data_zh.log \
# --sample_size1 100000 \
# --seed 1024 \
# --lang zh \
# --analyse_type data


## EN SFT Data, 要用未上采样的数据集
# python /mnt/nas/pengru.pr/DataMan/analysis/statistic_sft_data.py \
# --input_path /mnt/nas/pengru.pr/data/DataMan/Qwen-sft/Qwen2-en-suppl/final_finetune_train.jsonl \
# --output_path /mnt/nas/pengru.pr/checkpoints/DataMan/analysis/en/statistic_sft_data_en.log \
# --sample_size1 100000 \
# --seed 1024 \
# --lang en \
# --analyse_type data