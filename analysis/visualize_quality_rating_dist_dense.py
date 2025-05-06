import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import re
from matplotlib.ticker import FormatStrFormatter

# Define evaluation criteria and sources
criteria_group_1 = ["accuracy", "coherence", "language_consistency", "semantic_density", 
                     "knowledge_novelty", "topic_focus", "creativity"]
criteria_group_2 = ["professionalism", "style_consistency", "grammatical_diversity", 
                     "structural_standardization", "originality", "sensitivity", "overall_score"] 
sources = ['CommonCrawl', 'C4', 'Github', 'Wikipedia', 'ArXiv', 'Book', 'StackExchange']

# Create an empty 3D array to store results
data_shape = (len(criteria_group_1) + len(criteria_group_2), len(sources), 5)
data = np.zeros(data_shape, dtype=float)

# Use a dictionary for source and criterion indexing
source_idx = {source: idx for idx, source in enumerate(sources)}
criterion_idx = {criterion: idx for idx, criterion in enumerate(criteria_group_1 + criteria_group_2)}

# Extract data from the log file
log_file = "xxxxxxxxxxxxxx/quality_rating_dist.log"
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        # 使用频数
        # match = re.match(r'(\w[\w\s]*?) - (\w[\w\s]*?) - (\d): (\d+)', line.strip())
        # if match:
        #     criterion, source, rating, count = match.groups()
        #     rating = int(rating) - 1  # 将 1-5 评分转换为 0-4 索引
        #     count = int(count.strip().replace(',', ''))  # 转换为整数并移除逗号

        #     criterion_index = criterion_idx.get(criterion)  # 获取索引
        #     source_index = source_idx.get(source)
        #     if criterion_index is not None and source_index is not None:
        #         data[criterion_index, source_index, rating] = count

        # 使用百分比
        match = re.match(r'(\w[\w\s]*?) - ([A-Za-z0-9_]+) - (\d): \d+, (\d+\.\d+)%', line.strip())
        if match:
            criterion, source, rating, percentage = match.groups()
            rating = int(rating) - 1  # Convert rating to 0-4 index
            percentage = float(percentage.strip()) / 100  # Convert to decimal
            criterion_index = criterion_idx.get(criterion)  # Get index
            source_index = source_idx.get(source)
            if criterion_index is not None and source_index is not None:
                data[criterion_index, source_index, rating] = percentage

# 重新排序数据并展平
sorted_data = np.zeros(data_shape, dtype=float)
for criterion_idx in range(len(criteria_group_1 + criteria_group_2)):
    for source_idx in range(len(sources)):
        for rating in range(5):  # 将 1-5 评分转换为 0-4 索引
            sorted_data[criterion_idx, source_idx, rating] = data[criterion_idx, source_idx, rating]
sorted_data_flatten = sorted_data.flatten()

# 将数据转换为长格式，便于 Seaborn 使用
# 将criteria中每个值复制35次，['Accuracy'*35, 'Coherence'*35, ..., 'Sensitivity'*35, 'Overall_Score'*35]
# 将sources中每个值复制5次拼接在一起（'CC'*5, 'C4'*5, ..., "Book"*5)，再把这个整体复制14次[（'CC'*5, 'C4'*5, ..., "Book"*5)*14]
# 将sorted_data，7 * 14 * 5的3维数组展平成490个数的数组
df = pd.DataFrame({
    'Criterion': np.repeat(criteria_group_1 + criteria_group_2, len(sources) * 5),
    'Source_Domain': np.tile(np.repeat(sources, 5), len(criteria_group_1) + len(criteria_group_2)),
    'Rating': sorted_data_flatten,
})

# # Create subplots: 14 rows, 7 columns
fig, axes = plt.subplots(nrows=len(criteria_group_1) + len(criteria_group_2), ncols=len(sources), figsize=(16, 12), squeeze=False)

# Plot barplots
for i, criterion in enumerate(criteria_group_1 + criteria_group_2):
    for j, source_domain in enumerate(sources):
        ax = axes[i, j]
        data_to_plot = df[(df['Criterion'] == criterion) & (df['Source_Domain'] == source_domain)]

        # Create a barplot
        barplot = sns.barplot(
            x=[1,2,3,4,5],
            y=data_to_plot['Rating'].values, 
            ax=ax,
            color='#0033CC'
        )
        
        # 在柱子顶部显示数值
        for p in barplot.patches:
            ax.annotate(
                f'{p.get_height():.2f}',  # 显示柱子的数值，保留两位小数
                (p.get_x() + p.get_width() / 2, p.get_height()),  # 在柱子顶部对齐
                ha='center', va='bottom',  # 文本中心对齐
                fontsize=8, color='black'
            )

        # 仅设置 y 轴的刻度及刻度值在第一列
        # if j == 0:  # 第一列
        #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # 设置 y 轴刻度格式
        # else:

        ax.set_ylim(0, 1)  # 假设评分是百分比

        # 仅设置下方子图的 x 轴刻度
        if i == len(criteria_group_1) + len(criteria_group_2) - 1:  # 最后一行
            ax.set_xticks([0, 1, 2, 3, 4])  # 设置刻度
            ax.set_xticklabels(['1', '2', '3', '4', '5'])  # 设置刻度标签
            ax.set_xlabel(source_domain, rotation=0, ha='center', va='center', fontsize=12, labelpad=10)  # 添加 x 轴标签
        else:
            ax.set_xticks([])  # 隐藏其他行的 x 轴刻度

        ax.set_ylabel('')  # 隐藏 y 轴标签
        ax.set_yticks([])  # 隐藏 y 轴刻度

# Add criteria labels at the left for all rows
criteria_texts = ["Accuracy", "Coherence", "Language Cons.", "Semantic Dens.", "Knowledge Novel.", 
                "Topic Focus", "Creativity", "Professionalism", "Style Cons.", "Grammatical Dive.", 
                "Structural Stand.", "Originality", "Sensitivity", "Overall Score"]

for j, criterion in enumerate(criteria_texts):
    fig.text(0.00, 1 - (j + 0.5) / len(criteria_texts), criterion, ha='right', va='center', fontsize=12)

# Adjust layout
plt.tight_layout(pad=0.1)
plt.savefig('xxxxxxxxxxxxxxx/quality_rating_dist.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
