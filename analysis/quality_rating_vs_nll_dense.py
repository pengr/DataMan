import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# 定义输入路径
input_path = "xxxxxxxxxxxxxxx/Llama-2-7b-hf-ppl"
dataset = load_from_disk(input_path)
# 移除不需要的列
columns_to_remove = ['input_ids', 'text', 'source_domain', 'document_index', 
                     'document_position', 'length', 'application_domain']
dataset = dataset.remove_columns(columns_to_remove)
# 在数据集上划分
# dataset = dataset.shard(300, 1, contiguous=True)
# 提取质量标准分数
quality_columns = [
    'accuracy', 'coherence', 'language_consistency', 'semantic_density',
    'knowledge_novelty', 'topic_focus', 'creativity', 'professionalism',
    'style_consistency', 'grammatical_diversity', 'structural_standardization',
    'originality', 'sensitivity', 'overall_score'
]
# 将字符串转换为浮点数组
data = [np.array(list(map(float, dataset[col])), dtype=float) for col in quality_columns]
# 转换 avg_loglikelihood 列
avg_loglikelihood = np.array(list(map(float, dataset['avg_loglikelihood'])), dtype=float)
# 计算相关性
spearman_values = []
pearson_values = []

# 使用 tqdm 添加进度条
for col in tqdm(quality_columns, desc="Calculating Correlation", unit="column"):
    column_data = data[quality_columns.index(col)]
    
    correlation, _ = pearsonr(column_data, avg_loglikelihood)
    pearson_values.append(correlation)
    
    correlation, _ = spearmanr(column_data, avg_loglikelihood)
    spearman_values.append(correlation)

# 创建子图，2行7列布局
fig, axes = plt.subplots(2, 7, figsize=(18, 6))  # 可以根据需要调整尺寸
axes = axes.flatten()  # 展平子图数组
# 使用 cubehelix 调色板
cmap = sns.cubehelix_palette(light=1, as_cmap=True)

# 绘制每个子图
quality_name_columns = [
    'Accuracy', 'Coherence', 'Language Consistency', 'Semantic Density',
    'Knowledge_novelty', 'Topic Focus', 'Creativity', 'Professionalism',
    'Style Consistency', 'Grammatical Diversity', 'Structural Standardization',
    'Originality', 'Sensitivity', 'Overall Score'
]

# 使用 tqdm 添加进度条
for i, (ax, d) in tqdm(enumerate(zip(axes, data)), desc="Plotting", total=len(data), unit="subplot"):
    sns.kdeplot(x=d, y=avg_loglikelihood, ax=ax, fill=True, cmap=cmap, thresh=0, levels=10, bw_adjust=0.5)
    
    # 设置子图标题
    ax.set_title(f'$r$: {spearman_values[i]:.2f}, $\\rho$: {pearson_values[i]:.2f}', fontsize=10)
    ax.set_xlabel(quality_name_columns[i], fontsize=10)  # 设置 x 轴标签
    ax.set_ylabel('Negative Log-Likelihood', fontsize=10) if i % 7 == 0 else ax.set_yticks([])  # 仅在第一列显示 y 轴标签

# 添加颜色条，使用数据的最小和最大值
mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])  # 别忘了要先设置为一个空数组
# 直接使用数据创建颜色条
cbar = fig.colorbar(
    mappable,
    ax=axes,
    orientation='vertical',
    fraction=0.02,
    pad=0.04
)

# 调整布局
plt.subplots_adjust(hspace=0.4, wspace=0.1, right=0.87)  # right=0.87为颜色条留出空间
plt.savefig('xxxxxxxxxxxx/quality_rating_vs_nll.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
