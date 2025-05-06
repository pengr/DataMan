import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
from scipy.stats import pearsonr, spearmanr

# 创建数据
Val_PPL = np.array([10.7, 13.34, 13.60, 15.98, 11.32, 13.01, 10.6, 14.38, 10.68, 13.54, 10.67, 14.97, 10.70, 10.63, 10.09, 10.82, 10.72, 11.08, 10.87, 11.01, 10.35, 10.68, 11.27, 11.1, 10.11, 12.11, 10.74, 10.41, 23.83, 12.84, 11.75, 10.21, 10.52])
ARC_E_ICL = np.array([57.5, 52.8, 49.5, 49.2, 53.5, 52.7, 56.4, 65.6, 59.3, 66.6, 60.7, 60.4, 59.6, 59.2, 60.6, 62.8, 63.6, 60.8, 64.6, 63.5, 63.0, 64.0, 64.4, 66.2, 63.2, 62.1, 63.0, 61.6, 42.0, 53.7, 54.3,  60.7, 66.1])
ARC_C_ICL = np.array([27.6, 26.3, 25.3, 25.1, 25.6, 27.3, 28.4, 33.1, 29.8, 34.6, 30.4, 30.9, 29.8, 30.2, 29.3, 29.4, 32.5, 30.6, 33.4, 32.8, 31.0, 31.7, 32.2, 31.9, 32.4, 31.9, 32.0, 30.8, 23.0, 26.0, 26.2,  31.3, 34.0])
SciQ_ICL = np.array([87.7, 85.9, 83.6, 83.7, 84.6, 79.7, 85.8, 87.9, 88.1, 89.6, 88.8, 86.8, 89.0, 88.0, 90.3, 90.3, 90.3, 87.8, 89.3, 90.5, 89.7, 90.7, 91.1, 91.4, 91.1, 89.8,  90.3, 91.2, 69.4, 83.8, 87.1,  90.6, 90.7])
LogiQA_ICL = np.array([24.1, 25.2, 23.5, 22.0, 26.1, 26.4, 24.9, 24.1, 25.0, 24.6, 26.6, 25.0, 23.8, 24.3, 24.4, 24.7, 26.7, 24.3, 26.3, 24.3, 25.3, 25.3, 24.0, 25.2, 25.5, 25.3, 28.7, 27.3, 25.7, 26.4, 24.1, 24.1, 26.1])
BoolQ_ICL = np.array([57.5, 60.3, 57.9, 61.4, 58.0, 60.5, 59.3, 60.9, 61.4, 58.3, 60.1, 60.9, 61.4, 58.7, 60.1, 61.7, 61.6, 61.4, 62.0, 62.1, 61.4, 57.7, 61.2, 57.2, 61.3, 59.3, 61.6, 61.9, 55.2, 61.8, 62.0, 60.8, 59.2])
HellaSwag_ICL = np.array([44.0, 35.8, 44.8, 34.6, 41.6, 41.1, 44.9, 39.4, 43.9, 45.5, 45.4, 36.1, 43.2, 44.5, 47.7, 49.8, 50.8, 51.9, 50.6, 47.2, 50.2, 49.0, 45.0, 48.4, 50.4, 45.9, 50.3, 50.0, 31.0, 38.2, 42.0, 51.3, 51.5])
PIQA_ICL = np.array([68.6, 61.4, 69.4, 65.0, 65.6, 66.1, 68.6, 62.5, 68.3, 66.4, 69.1, 57.8, 67.4, 68.7, 69.0, 69.3, 70.6, 71.2, 69.8, 67.9, 70.1, 70.5, 66.1, 71.2, 70.6, 70.6, 70.7, 69.0, 61.3, 63.9, 68.3, 71.0, 70.7])
WinoGrande_ICL = np.array([52.5, 52.2, 55.6, 49.1, 53.4, 52.3, 55.8, 53.1, 54.6, 52.9, 54.2, 52.2, 56.0, 53.5, 54.4, 55.6, 55.1, 58.6, 56.1, 55.6, 57.6, 56.2, 53.3, 54.7, 56.6, 54.5, 57.8, 56.3, 50.3, 50.5, 52.0, 57.9, 58.3])
NQ_ICL = np.array([4.1, 4.7, 3.1, 2.7, 2.9, 2.5,  4.5, 5.7, 4.4, 3.8, 4.3, 2.4, 4.6, 5.3, 5.8, 7.0, 7.0, 4.7, 7.7, 6.2, 7.6, 8.0, 6.6, 7.5, 6.8, 7.1, 7.2, 7.1, 0.8, 4.3, 4.7, 7.7, 7.8])
MMLU_ICL = np.array([25.7, 24.7, 25.2, 24.7, 24.0, 24.4, 23.8, 25.3, 26.9, 25.0, 27.1, 26.3, 25.4, 25.1, 26.1, 26.2, 25.2, 25.6, 25.2, 24.8, 25.8, 24.7, 25.2, 25.9, 25.3, 27.0, 25.1, 24.0, 25.4, 25.0, 25.7, 24.2, 26.9])
Model= np.array(['Uniform', 'DSIR Wiki', 'DSIR Book', 'Perplexity lowest', 'Perplexity highest', 'Writing Style top-k', 'Writing Style τ=2', 
'Facts & Trivia top-k', 'Facts & Trivia τ=2', 'Edu. Value top-k', 'Edu. Value τ=2', 'Required Exp. top-k', 'Required Exp. τ=2',
'Criteria mix τ=2', 'Uniform +50% data', 'Accuracy top-k', 'Coherence top-k', 'Creativity top-k', 'Grammatical Diversity top-k',
'Knowledge Novelty top-k', 'Language Consistency top-k', 'Originality top-k', 'Professionalism top-k', 'Semantic Density top-k',
'Sensitivity top-k', 'Structural Standardization top-k', 'Style Consistency top-k', 'Topic Focus top-k', 'Overall Score l=1', 
'Overall Score l=2', 'Overall Score l=3', 'Overall Score l=4', 'Overall Score l=5'])

data = {
    'Perplexity': Val_PPL, 
    'ARC_E_ICL': ARC_E_ICL,
    'ARC_C_ICL': ARC_C_ICL,
    'SciQ_ICL': SciQ_ICL,
    'LogiQA_ICL': LogiQA_ICL,
    'BoolQ_ICL': BoolQ_ICL,
    'HellaSwag_ICL': HellaSwag_ICL,
    'PIQA_ICL': PIQA_ICL,
    'WinoGrande_ICL': WinoGrande_ICL,
    'NQ_ICL': NQ_ICL,
    'MMLU_ICL': MMLU_ICL,
    'Model': Model
}
df = pd.DataFrame(data)

# 创建子图
fig, axs = plt.subplots(2, 5, figsize=(12, 6))  # 矩形形状
axs = axs.flatten()
# 生成每个散点图
titles = ['ARC-e', 'ARC-c', 'SciQ', 'LogiQA', 'BoolQ', 'HellaSwag', 'PIQA', 'WinoGrande', 'NQ', 'MMLU']
ICL_List = ["ARC_E_ICL", "ARC_C_ICL", "SciQ_ICL", "LogiQA_ICL", "BoolQ_ICL", "HellaSwag_ICL", "PIQA_ICL", "WinoGrande_ICL", "NQ_ICL", "MMLU_ICL"]
# 记录每个模型名
unique_models = df['Model'].unique()
num_models = len(unique_models)
# 获取调色板并确保有足够颜色
palette = sns.color_palette("husl", num_models)
for i, title in enumerate(titles):
    ICL = ICL_List[i]
    sns.scatterplot(data=df, x='Perplexity', y=ICL, hue='Model', ax=axs[i], alpha=1.0, legend=False, s=30)
    axs[i].set_title(title, fontweight='bold')  # 设置为粗体 
    # 只保留左边和下边的边框
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    if i ==0:
        axs[i].set(title=title, xlabel='', ylabel='ICL Performance')  # 设置标题并隐藏轴标签
        axs[i].tick_params(axis='x', labelbottom=False)  # 隐藏横坐标刻度的数值，仅保留刻度线
    elif 1<= i < 5:
        axs[i].set(title=title, xlabel='', ylabel='')  # 设置标题并隐藏轴标签
        axs[i].tick_params(axis='x', labelbottom=False)  # 隐藏横坐标刻度的数值，仅保留刻度线
    elif i==5:
        axs[i].set(title=title, xlabel='Perplexity', ylabel='ICL Performance')  # 设置标题并隐藏轴标签
    elif 5< i < 10:
        axs[i].set(title=title, xlabel='Perplexity', ylabel='')  # 设置标题并隐藏轴标签
    # 计算并打印 Pearson 和 Spearman 相关系数
    pearson_corr, _ = pearsonr(df['Perplexity'], df[ICL])
    spearman_corr, _ = spearmanr(df['Perplexity'], df[ICL])
    print(f"{title}: Pearson correlation = {pearson_corr:.2f}, Spearman correlation = {spearman_corr:.2f}")
    
    # 添加相关系数信息到右上角
    textstr = f'$r$: {pearson_corr:.2f}\n$\\rho$: {spearman_corr:.2f}'
    props = dict(boxstyle='round', facecolor='none', edgecolor='none')  # 设置文本框完全透明
    axs[i].text(0.75, 0.95, textstr, transform=axs[i].transAxes, fontsize=11,
                verticalalignment='top', bbox=props)  # 移除 alpha 参数

# 创建统一图例，横排放置
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=6) for i in range(num_models)]
fig.legend(handles, unique_models, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize='smaller', frameon=False, facecolor='none')
# 保存图像
plt.tight_layout()
plt.savefig('xxxxxx/icl_vs_ppl.pdf', bbox_inches='tight', pad_inches=0)
plt.show()