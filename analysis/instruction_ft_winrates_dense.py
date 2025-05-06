import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['Accuracy', 'Professionalism', 'Language Consistency', 'Originality', 'Style Consistency', 
'Topic Focus', 'Grammatical Diversity', 'Structural Standardization', 'Knowledge Novelty', 'Creativity', 
'Semantic Density', 'Sensitivity', 'Coherence', 'Overall Score&5']
values_1 = [67.1, 68.9, 69.2, 69.5, 69.7, 69.8, 70.0, 70.0, 70.5, 71.4, 71.5, 72.0, 72.9, 78.5] # win rate
values_2 = [32.9, 31.1, 30.8, 30.5, 30.3, 30.2, 30.0, 30.0, 29.5, 28.6, 28.5, 28.0, 27.1, 21.5] # loss rate

# 设置条形图的位置
y = np.arange(len(categories))

# 创建条形图
fig, ax = plt.subplots(figsize=(8, 6))

# 设置条形宽度
bar_width = 0.6

# 绘制第一个条形，颜色加深
ax.barh(y, values_1, bar_width, color='#0033CC', edgecolor='black', alpha=0.9, label='Value 1')  # 蓝色
# 绘制第二个条形，颜色加深
ax.barh(y, values_2, bar_width, color='#CC0033', edgecolor='black', alpha=0.9, left=values_1, label='Value 2')  # 红色

# 添加数字标注，位置调整为条形内
for i in range(len(categories)):
    ax.text(values_1[i] / 2, i, f'{values_1[i]}%', va='center', ha='center', color='white', fontsize=12)
    ax.text(values_1[i] + values_2[i] / 2, i, f'{values_2[i]}%', va='center', ha='center', color='white', fontsize=12)

# 添加误差条
error = [1.5] * len(values_1)  # 示例：为每个值设置相同的误差
ax.errorbar(values_1, y, xerr=error, fmt='none', color='black', capsize=5)

# 设置y轴和x轴标签
ax.set_yticks(y)
ax.set_yticklabels(categories, fontsize=12)

# 在每个条形的末尾绘制“uniform”标签
for i in range(len(categories)):
    ax.text(values_1[i] + values_2[i] +1, i, 'Educational Value', va='center', ha='left', fontsize=12)

# 显示网格，设置为虚线
ax.grid(axis='x', linestyle='--')  # 添加 linestyle 参数设置为虚线

# 移除上下的x轴,左右的y轴
ax.spines['top'].set_visible(False)  # 移除上面的x轴
ax.spines['bottom'].set_visible(False)  # 移除下面的x轴
ax.spines['left'].set_visible(False)  # 移除左边的x轴
ax.spines['right'].set_visible(False)  # 移除右边的x轴

# 移除x轴刻度值
ax.set_xticklabels([])  # 移除x轴刻度

# 调整布局并显示图形
plt.tight_layout()
plt.savefig('xxxxxxx/instruction_ft_winrates.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

