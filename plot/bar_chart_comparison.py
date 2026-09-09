import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.7, 3.3))

# 数据
cifar_categories = ['Standard Aug', 'Raw Duplication', 'Generative Inflation', 'Semantic Inflation']
cifar_values = [92.8, 93.0, 93.4, 94.5]

imagenet_categories = ['Standard Aug', 'Raw Duplication', 'Generative Inflation', 'Semantic Inflation']
imagenet_values = [91.9, 92.0, 93.2, 95.7]

# 颜色
colors = ['#a5becc', '#9c9c9c', '#f5b095', '#4b86b4']

# 绘制 CIFAR-10 性能图
bar_width = 0.6
x = np.arange(len(cifar_categories))
bars1 = ax1.bar(x, cifar_values, width=bar_width, color=colors)

# 设置 CIFAR-10 图表属性
ax1.set_title('CIFAR-10 Performance', pad=10)
ax1.set_ylim(92.0, 95.5)
ax1.set_ylabel('Linear Accuracy (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(cifar_categories, rotation=15, ha='right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 在柱状图上添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height}', ha='center', va='bottom')

# 绘制 ImageNette 性能图
bars2 = ax2.bar(x, imagenet_values, width=bar_width, color=colors)

# 设置 ImageNette 图表属性
ax2.set_title('ImageNette Performance', pad=10)
ax2.set_ylim(91.0, 96.0)
ax2.set_ylabel('Linear Accuracy (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(imagenet_categories, rotation=15, ha='right')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 在柱状图上添加数值标签
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height}', ha='center', va='bottom')

# 调整布局
plt.tight_layout()

# 保存为PDF
plt.savefig('performance_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)
