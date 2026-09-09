import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,  # 增大基础字体
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.2,
})

# 数据
methods = ['Standard\nAug', 'Raw\nDuplication', 'Generative\nInflation', 'SDI\n(Ours)']
cifar10 = [92.8, 93.0, 93.4, 94.5]
imagenette = [91.9, 92.0, 93.2, 95.7]

x = np.arange(len(methods))
width = 0.35

# 双栏大图尺寸
fig, ax = plt.subplots(figsize=(7.5, 3.4))

# 统一配色方案：深浅成对
# CIFAR-10: 浅蓝渐进到深蓝 (SDI突出)
colors_cifar = ['#b6c7e0', '#b6c7e0', '#b6c7e0', '#3173a4']
# ImageNette: 浅橙渐进到深橙 (SDI突出)
colors_imagenette = ['#eebb8c', '#eebb8c', '#eebb8c', '#e0822a']

# 绑制柱状图
bars1 = ax.bar(x - width/2, cifar10, width, label='CIFAR-10', 
               color=colors_cifar, edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x + width/2, imagenette, width, label='ImageNette', 
               color=colors_imagenette, edgecolor='white', linewidth=0.8)

# 标注数值 - 增大字体
for bar, val in zip(bars1, cifar10):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.15, f'{val:.1f}', 
            ha='center', va='bottom', fontsize=16, fontweight='bold', color='#3173a4')
for bar, val in zip(bars2, imagenette):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.15, f'{val:.1f}', 
            ha='center', va='bottom', fontsize=16, fontweight='bold', color='#e0822a')

# 虚线 baseline (Standard Aug)
ax.hlines(y=91.9, xmin=-0.5, xmax=3.7, colors='#777777', linestyles='--', linewidth=1.2, alpha=0.8)

# 增益箭头标注 (ImageNette: 91.9 -> 95.7) - 使用橙色系
ax.annotate('', xy=(3 + width/2, 95.7), xytext=(3 + width/2, 91.9),
            arrowprops=dict(arrowstyle='->', color='#e0822a', lw=2))
ax.text(3 + width/2 + 0.18, 93.8, '+3.8%', fontsize=17, color='#e0822a', fontweight='bold')

# 样式 - 增大字体
ax.set_ylabel('Linear Probe Accuracy (%)', fontsize=19)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=17)
ax.set_ylim(90.5, 97)
ax.legend(loc='upper left', frameon=True, fontsize=16, framealpha=0.95)

# 去掉上右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("performance_comparison.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.savefig("../kbs_submission/figures/performance_comparison.pdf", format="pdf", bbox_inches="tight", dpi=300)
# plt.show()
print("图片已保存")
