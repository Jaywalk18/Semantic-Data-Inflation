import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "mathtext.fontset": "stix",
})

methods = ['Std Aug', 'Raw Dup', 'SDI (Ours)']
accuracies = [69.91, 69.91, 72.35]
gain = accuracies[2] - accuracies[0]  # vs Standard, matching the manuscript headline

# ICLR 单栏宽度推荐：5.5 × 2.3 inch
fig, ax = plt.subplots(figsize=(4.5, 2))

# 统一配色：浅蓝基线 + 深蓝SDI
colors = ['#b6c7e0', '#b6c7e0', '#3173a4']
bars = ax.barh(methods, accuracies, color=colors, height=0.45)

# 设置 x 轴范围
ax.set_xlim(68, 73.2)

# 去掉多余边框
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# y 轴刻度靠近 bar
ax.tick_params(axis='y', pad=3, length=0)

# 数值与虚线
for i, (bar, acc, name) in enumerate(zip(bars, accuracies, methods)):
    y = bar.get_y() + bar.get_height()/2
    line_color = '#3173a4' if name == 'SDI (Ours)' else 'gray'

    ax.plot([acc, acc+0.25], [y, y], linestyle=':', color=line_color, linewidth=1)
    ax.text(acc+0.3, y, f'{acc:.2f}%', ha='left', va='center', fontsize=11)

# SDI 的 gain 注释（上移避免碰撞）
sdi_y = bars[2].get_y() + bars[2].get_height()/2
ax.text(71.3, sdi_y - 0.6,
        f'+{gain:.2f}% vs Standard.',
        ha='left', va='center', fontsize=11, color='#3173a4')

# x 轴
ax.set_xlabel('Top-1 Linear Probe Accuracy (%)', labelpad=5)
ax.spines['left'].set_visible(False)
ax.spines['left'].set_visible(True)

plt.savefig("imagenet_iclr.pdf", format="pdf", bbox_inches="tight")

plt.tight_layout()
plt.show()
