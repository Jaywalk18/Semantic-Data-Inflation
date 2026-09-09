import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import os

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 14  # 增加默认字体大小

# Read Excel file
file_path = 'CIFAR-10 (2).xlsx'
data = pd.read_excel(file_path)

# Smoothing function
def smooth(data, window_size=14, sigma=1000):
    return gaussian_filter1d(data, sigma=sigma, truncate=window_size//2)

# Initialize square figure
fig = plt.figure(figsize=(6, 7))  # 略微增大图表尺寸

# Define colors and labels
colors = [
    (230/255, 183/255, 69/255),  # Yellow - Object-level
    (126/255, 153/255, 244/255),  # Blue - Pixel-level
    (122/255, 182/255, 86/255),   # Green - Hybrid
    (157/255, 158/255, 163/255),  # Gray - Original
    (204/255, 124/255, 113/255)   # Red - Baseline
]

# Use consistent terminology with table
labels = ['Object-level', 'Pixel-level', 'Hybrid', 'Original', 'Baseline']
old_labels = ['YOLO', 'SAM', 'YOLO-SAM', 'ORIGIN', 'BASELINE']

# Draw main plot
ax = plt.gca()
lines = []
for i, (label, old_label) in enumerate(zip(labels, old_labels)):
    smoothed = smooth(data[old_label], sigma=15)
    line, = ax.plot(data['global_step'], smoothed, label=label, color=colors[i], linewidth=2.5)  # 增加线宽
    lines.append(line)
    # Draw original data (semi-transparent)
    ax.plot(data['global_step'], data[old_label], color=colors[i], alpha=0.1, linewidth=0.5)

# Set axis range
ax.set_ylim(59, 96)
ax.set_xlim(0, data['global_step'].max() * 1.02)

# Add title without "Figure." 前缀
# plt.title('Training Dynamics with Semantic Guidance', fontsize=14)

# Add legend
ax.legend(loc='lower right', shadow=False, fontsize=14, frameon=True, 
          ncol=2, bbox_to_anchor=(1, 0.01))

# Set axis labels
ax.set_xlabel('Training Steps', fontsize=14)
ax.set_ylabel('Linear Evaluation Accuracy (%)', fontsize=14)

# Custom formatter for x-axis
def custom_formatter(x, pos):
    if x == 0:
        return '0'
    return f'{int(x/1000)}k'

# Apply custom format
ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

# Set consistent font size for tick labels
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(14)

# Add grid
ax.grid(True, linestyle='--', alpha=0.4, which='both')

# Create inset for early stage
axins1 = inset_axes(ax, width="35%", height="30%", loc='lower left', 
                   bbox_to_anchor=(0.08, 0.1, 0.5, 0.7), bbox_transform=ax.transAxes)

# Draw zoomed part 1
x_min1, x_max1 = 8000, 12000
y_min1, y_max1 = 60, 85
for i, (label, old_label) in enumerate(zip(labels, old_labels)):
    smoothed = smooth(data[old_label], sigma=15)
    mask = (data['global_step'] >= x_min1) & (data['global_step'] <= x_max1)
    axins1.plot(data['global_step'][mask], smoothed[mask], color=colors[i], linewidth=2)

axins1.set_xlim(x_min1, x_max1)
axins1.set_ylim(y_min1, y_max1)
axins1.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
axins1.grid(True, linestyle='--', alpha=0.4)

# 使用白色底色，确保标题在顶部并且不会被连线干扰
title1 = 'Early Training'
axins1.set_title('')  # 清除默认标题

# 添加白色底色矩形和标题
rect1 = patches.Rectangle((-0.2, 1.075), 1.4, 0.2, transform=axins1.transAxes, 
                         color='white', alpha=0.9, zorder=5, clip_on=False)
axins1.add_patch(rect1)
axins1.text(0.5, 1.15, title1, fontsize=14, transform=axins1.transAxes,
            ha='center', va='center', zorder=6, clip_on=False)

# Set consistent font size for inset 1
for label in axins1.get_xticklabels() + axins1.get_yticklabels():
    label.set_fontsize(14)

# Create inset for mid stage
axins2 = inset_axes(ax, width="35%", height="30%", loc='center left', 
                   bbox_to_anchor=(0.25, 0.25, 0.6, 0.6), bbox_transform=ax.transAxes)

# Draw zoomed part 2
x_min2, x_max2 = 25000, 35000
y_min2, y_max2 = 77, 88
for i, (label, old_label) in enumerate(zip(labels, old_labels)):
    smoothed = smooth(data[old_label], sigma=15)
    mask = (data['global_step'] >= x_min2) & (data['global_step'] <= x_max2)
    axins2.plot(data['global_step'][mask], smoothed[mask], color=colors[i], linewidth=2)

axins2.set_xlim(x_min2, x_max2)
axins2.set_ylim(y_min2, y_max2)
axins2.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
axins2.grid(True, linestyle='--', alpha=0.4)

# 添加白色底色矩形和标题
title2 = 'Mid Training'
axins2.set_title('')  # 清除默认标题
rect2 = patches.Rectangle((-0.2, 1.075), 1.4, 0.2, transform=axins2.transAxes, 
                         color='white', alpha=0.7, zorder=5, clip_on=False)
axins2.add_patch(rect2)
axins2.text(0.5, 1.15, title2, fontsize=14, transform=axins2.transAxes,
            ha='center', va='center', zorder=6, clip_on=False)

# Set consistent font size for inset 2
for label in axins2.get_xticklabels() + axins2.get_yticklabels():
    label.set_fontsize(14)

# 添加最后阶段的放大图
axins3 = inset_axes(ax, width="35%", height="30%", loc='upper left', 
                   bbox_to_anchor=(0.65, -0.22, 0.6, 0.9), bbox_transform=ax.transAxes)

# Draw zoomed part 3 (final stage)
x_min3, x_max3 = 450000, 500000  # 最后阶段的步骤
y_min3, y_max3 = 92, 95  # 调整y轴范围以更好地显示差异
for i, (label, old_label) in enumerate(zip(labels, old_labels)):
    smoothed = smooth(data[old_label], sigma=15)
    mask = (data['global_step'] >= x_min3) & (data['global_step'] <= x_max3)
    axins3.plot(data['global_step'][mask], smoothed[mask], color=colors[i], linewidth=2)

axins3.set_xlim(x_min3, x_max3)
axins3.set_ylim(y_min3, y_max3)
axins3.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
axins3.grid(True, linestyle='--', alpha=0.4)

# 添加白色底色矩形和标题
title3 = 'Final Training'
axins3.set_title('')  # 清除默认标题
rect3 = patches.Rectangle((-0.2, 1.075), 1.4, 0.2, transform=axins3.transAxes, 
                         color='white', alpha=1.0, zorder=5, clip_on=False)
axins3.add_patch(rect3)
axins3.text(0.5, 1.15, title3, fontsize=14, transform=axins3.transAxes,
            ha='center', va='center', zorder=6, clip_on=False)

# Set consistent font size for inset 3
for label in axins3.get_xticklabels() + axins3.get_yticklabels():
    label.set_fontsize(14)

# Make main frame border thicker
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Add connection lines using mark_inset
mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5", lw=1, ls="--")
mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5", lw=1, ls="--")
mark_inset(ax, axins3, loc1=2, loc2=4, fc="none", ec="0.5", lw=1, ls="--")

# Apply tight layout to make sure everything fits
plt.tight_layout()

# 保存为PNG格式
plt.savefig('training_curves_figure.png', dpi=300)

# 确保已经绘制了所有内容
fig.canvas.draw()

# 尝试使用不同的保存模式来保存PDF
try:
    # 方法1: 使用最基本的保存方式
    plt.savefig('training_curves_figure_1.pdf', dpi=300)
except Exception as e:
    print(f"Method 1 failed: {e}")

try:
    # 方法2: 不使用tight_bbox
    plt.savefig('training_curves_figure_2.pdf', bbox_inches=None, dpi=300)
except Exception as e:
    print(f"Method 2 failed: {e}")

try:
    # 方法3: 使用不同的后端
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages('training_curves_figure_3.pdf') as pdf:
        pdf.savefig(fig)
except Exception as e:
    print(f"Method 3 failed: {e}")

# 显示图形
plt.show()

print("图形已保存为PNG格式和几种PDF格式，请检查哪一种PDF文件效果最好。")
