import matplotlib.pyplot as plt
import numpy as np

# Configure Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Method names
methods = ['Gen. Inf.', 'SDI-Y', 'SDI-S', 'SDI-YS']

# Updated data values
gpu_time = [750, 120, 160, 140]  # 从秒转换为约 12.5分钟, 2分钟, 2.7分钟, 2.3分钟
memory = [6.2, 1.3, 2.1, 1.7]  # 内存使用保持不变

# Modern color scheme
color1 = '#3182bd'  # Blue (GPU time)
color2 = '#31a354'  # Green (Memory)

# Create compact horizontal bar chart
fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

y_pos = np.arange(len(methods))
width = 0.35

# First bars - GPU time
bars1 = ax.barh(y_pos + width/2, gpu_time, width, color=color1, 
                label='GPU Time (s)', edgecolor='white', linewidth=0.5)

# Second axis for memory usage
ax2 = ax.twiny()
bars2 = ax2.barh(y_pos - width/2, memory, width, color=color2, 
                 label='Memory (GB)', edgecolor='white', linewidth=0.5)

# Set y-axis ticks and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.invert_yaxis()  # Invert y-axis to place Gen. Inf. at top

ax.set_xlabel('GPU Time (seconds)', fontweight='bold')
ax2.set_xlabel('Memory (GB)', fontweight='bold')

# Add grid lines on x-axis only
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)  # Place grid lines behind bars

# Add value labels
for i, v in enumerate(gpu_time):
    ax.text(v + 15, i + width/2, f"{v}s", va='center', fontsize=9)
for i, v in enumerate(memory):
    ax2.text(v + 0.2, i - width/2, f"{v}GB", va='center', fontsize=9)

# Set reasonable axis limits
ax.set_xlim(0, max(gpu_time) * 1.15)
ax2.set_xlim(0, max(memory) * 1.25)

# 在右侧添加速度和内存节省比例（以两行方式排列）
speedup = [f"{gpu_time[0]/t:.1f}×" for t in gpu_time]
memoryred = [f"{memory[0]/m:.1f}×" for m in memory]

# 为避免与图例重叠，调整注释位置并以两行排列
for i in range(1, len(methods)):
    # 第一行：速度提升
    ax.text(max(gpu_time)*0.45, i+0.16, f"↓ {speedup[i]} faster", 
            fontsize=10, color='darkblue', ha='left', va='center')
    # 第二行：内存节省
    ax.text(max(gpu_time)*0.45, i-0.16, f"↓ {memoryred[i]} less memory", 
            fontsize=10, color='darkgreen', ha='left', va='center')

# Add legend and keep in right bottom
handles = [bars1, bars2]
labels = ['GPU Time (s)', 'Memory (GB)']
ax.legend(handles=handles, labels=labels, loc='lower right', frameon=True, 
          facecolor='white', edgecolor='lightgray', framealpha=0.9)

# Fine-tune layout
plt.tight_layout()

# Add subtle border
for spine in ax.spines.values():
    spine.set_edgecolor('lightgray')
for spine in ax2.spines.values():
    spine.set_edgecolor('lightgray')

plt.savefig('compact_comparison.pdf', bbox_inches='tight', dpi=300)
plt.show()
