import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15  # 默认字体大小为14

# Create figure with three subplots
fig, ax = plt.subplots(1, 3, figsize=(13, 5))

# Common parameters
n_samples = 30
np.random.seed(42)

# Define colors
original_color = '#3366CC'  # blue
traditional_color = '#AAAAAA'  # gray
generative_color = '#FF9900'  # orange
semantic_guided_color = '#109618'  # green

# 创建空的句柄和标签列表，用于最后统一绘制图例
handles = []
labels = []

# Create base original data points (for all three plots)
original_x = np.random.normal(0, 0.3, n_samples)
original_y = np.random.normal(0, 0.3, n_samples)

# ---- 1. Traditional Augmentation ----
ax[0].set_title('Traditional Augmentation', fontsize=15)
# Original points - 不设置label以避免在子图中显示图例
original_scatter = ax[0].scatter(original_x, original_y, s=80, color=original_color, 
                              edgecolor='black')

# Add semantic region indicator
main_ellipse = Ellipse(xy=(0, 0), width=1.0, height=0.7, 
                      angle=-30, edgecolor='black', fc='None', lw=1, ls='--')
ax[0].add_patch(main_ellipse)

# Generate traditional augmented samples (random spread)
traditional_x = original_x + np.random.normal(0, 0.7, n_samples)
traditional_y = original_y + np.random.normal(0, 0.7, n_samples)
traditional_scatter = ax[0].scatter(traditional_x, traditional_y, s=50, color=traditional_color, 
                                 alpha=0.7)

# Add annotation - 优化注释的背景和位置
ax[0].annotate('Random transformations\nignoring semantic structure', 
              xy=(-1, 1), xytext=(0, 0.8),  # 将y坐标从0.2改为0.8，使文字向上移动
              ha='center', fontsize=14, 
              # 增加边框边距和背景透明度，使文字更清晰
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray", alpha=0.85))


# ---- 2. Generative Data Inflation ----
ax[1].set_title('Generative Data Inflation', fontsize=15)
# Original points - 不设置label
ax[1].scatter(original_x, original_y, s=80, color=original_color, edgecolor='black')

# Add semantic region indicator
main_ellipse = Ellipse(xy=(0, 0), width=1.0, height=0.7, 
                      angle=-30, edgecolor='black', fc='None', lw=1, ls='--')
ax[1].add_patch(main_ellipse)

# Generate new data clusters - forming new "modes" in the distribution
n_clusters = 4
cluster_centers_x = np.random.uniform(-1.5, 1.5, n_clusters)
cluster_centers_y = np.random.uniform(-1.5, 1.5, n_clusters)

generative_x = []
generative_y = []

# Create generative samples around each cluster center
for i in range(n_clusters):
    n_points = np.random.randint(8, 15)
    cx, cy = cluster_centers_x[i], cluster_centers_y[i]
    
    # Generate cluster of points
    gx = cx + np.random.normal(0, 0.2, n_points)
    gy = cy + np.random.normal(0, 0.2, n_points)
    
    generative_x.extend(gx)
    generative_y.extend(gy)
    
    # Add ellipse to show generative cluster
    width = np.random.uniform(0.5, 0.8)
    height = np.random.uniform(0.3, 0.6)
    angle = np.random.uniform(0, 180)
    ellipse = Ellipse(xy=(cx, cy), width=width, height=height, 
                     angle=angle, edgecolor=generative_color, fc=generative_color, alpha=0.1)
    ax[1].add_patch(ellipse)

# Plot generative points - 不设置label
generative_scatter = ax[1].scatter(generative_x, generative_y, s=50, color=generative_color, 
                                alpha=0.8, edgecolor='black', linewidth=0.5)

# Add some connecting lines to show relationship to original data
for i in range(n_clusters):
    cx, cy = cluster_centers_x[i], cluster_centers_y[i]
    # Find closest original point
    distances = [(x-cx)**2 + (y-cy)**2 for x, y in zip(original_x, original_y)]
    closest_idx = np.argmin(distances)
    # Draw connection
    ax[1].plot([original_x[closest_idx], cx], [original_y[closest_idx], cy], 
             color=generative_color, alpha=0.3, linestyle='--')

# 调整标注位置 - Generated Mode向左移动
# 识别最左上方的簇
left_top_cluster_idx = np.argmax([cy - cx for cx, cy in zip(cluster_centers_x, cluster_centers_y)])
left_top_cx, left_top_cy = cluster_centers_x[left_top_cluster_idx], cluster_centers_y[left_top_cluster_idx]

ax[1].annotate('Generated Mode', xy=(left_top_cx, left_top_cy), 
             xytext=(left_top_cx-0.7, left_top_cy+0.65),  # 左移标注
             fontsize=14, 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", 
                            color='black', alpha=0.7))

# 识别最下方的簇用于Style-transferred Region标注
bottom_cluster_idx = np.argmin([cy for cx, cy in zip(cluster_centers_x, cluster_centers_y)])
bottom_cx, bottom_cy = cluster_centers_x[bottom_cluster_idx], cluster_centers_y[bottom_cluster_idx]

# Style-transferred Region向下移
ax[1].annotate('Style-transferred Region', xy=(bottom_cx, bottom_cy), 
             xytext=(bottom_cx+0.4, bottom_cy+0.3),  # 向下移动标注
             fontsize=14, 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", 
                            color='black', alpha=0.7))

# ---- 3. Semantic-Guided Data Inflation ----
ax[2].set_title('Semantic-Guided Data Inflation (SDI)', fontsize=15)
# Original points - 不设置label
ax[2].scatter(original_x, original_y, s=80, color=original_color, edgecolor='black')

# Add semantic region
main_ellipse = Ellipse(xy=(0, 0), width=1.0, height=0.7, 
                      angle=-30, edgecolor='black', fc='None', lw=1, ls='--')
ax[2].add_patch(main_ellipse)

# 重新设计第三个子图中的语义引导区域，确保Object Detection区域内有样本点

# 创建两个不同的语义区域
semantic_samples_detection = 25  # Object Detection区域样本数
semantic_samples_segmentation = 25  # Segmentation区域样本数

semantic_x = []
semantic_y = []
semantic_group = []  # 添加组标识，用于区分不同的语义区域

# 对象检测引导区域 - 左下区域
np.random.seed(100)  # 设置新的种子以确保可重复性
# 中心点设在(-0.8, -0.4)
center_detection_x, center_detection_y = -0.8, -0.4

# 生成对象检测区域的点 - 使用正态分布确保点集中在区域内
for i in range(semantic_samples_detection):
    x = np.random.normal(center_detection_x, 0.25)
    y = np.random.normal(center_detection_y, 0.2)
    semantic_x.append(x)
    semantic_y.append(y)
    semantic_group.append(1)

# 分割引导区域 - 右下区域
np.random.seed(200)  # 设置新的种子
# 中心点设在(0.5, -0.5)
center_segmentation_x, center_segmentation_y = 0.5, -0.5

# 生成分割区域的点 - 使用正态分布
for i in range(semantic_samples_segmentation):
    x = np.random.normal(center_segmentation_x, 0.2)
    y = np.random.normal(center_segmentation_y, 0.15)
    semantic_x.append(x)
    semantic_y.append(y)
    semantic_group.append(2)

# 分别获取两个区域的点
group1_x = [semantic_x[i] for i in range(len(semantic_x)) if semantic_group[i] == 1]
group1_y = [semantic_y[i] for i in range(len(semantic_y)) if semantic_group[i] == 1]
group2_x = [semantic_x[i] for i in range(len(semantic_x)) if semantic_group[i] == 2]
group2_y = [semantic_y[i] for i in range(len(semantic_y)) if semantic_group[i] == 2]

# 分别绘制两个区域的点
semantic_scatter1 = ax[2].scatter(group1_x, group1_y, s=50, color=semantic_guided_color, 
                               alpha=0.7, edgecolor='black', linewidth=0.5)
semantic_scatter2 = ax[2].scatter(group2_x, group2_y, s=50, color=semantic_guided_color, 
                               alpha=0.8, edgecolor='black', linewidth=0.5)

# 添加更清晰的区域着色
detection_region = Ellipse(xy=(center_detection_x, center_detection_y), width=1.0, height=0.7, 
                         angle=-20, edgecolor=semantic_guided_color, 
                         fc=semantic_guided_color, alpha=0.1)
segmentation_region = Ellipse(xy=(center_segmentation_x, center_segmentation_y), width=0.8, height=0.6, 
                            angle=15, edgecolor=semantic_guided_color, 
                            fc=semantic_guided_color, alpha=0.1)
ax[2].add_patch(detection_region)
ax[2].add_patch(segmentation_region)

# 添加连接原始数据的箭头
# 从原点到对象检测区域
ax[2].arrow(0, 0, center_detection_x * 0.7, center_detection_y * 0.7, 
          head_width=0.1, head_length=0.1, 
          fc=semantic_guided_color, ec=semantic_guided_color, alpha=0.6)

# 从原点到分割区域
ax[2].arrow(0, 0, center_segmentation_x * 0.7, center_segmentation_y * 0.7, 
          head_width=0.1, head_length=0.1, 
          fc=semantic_guided_color, ec=semantic_guided_color, alpha=0.6)

# 添加语义区域标注
ax[2].annotate('Object Detection\nGuided Region', xy=(center_detection_x, center_detection_y), 
              xytext=(center_detection_x - 0.6, center_detection_y - 0.8),
              fontsize=14, 
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85),
              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", 
                            color='black', alpha=0.7))

ax[2].annotate('Segmentation\nGuided Region', xy=(center_segmentation_x, center_segmentation_y), 
              xytext=(center_segmentation_x + 0.2, center_segmentation_y - 0.8),
              fontsize=14, 
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85),
              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", 
                            color='black', alpha=0.7))

# 添加SDI方法的整体说明文字
ax[2].annotate('Guided by task semantics\nEnhancing feature diversity', 
              xy=(0, 0), xytext=(0, 0.8),  # 位置在图的上方中央
              ha='center', fontsize=14, 
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray", alpha=0.85))

# Set common styling for all subplots
for i in range(3):
    ax[i].set_xlim(-2, 2)
    ax[i].set_ylim(-1.5, 1.5)
    ax[i].set_xlabel('Feature Dimension 1', fontsize=15)
    if i == 0:  # 只在第一个子图显示y轴标签
        ax[i].set_ylabel('Feature Dimension 2', fontsize=15)
    else:
        ax[i].set_ylabel('')  # 移除其他子图的y轴标签
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].grid(True, linestyle='--', alpha=0.3)

# 调整布局，为底部的图例留出空间
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.26)

# 创建统一的图例 - 添加半透明底色
# 为统一图例创建句柄和标签
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=original_color, markersize=10, 
              markeredgecolor='black', label='Original Samples'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=traditional_color, markersize=8, 
              label='Traditional Augmented Samples'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=generative_color, markersize=8, 
              markeredgecolor='black', label='Generated Samples'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=semantic_guided_color, markersize=8, 
              markeredgecolor='black', label='Semantic-Guided Samples')
]

# 在统一图例底部添加半透明底色矩形
legend = fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
                  bbox_to_anchor=(0.5, 0.02), 
                  prop={'family': 'Times New Roman', 'size': 14}, 
                  frameon=True, fancybox=True)

# 添加图例底部半透明矩形 - 在主图上获取图例位置并创建矩形
fig.canvas.draw()  # 确保图例已渲染以获取其位置
legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
# 扩大矩形尺寸
rect_x0 = legend_bbox.x0 - 0.05
rect_width = legend_bbox.width + 0.1
rect_y0 = legend_bbox.y0 - 0.01
rect_height = legend_bbox.height + 0.02

# 在图例下添加半透明矩形
rect = plt.Rectangle((rect_x0, rect_y0), rect_width, rect_height,
                   transform=fig.transFigure, fill=True,
                   color='white', alpha=0.8, zorder=-1)
fig.patches.append(rect)

# 将图例移到最前面
legend.set_zorder(100)

# Save as PDF
with PdfPages('data_augmentation_comparison.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight')

plt.show()
