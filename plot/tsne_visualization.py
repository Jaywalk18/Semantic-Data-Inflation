import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.manifold import TSNE
import random
import time
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LinearSegmentedColormap

# Set high-quality fonts properly
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # Better math font compatibility
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.dpi'] = 150

from sklearn.neighbors import KernelDensity
import scipy.ndimage
from matplotlib.colors import to_hex, to_rgba

# Professional color schemes
import seaborn as sns
sns.set_style("whitegrid", {'grid.linestyle': ':'})
palette = sns.color_palette('tab10', 10)
colors = [to_hex(x) for x in palette]

MAX_POINTS_PER_CLASS = 300
EPOCHS_TO_VISUALIZE = [1, 1, 2, 5]
BASE_MODEL_PATH = 'cifar10_resnet50_epoch_{}.pth'
OUTPUT_PDF_FILE = 'cifar10_tsne_comparison.pdf'
USE_PRETRAINED = True
SUBPLOT_TITLES = ["Standard Aug", "Raw Duplication", "Generative Inflation", "SDI (ours)"]

def plot_kde_contour(features_tsne, labels, class_id, ax, color, alpha=0.15):
    mask = (labels == class_id)
    if np.sum(mask) > 5:
        points = features_tsne[mask]
        kde = KernelDensity(bandwidth=2.0, kernel='gaussian').fit(points)
        x_min, x_max = features_tsne[:,0].min()-1, features_tsne[:,0].max()+1
        y_min, y_max = features_tsne[:,1].min()-1, features_tsne[:,1].max()+1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
        density = np.exp(kde.score_samples(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
        density = scipy.ndimage.gaussian_filter(density, sigma=3)

        # Create custom colormap for smoother gradient effect
        rgba_color = to_rgba(color)
        base_color = np.array([rgba_color[0], rgba_color[1], rgba_color[2], 0])
        top_color = np.array([rgba_color[0], rgba_color[1], rgba_color[2], alpha*1.5])
        cmap = LinearSegmentedColormap.from_list(f"custom_{color}", [base_color, top_color])
        
        # Enhanced fill and contour effects
        min_level = density.max() * 0.1
        levels = np.linspace(min_level, density.max(), 8)
        ax.contourf(xx, yy, density, levels=levels, cmap=cmap, antialiased=True, zorder=1)
        
        # More refined contour line effect
        contour = ax.contour(xx, yy, density, levels=[density.max() * 0.2],
                  colors=[to_rgba(color, 0.95)], linewidths=1.6, antialiased=True, zorder=2)
        
        # Add glow effect to contour lines
        for collection in contour.collections:
            collection.set_path_effects([
                PathEffects.withStroke(linewidth=2.5, foreground='white', alpha=0.4)
            ])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                             shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=False, num_workers=0)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    if USE_PRETRAINED:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        print("Loaded pretrained ResNet-50 model with ImageNet weights.")
    else:
        model = resnet50(weights=None)
        print("Initialized ResNet-50 model with random weights.")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters()
                   if 'fc' not in name], 'lr': 0.0001},
        {'params': model.fc.parameters(), 'lr': 0.001}
    ], momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    max_epochs = max(EPOCHS_TO_VISUALIZE)
    all_models_exist = True
    for epoch in EPOCHS_TO_VISUALIZE:
        model_path = BASE_MODEL_PATH.format(epoch)
        if not os.path.exists(model_path):
            all_models_exist = False
            break

    if not all_models_exist:
        print("Fine-tuning the model on CIFAR-10...")
        for epoch in range(1, max_epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                if (i+1) % 100 == 0:
                    print(f'Epoch: {epoch}/{max_epochs} | Batch: {i+1}/{len(trainloader)} | '
                          f'Loss: {running_loss/100:.3f} | Acc: {100.*correct/total:.2f}%')
                    running_loss = 0.0

            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            acc = 100. * correct / total
            print(f'Epoch: {epoch} | Test Loss: {test_loss/len(testloader):.3f} | '
                  f'Test Acc: {acc:.2f}% | Time: {time.time()-start_time:.2f}s')
            scheduler.step()
            if epoch in EPOCHS_TO_VISUALIZE:
                model_path = BASE_MODEL_PATH.format(epoch)
                torch.save(model.state_dict(), model_path)
                print(f'Model saved after epoch {epoch}: {model_path}')

    tsne_results = []

    print('\n[Feature Extraction + t-SNE]')
    for i, epoch in enumerate(EPOCHS_TO_VISUALIZE):
        model_path = BASE_MODEL_PATH.format(epoch)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"Extracting features for t-SNE visualization after epoch {epoch}...")
            feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
            features = []
            labels_list = []
            with torch.no_grad():
                for data in testloader:
                    images, labels_ = data
                    images, labels_ = images.to(device), labels_.to(device)
                    output = feature_extractor(images)
                    output = output.view(output.size(0), -1)
                    features.append(output.cpu().numpy())
                    labels_list.append(labels_.cpu().numpy())
            features = np.concatenate(features, axis=0)
            labels_list = np.concatenate(labels_list)
            # Sample to reduce point count
            if MAX_POINTS_PER_CLASS > 0:
                sampled_features = []
                sampled_labels = []
                for class_id in range(10):
                    class_indices = np.where(labels_list == class_id)[0]
                    if len(class_indices) > MAX_POINTS_PER_CLASS:
                        sampled_indices = random.sample(list(class_indices), MAX_POINTS_PER_CLASS)
                    else:
                        sampled_indices = class_indices
                    sampled_features.append(features[sampled_indices])
                    sampled_labels.append(labels_list[sampled_indices])
                features = np.concatenate(sampled_features, axis=0)
                labels_list = np.concatenate(sampled_labels)
                print(f"Sampled data: {features.shape[0]} points (max {MAX_POINTS_PER_CLASS} per class)")

            # PCA dimensionality reduction to speed up t-SNE
            print(f"Original feature dimension: {features.shape[1]}")
            if features.shape[1] > 50:
                print("Performing PCA reduction before t-SNE...")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=50)
                features = pca.fit_transform(features)
                print(f"Reduced feature dimension: {features.shape[1]}")
            print(f"Performing t-SNE dimensionality reduction for epoch {epoch}...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            features_tsne = tsne.fit_transform(features)
            tsne_results.append((features_tsne, labels_list))
            print(f"t-SNE visualization for epoch {epoch} completed.")
        else:
            print(f"Model for epoch {epoch} not found: {model_path}")
            tsne_results.append((None, None))

    # Create high-quality visualization
    n_cols = len(tsne_results)
    
    # Create beautiful figure with background
    fig = plt.figure(figsize=(5 * n_cols, 6), facecolor='#fdfdfd')
    
    # Use tight_layout at the beginning to ensure proper rendering with Times New Roman
    plt.tight_layout()
    
    # Then adjust subplot parameters explicitly
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.92, wspace=0.1, hspace=0.1)
    
    axes = []

    # Find global min/max values
    x_min_global, x_max_global = float('inf'), float('-inf')
    y_min_global, y_max_global = float('inf'), float('-inf')

    for features_tsne, _ in tsne_results:
        if features_tsne is not None:
            x_min, x_max = features_tsne[:, 0].min(), features_tsne[:, 0].max()
            y_min, y_max = features_tsne[:, 1].min(), features_tsne[:, 1].max()
            x_min_global = min(x_min_global, x_min)
            x_max_global = max(x_max_global, x_max)
            y_min_global = min(y_min_global, y_min)
            y_max_global = max(y_max_global, y_max)

    # Enlarge margins to prevent clipping
    x_range = x_max_global - x_min_global
    y_range = y_max_global - y_min_global
    margin = max(x_range, y_range) * 0.25
    x_min_global -= margin
    x_max_global += margin
    y_min_global -= margin
    y_max_global += margin

    # Create each subplot
    for i in range(n_cols):
        ax = fig.add_subplot(1, n_cols, i+1)
        ax.set_xlim(x_min_global, x_max_global)
        ax.set_ylim(y_min_global, y_max_global)
        axes.append(ax)

    scatter_handles = [None for _ in range(10)]

    for k, (features_tsne, lbls) in enumerate(tsne_results):
        ax = axes[k]
        if features_tsne is None:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   fontsize=20, fontweight='semibold', transform=ax.transAxes,
                   family='Times New Roman')  # Explicitly set font family here
            continue

        # Draw background contours first
        for j in range(10):
            mask = (lbls == j)
            plot_kde_contour(features_tsne, lbls, j, ax, colors[j], alpha=0.18)

        # Then draw scatter points
        for j in range(10):
            mask = (lbls == j)
            x = features_tsne[mask, 0]
            y = features_tsne[mask, 1]
            
            # Add halo effect
            ax.scatter(
                x, y, s=60, color='white', edgecolors='none', alpha=0.25, zorder=2
            )
            
            # Draw main scatter points
            handle = ax.scatter(
                x, y,
                
                color=colors[j],
                label=classes[j] if k==0 else None,
                s=32, alpha=0.92, edgecolors='#ffffff', linewidths=0.8, zorder=3,
                clip_on=True
            )
            
            if k == 0:
                scatter_handles[j] = handle

        # Set subplot style
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        # Add title with border style
        title = ax.set_title(SUBPLOT_TITLES[k], fontsize=20, fontweight='bold', pad=15, family='Times New Roman')
        title.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white')
        ])
        
        # Beautify borders with KDE style
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#333333')
            spine.set_linewidth(1.8)
            spine.set_linestyle('-')
        
        # Set background style
        ax.patch.set_facecolor('#f9f9f9')
        ax.grid(True, linestyle='--', alpha=0.25, color='gray', zorder=0, linewidth=0.8)

    # Add elegant legend
    leg = fig.legend(
        handles=scatter_handles,
        labels=classes,
        loc='lower center', ncol=10, 
        fontsize=17,
        frameon=True,
        fancybox=True,
        shadow=True,
        edgecolor='#222222',
        columnspacing=2.0,
        handletextpad=0.8,
        title_fontsize=18,
        bbox_to_anchor=(0.5, 0.02),
        prop={'family': 'Times New Roman'}  # Explicitly set font family for legend
    )
    
    # Beautify legend
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_linewidth(1.2)
    leg.get_frame().set_edgecolor('#222222')

    # Add global title
    suptitle = fig.suptitle('Feature Space Visualization with t-SNE', 
                fontsize=24, fontweight='bold', y=0.98, family='Times New Roman')
    suptitle.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Add annotation
    # plt.figtext(0.5, 0.01, 'CIFAR-10 Dataset - ResNet50 Feature Embeddings', ha='center', fontsize=16, fontstyle='italic', family='Times New Roman', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#999999', linewidth=1))

    # Use tight_layout with rect to ensure proper spacing
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save figures with higher pad_inches to ensure edges aren't clipped
    plt.savefig(OUTPUT_PDF_FILE, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.savefig(OUTPUT_PDF_FILE.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f't-SNE visualization saved as: {OUTPUT_PDF_FILE} and {OUTPUT_PDF_FILE.replace(".pdf", ".png")}')
    plt.show()

if __name__ == "__main__":
    main()
