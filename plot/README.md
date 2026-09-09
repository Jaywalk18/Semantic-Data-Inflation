# Plotting Scripts for SDI Paper

This directory contains all the plotting scripts used to generate figures in the Semantic Data Inflation (SDI) paper.

## Files Overview

| File | Description | Output |
|------|-------------|--------|
| `imagenet_bar_chart.py` | ImageNet linear probe accuracy comparison (horizontal bar chart) | `imagenet_iclr.pdf` |
| `performance_comparison.py` | CIFAR-10 and ImageNette performance comparison (grouped bar chart) | `performance_comparison.pdf` |
| `tsne_visualization.py` | t-SNE feature space visualization for different augmentation methods | `cifar10_tsne_comparison.pdf` |
| `bar_chart_comparison.py` | Side-by-side bar charts for CIFAR-10 and ImageNette | `performance_comparison.pdf` |
| `point_cloud_visualization.py` | Data augmentation strategy comparison (point cloud illustration) | `data_augmentation_comparison.pdf` |
| `training_curves.py` | Training dynamics with different semantic guidance methods | `training_curves_figure.pdf` |
| `efficiency_comparison.py` | GPU time and memory usage comparison | `compact_comparison.pdf` |
| `figures_presentation.pptx` | PowerPoint file for framework diagrams and additional figures | - |

## Requirements

```bash
pip install matplotlib numpy seaborn scikit-learn torch torchvision pandas scipy
```

## Usage

Run any script directly:

```bash
python imagenet_bar_chart.py
python performance_comparison.py
python tsne_visualization.py
# ... etc
```

## Note

- All scripts use Times New Roman font for publication-quality figures
- Output figures are saved in PDF format for high-quality vector graphics
- Some scripts (e.g., `tsne_visualization.py`, `training_curves.py`) require data files that are not included in this repository
