
# Dataset Information

This directory is used to store datasets for Semantic Data Inflation (SDI).

## Automatic Download

The following datasets will be automatically downloaded when running the script:
- CIFAR-10
- CIFAR-100

## Manual Download Instructions

### ImageNette Dataset

1. Download the ImageNette dataset:
```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
```

2. Extract the dataset:
```bash
tar -xzf imagenette2-320.tgz
```

3. Move or link the extracted folder to this directory (optional):
```bash
ln -s /path/to/imagenette2-320 ./imagenette
```

### Custom Datasets

For custom datasets, organize your data in the standard ImageFolder format:
```
custom_dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

Then point to this directory using the `--train_dir` and `--val_dir` arguments.
