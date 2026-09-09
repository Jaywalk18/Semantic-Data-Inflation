# Model Files

This directory is used to store model weights for the Semantic Data Inflation (SDI) framework.

## Required Models

### SAM (Segment Anything Model)

1. Download the SAM model:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ./
```

2. The model should be placed in the project root directory (not in this folder).

### YOLO Model

The YOLO model will be automatically downloaded on first run, but you can pre-download it:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
```

The downloaded model will be stored in the ultralytics cache directory.

## Model Verification

To verify that models are correctly installed, run:

```bash
python -c "import os; print(f'SAM model exists: {os.path.exists(\"sam_vit_h_4b8939.pth\")}')"
```

This should output: `SAM model exists: True` if the model is correctly downloaded.