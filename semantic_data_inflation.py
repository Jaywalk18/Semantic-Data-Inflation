
import sys
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import argparse
import warnings
import shutil
import json
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache
import hashlib
import time

# --- Script Information ---
# This script processes a given image dataset by applying a multi-scale adaptive mechanism (SDI).
# It generates a new, mixed dataset containing both original and augmented images.
# Augmentations are selected based on image quality metrics, using YOLO for object detection
# and Segment-Anything Model (SAM) for segmentation.
#
# Usage:
# python process_dataset.py --train_dir ./path/to/train --val_dir ./path/to/val --output_dir ./output --device cuda:0 --use_half_precision

warnings.filterwarnings("ignore")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="SDI Multi-scale Adaptive Mechanism")
parser.add_argument("--train_dir", type=str, default='./data/train', help="Directory for training data.")
parser.add_argument("--val_dir", type=str, default='./data/val', help="Directory for validation data.")
parser.add_argument("--output_dir", type=str, default='./output/APTOS2019', help="Root directory for output.")
parser.add_argument("--data_name", type=str, default='custom', choices=['cifar10', 'cifar100', 'imagenet', 'custom'], help="Name of the dataset.")
parser.add_argument("--alpha", type=float, default=0.4, help="Weight for resolution factor in quality score.")
parser.add_argument("--beta", type=float, default=0.3, help="Weight for clarity factor in quality score.")
parser.add_argument("--gamma", type=float, default=0.3, help="Weight for information density factor in quality score.")
parser.add_argument("--device", type=str, default='cuda:0', help="Device to run on (e.g., 'cuda:0', 'cpu').")
parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing images.")
parser.add_argument("--sam_batch_size", type=int, default=8, help="Batch size for SAM-related processing.")
parser.add_argument("--use_half_precision", action='store_true', help="Use half-precision (FP16) for SAM inference to speed up and save memory.")
parser.add_argument("--sam_cache_size", type=int, default=50, help="Number of SAM image encodings to cache in memory.")
parser.add_argument("--yolo_model_path", type=str, default="yolo11x.pt", help="Path to YOLO model weights.")
parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="Path to SAM model weights.")
args = parser.parse_args()


# --- Device and Optimization Setup ---
def setup_device(device_str):
    """Sets up the computation device (CPU or GPU)."""
    if 'cuda' in device_str and torch.cuda.is_available():
        try:
            gpu_id = int(device_str.split(':')[-1])
            if gpu_id < torch.cuda.device_count():
                device = f'cuda:{gpu_id}'
                torch.cuda.set_device(gpu_id)
            else:
                print(f"Warning: GPU {gpu_id} not available. Falling back to GPU 0.")
                device = "cuda:0"
                torch.cuda.set_device(0)
        except (ValueError, IndexError):
            print("Warning: Invalid CUDA device format. Falling back to GPU 0.")
            device = "cuda:0"
            torch.cuda.set_device(0)
    else:
        if 'cuda' in device_str:
            print("Warning: CUDA not available. Falling back to CPU.")
        device = "cpu"
    return device

device = setup_device(args.device)
args.num_workers = min(args.num_workers, mp.cpu_count())

print(f"Using device: {device}")
print(f"Using {args.num_workers} workers, batch size: {args.batch_size}")

if device.startswith('cuda'):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"CUDA optimizations enabled on {device}.")


# --- Model and Cache Initialization ---
print("Loading YOLO model...")
try:
    yolo_model = YOLO(args.yolo_model_path)
    if device.startswith('cuda'):
        yolo_model.to(device)
    print(f"YOLO model loaded from {args.yolo_model_path}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Trying alternative paths...")
    for alt_path in ["yolov8x.pt", "yolo8x.pt", "yolov8n.pt"]:
        if os.path.exists(alt_path):
            print(f"Found alternative YOLO model at {alt_path}")
            yolo_model = YOLO(alt_path)
            if device.startswith('cuda'):
                yolo_model.to(device)
            print(f"YOLO model loaded from {alt_path}")
            break
    else:
        raise FileNotFoundError(f"No YOLO model found at {args.yolo_model_path} or alternative paths")

# SAM model will be loaded lazily when needed
sam_model = None
sam_predictor = None

class SAMImageCache:
    """A simple LRU cache for SAM image encodings to speed up inference."""
    def __init__(self, max_size=50):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size

    def _get_image_hash(self, image_np):
        return hashlib.md5(image_np.tobytes()).hexdigest()

    def get_encoding(self, image_np):
        image_hash = self._get_image_hash(image_np)
        if image_hash in self.cache:
            self.access_order.remove(image_hash)
            self.access_order.append(image_hash)
            return self.cache[image_hash]
        return None

    def set_encoding(self, image_np, encoding):
        image_hash = self._get_image_hash(image_np)
        if len(self.cache) >= self.max_size:
            oldest_hash = self.access_order.pop(0)
            del self.cache[oldest_hash]
        self.cache[image_hash] = encoding
        self.access_order.append(image_hash)

    def clear(self):
        self.cache.clear()
        self.access_order.clear()

sam_image_cache = SAMImageCache(max_size=args.sam_cache_size)

def get_sam_predictor():
    """Lazily initializes and returns the SAM predictor."""
    global sam_model, sam_predictor
    if sam_predictor is None:
        if not os.path.exists(args.sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found at {args.sam_checkpoint}. Please download it.")
        print("Initializing SAM model...")
        sam_model = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
        sam_model.to(device=device)
        if args.use_half_precision and device.startswith('cuda'):
            sam_model.half()
            print("SAM model converted to half-precision (FP16).")
        sam_predictor = SamPredictor(sam_model)
        print("SAM model initialized.")
    return sam_predictor


# --- Image Processing and Saving Utilities ---
def draw_mask_fast(image, mask, alpha=0.5):
    """Draws a mask on an image using OpenCV for speed."""
    if mask is None: return image
    color = np.array([30, 144, 255], dtype=np.uint8)  # Blue
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask] = color
    return cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)

def draw_box_fast(image, box, color=(0, 255, 0), thickness=2):
    """Draws a bounding box on an image using OpenCV."""
    if box is None: return image
    x1, y1, x2, y2 = map(int, box)
    return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color, thickness)

def save_image_fast(image_np, save_path, mask=None, bbox=None, model_type=""):
    """Saves an image with optional mask and bbox overlays using OpenCV."""
    try:
        result_img = image_np.copy()
        if "SAM" in model_type and mask is not None:
            result_img = draw_mask_fast(result_img, mask)
        if "YOLO" in model_type and bbox is not None:
            img_size = max(image_np.shape[:2])
            thickness = max(1, int(img_size / 200))
            result_img = draw_box_fast(result_img, bbox, thickness=thickness)

        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_bgr)
        return True
    except Exception as e:
        print(f"Error saving image {save_path}: {e}")
        return False

SEMANTIC_SCALES = {"S1": "YOLO", "S2": "SAM", "S3": "YOLO-SAM"}


# --- Dataset Loading ---
print(f"Loading dataset: {args.data_name}")
trainset, testset, classes_name = None, None, []

if args.data_name == 'custom':
    if os.path.exists(args.train_dir):
        trainset = datasets.ImageFolder(root=args.train_dir, transform=transforms.ToTensor())
        classes_name = trainset.classes
        print(f"Loaded {len(trainset)} training images from {args.train_dir}")
    if os.path.exists(args.val_dir):
        testset = datasets.ImageFolder(root=args.val_dir, transform=transforms.ToTensor())
        if not classes_name and testset:
            classes_name = testset.classes
        print(f"Loaded {len(testset) if testset else 0} validation images from {args.val_dir}")
else: # Add other dataset loaders here if needed (e.g., CIFAR)
    raise NotImplementedError(f"Dataset loader for '{args.data_name}' is not implemented.")

if trainset is None and testset is None:
    raise ValueError("No valid dataset found. Check --train_dir and --val_dir paths.")


# --- Image Quality and Scale Selection ---
@lru_cache(maxsize=1000)
def calculate_image_quality(height, width, laplacian_var, entropy):
    """Calculates a quality score for an image based on its properties."""
    resolution_factor = min(1.0, max(height, width) / 512.0)
    clarity_factor = min(1.0, (laplacian_var * (resolution_factor ** 2)) / 500.0)
    info_density = min(1.0, entropy / 8.0)
  
    quality_score = (args.alpha * resolution_factor +
                     args.beta * clarity_factor +
                     args.gamma * info_density)
    return quality_score

def analyze_image_properties(image_np):
    """Analyzes an image to extract properties for quality calculation."""
    height, width = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist /= hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return calculate_image_quality(height, width, laplacian_var, entropy)

def select_semantic_scale(quality_score):
    """Selects the semantic processing scale based on the quality score."""
    if quality_score < 0.35:
        return "S1"  # Low quality -> Use robust detector (YOLO)
    elif quality_score < 0.65:
        return "S3"  # Medium quality -> Combine detector and segmenter (YOLO-SAM)
    else:
        return "S2"  # High quality -> Use fine-grained segmenter (SAM)

# --- Core Processing Logic ---
def save_to_mixed_dataset(image, class_name, index, split_name, suffix="original", mask=None, bbox=None, model_type=""):
    """Saves an image directly to the final mixed dataset directory."""
    output_dir = os.path.join(args.output_dir)
    class_dir = os.path.join(output_dir, split_name, class_name)
    os.makedirs(class_dir, exist_ok=True)
  
    filename = f"{index}_{class_name}_{suffix}.png"
    save_path = os.path.join(class_dir, filename)

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0) * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    save_image_fast(image, save_path, mask=mask, bbox=bbox, model_type=model_type)


def process_yolo_batch(batch_data):
    """Performs YOLO inference on a batch of images."""
    if not batch_data: return []
    images_np = [data[0] for data in batch_data]
  
    with torch.no_grad():
        yolo_results = yolo_model(images_np, verbose=False, device=device)
  
    results = []
    for i, (image_np, label_idx, index, split_name, scale, score) in enumerate(batch_data):
        best_box = None
        max_conf = 0
        if i < len(yolo_results) and yolo_results[i].boxes and len(yolo_results[i].boxes) > 0:
            best_box_idx = yolo_results[i].boxes.conf.argmax()
            best_box = yolo_results[i].boxes.xyxy[best_box_idx].cpu().numpy()
            max_conf = yolo_results[i].boxes.conf[best_box_idx].item()
        results.append((image_np, label_idx, index, split_name, scale, score, best_box, max_conf))
    return results


def process_sam_batch(batch_data):
    """Performs SAM inference on a batch of pre-processed data."""
    if not batch_data: return []
    predictor = get_sam_predictor()
    results = []
    for image_np, label_idx, index, split_name, scale, score, bbox, conf in batch_data:
        mask = None
        try:
            # Use a full-image box if none is provided (for S2)
            input_box = bbox if bbox is not None else np.array([0, 0, image_np.shape[1], image_np.shape[0]])

            # Use cached encoding if available
            cached_encoding = sam_image_cache.get_encoding(image_np)
            if cached_encoding is not None:
                predictor.features = cached_encoding
                predictor.original_size = image_np.shape[:2]
                predictor.input_size = predictor.transform.apply_image(image_np).shape[:2]
                predictor.is_image_set = True
            else:
                predictor.set_image(image_np)
                sam_image_cache.set_encoding(image_np, predictor.features.clone())

            with torch.no_grad():
                masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
                mask = masks[0]
        except Exception as e:
            print(f"Error during SAM processing for image {index}: {e}")
        
        results.append((image_np, label_idx, index, split_name, scale, score, bbox, conf, mask))
    return results

def process_and_save_batch(batch_data):
    """Main batch processing pipeline: YOLO -> SAM -> Save."""
    s1_batch, s2_batch, s3_batch = [], [], []
    for data in batch_data:
        scale = data[4]
        if scale == "S1": s1_batch.append(data)
        elif scale == "S2": s2_batch.append(data)
        else: s3_batch.append(data)
        
    all_results = []
    # Process S1 (YOLO only)
    if s1_batch:
        yolo_processed = process_yolo_batch(s1_batch)
        all_results.extend([(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], None) for r in yolo_processed]) # Add empty mask

    # Process S2 (SAM only)
    if s2_batch:
        s2_with_dummy_box = [(d[0], d[1], d[2], d[3], d[4], d[5], None, 0) for d in s2_batch] # Add dummy box/conf
        sam_processed = process_sam_batch(s2_with_dummy_box)
        all_results.extend(sam_processed)
        
    # Process S3 (YOLO -> SAM)
    if s3_batch:
        yolo_processed = process_yolo_batch(s3_batch)
        sam_processed = process_sam_batch(yolo_processed)
        all_results.extend(sam_processed)

    # Save all results
    stats = []
    for image_np, label_idx, index, split, scale, score, bbox, conf, mask in all_results:
        class_name = classes_name[label_idx]
        model_type = SEMANTIC_SCALES[scale]
        save_to_mixed_dataset(image_np, class_name, index, split, f"augmented_{scale.lower()}", mask, bbox, model_type)
        stats.append(scale)
    return stats


def process_dataset(dataset, split_name):
    """Processes an entire dataset split (train or val)."""
    print(f"\nProcessing {len(dataset)} {split_name} images...")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=(device.startswith('cuda')))
    
    scale_stats = {"S1": 0, "S2": 0, "S3": 0}
    total_processed = 0
    start_time = time.time()

    with tqdm(total=len(dataset), desc=f"Processing {split_name}") as pbar:
        for i, (images, labels) in enumerate(dataloader):
            current_batch_size = images.shape[0]
            start_idx = i * args.batch_size
            
            # --- 1. Save Original Images ---
            for j in range(current_batch_size):
                img_idx = start_idx + j
                class_name = classes_name[labels[j].item()]
                save_to_mixed_dataset(images[j], class_name, img_idx, split_name, "original")

            # --- 2. Prepare for Augmentation ---
            prepared_data = []
            for j in range(current_batch_size):
                img_idx = start_idx + j
                image_np_rgb = (images[j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                quality_score = analyze_image_properties(image_np_rgb)
                selected_scale = select_semantic_scale(quality_score)
                prepared_data.append((image_np_rgb, labels[j].item(), img_idx, split_name, selected_scale, quality_score))
            
            # --- 3. Process and Save Augmented Images ---
            # Process in smaller chunks to manage SAM memory
            for k in range(0, len(prepared_data), args.sam_batch_size):
                chunk = prepared_data[k:k + args.sam_batch_size]
                processed_scales = process_and_save_batch(chunk)
                for scale in processed_scales:
                    scale_stats[scale] += 1

            total_processed += current_batch_size
            pbar.update(current_batch_size)
            pbar.set_postfix({
                'S1': scale_stats['S1'], 'S2': scale_stats['S2'], 'S3': scale_stats['S3'],
                'GPU_Mem': f'{torch.cuda.memory_allocated(device)/1e9:.1f}GB' if device.startswith('cuda') else 'N/A'
            })
            
            if i > 0 and i % 20 == 0 and device.startswith('cuda'):
                torch.cuda.empty_cache() # Periodically clear cache

    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n{split_name.upper()} PROCESSING SUMMARY:")
    print(f"Total processed: {total_processed}")
    print(f"Total time: {processing_time:.2f}s")
    if total_processed > 0:
        print(f"Average time per image: {processing_time/total_processed:.3f}s")
        print(f"Scale distribution: S1={scale_stats['S1']}, S2={scale_stats['S2']}, S3={scale_stats['S3']}")
        
    sam_image_cache.clear()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def main():
    """Main execution function."""
    print("=" * 60)
    print("SDI Multi-scale Adaptive Dataset Processing")
    print("=" * 60)
    
    # Print configuration
    print(f"Dataset: {args.data_name} ({len(classes_name)} classes)")
    print(f"Device: {device}")
    if device.startswith('cuda'):
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Half precision for SAM: {args.use_half_precision}")
    print("-" * 60)

    # Make sure output directories exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()
    
    try:
        # Process training set
        if trainset:
            process_dataset(trainset, "train")
        
        # Process validation set
        if testset:
            process_dataset(testset, "val")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup - explicitly manage global variables
        global sam_model, sam_predictor
        if sam_predictor is not None:
            del sam_predictor
            sam_predictor = None
        if sam_model is not None:
            del sam_model
            sam_model = None
        sam_image_cache.clear()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)
    print(f"Final mixed dataset saved to: {args.output_dir}")
    print(f"Total processing time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
