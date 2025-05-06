import sys

from PIL.ImImagePlugin import number
from sympy.stats.sampling.sample_numpy import numpy

sys.path.append("..")
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

# Define arguments
parser = argparse.ArgumentParser(description="Run Semantic Augmentation on Images With YOLO/SAM/YOLO-SAM.")
parser.add_argument("--model_name",type=str,default='SAM',choices=['YOLO', 'SAM', 'YOLO-SAM', 'ORIGIN'],help="Name of Model")
parser.add_argument("--train_dir", type=str, default='./data', help="Directory for data.")
parser.add_argument("--val_dir", type=str, default='./data', help="Directory for data.")
parser.add_argument("--image_index", type=int, default=0, help="The index for data.")
parser.add_argument("--data_name", type=str, default='custom', choices=['cifar10', 'cifar100', 'imagenet', 'custom'], help="Name of the dataset.")
args = parser.parse_args()

# Helper functions for displaying points, boxes, and masks.
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, image):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    mylw = 50/32*image.shape[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=mylw))


# Load a COCO-pretrained YOLO11n model
sam_checkpoint = "sam_vit_h_4b8939.pth"
yolo_model = YOLO("yolo11x.pt")
model_type = "vit_h"
device = "cuda"
if 'SAM' in args.model_name :
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

# Directory of input and output
output_dir = './output/'+args.data_name+args.model_name
os.makedirs(output_dir, exist_ok=True)
print(args.data_name)
# Download and load training and test sets
if args.data_name == 'cifar10':
    trainset = datasets.CIFAR10(root=args.train_dir, train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR10(root=args.val_dir, train=False, download=True, transform=transforms.ToTensor())
    classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
elif args.data_name == 'cifar100':
    trainset = datasets.CIFAR100(root=args.train_dir, train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR100(root=args.val_dir, train=False, download=True, transform=transforms.ToTensor())
    classes_name = trainset.classes
elif args.data_name == 'imagenet':
    trainset = datasets.ImageFolder(root=args.train_dir,transform=transforms.ToTensor())
    testset = datasets.ImageFolder(root=args.val_dir,transform=transforms.ToTensor())
    classes_name = trainset.classes
elif args.data_name == 'custom':
    testset = None
    trainset = None
    if args.train_dir and args.val_dir:
        trainset = datasets.ImageFolder(root=args.train_dir,transform=transforms.ToTensor())
        testset = datasets.ImageFolder(root=args.val_dir,transform=transforms.ToTensor())
        classes_name = trainset.classes
    elif args.train_dir:
        trainset = datasets.ImageFolder(root=args.train_dir,transform=transforms.ToTensor())
        classes_name = trainset.classes
    elif args.val_dir:
        testset = datasets.ImageFolder(root=args.val_dir,transform=transforms.ToTensor())
        classes_name = testset.classes
    else:
        raise ValueError("Invalid data_dir. Train_dir or val_dir need to be provided.")
else:
    raise ValueError("Invalid data_name. It should be 'cifar10', 'cifar100', 'imagenet', or 'custom'.")

# Merge training and test sets if both are not None
if trainset is not None and testset is not None:
    full_dataset = torch.utils.data.ConcatDataset([trainset, testset])
else:
    full_dataset = trainset if trainset is not None else testset

def process_and_save_image(image, label, index):
    if image.is_cuda:
        image = image.cpu()
    image = transforms.ToPILImage()(image)
    image = np.array(image)
    input_boxes = np.array([-1, -1, image.shape[1] - 2, image.shape[0] - 2])  
    max_confidence = 0
    max_bbox = input_boxes
    if 'YOLO' in args.model_name:
        # Run YOLO model inference
        results = yolo_model(image)
        if results is not None and len(results) > 0 and results[0].boxes is not None:
            # Find the bounding box with the highest confidence
            for box in results[0].boxes:
                confidence = box.conf.item()
                if confidence > max_confidence:
                    max_confidence = confidence
                    max_bbox = box.xyxy[0].cpu().numpy()

    input_boxes = max_bbox
    if 'SAM' in args.model_name:
        # Set the image for the SAM predictor
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        mask = masks[0]

    # Save the result
    plt.figure(figsize=(image.shape[1], image.shape[0]), dpi=1)
    ax = plt.gca()
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    plt.imshow(image)
    if 'SAM' in args.model_name:
        show_mask(mask, ax)
    if 'YOLO' in args.model_name:
        show_box(max_bbox, ax, image)
    class_name = classes_name[label]
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    save_filename = os.path.join(class_dir, f"{index}_{label}.png")
    plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# Process and save each image
for i, (image, label) in enumerate(tqdm(full_dataset, desc="Processing images")):
    if i>args.image_index:
        process_and_save_image(image, label, i)