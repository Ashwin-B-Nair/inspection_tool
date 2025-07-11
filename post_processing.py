import cv2
import os
import sys
import time
import torch
import argparse
import itertools
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from utils import find_bounding_box, create_gif, detect_defects

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

#Argument parser
parser = argparse.ArgumentParser(description="Defect Detection Script")
parser.add_argument('--image_folder', type=str, required=True,
                    help="Path to the folder containing input images")
parser.add_argument('--results_folder', type=str, required=True,
                    help="Path to the folder where results will be saved")

args = parser.parse_args()

# Folders
image_folder = args.image_folder
results_folder = args.results_folder
os.makedirs(results_folder, exist_ok=True)

default_params = (3, 'DEED', 1)
image_files = sorted([f for f in os.listdir(image_folder) if f.startswith('rgb_') and f.endswith('.png')])
# selected_images = image_files
selected_images = image_files[40:]
total_images = len(selected_images)


print(f"\n=== Running normal operation with default params ===")
start_total = time.perf_counter()

for idx, image_file in enumerate(selected_images, 1):
    img_path = os.path.join(image_folder, image_file)
    print(f"Processing img ({idx:03}/{total_images:03}): {image_file} ... ", end="")
    start_time = time.perf_counter()
    result = detect_defects(predictor, img_path, morph_params=default_params)
    elapsed = time.perf_counter() - start_time

    if result["status"] == "skipped":
        print(f"SKIPPED (no bbox) [{elapsed:.3f}s]")

    elif result["status"] == "processed":
        print(f"PROCESSED [{elapsed:.3f}s]")
        save_path = os.path.join(results_folder, image_file)
        cv2.imwrite(save_path, result["result_img"])
        cv2.imwrite(save_path, cv2.cvtColor(result["result_img"], cv2.COLOR_BGR2RGB))
        # img = Image.fromarray(result["result_img"])
        # img.save(save_path)

    elif result["status"] == "error":
        print(f"ERROR: {result['message']} [{elapsed:.3f}s]")

total_elapsed = time.perf_counter() - start_total
print(f"Total processing time for {total_images} images: {total_elapsed:.3f} seconds")
create_gif(results_folder, os.path.join(results_folder, 'gif.gif'), fps = 2)