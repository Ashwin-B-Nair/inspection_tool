import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import os
import time
import itertools
import sys
from utils import find_bounding_box, create_gif, detect_defects

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)



# Folders
image_folder = '/content/drive/MyDrive/inspection_tool/prototype1/rgb3'
base_results_folder = '/content/drive/MyDrive/inspection_tool/results/ablation_26_06' if ablation_mode else '/content/drive/MyDrive/inspection_tool/results/prototype1/trial3_rgb3'
os.makedirs(base_results_folder, exist_ok=True)

default_params = (3, 'DEED', 1)
# Generate all combinations
param_combinations = list(itertools.product(kernel_sizes, sequences, iterations_options))

# Select every 5th image
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
        save_path = os.path.join(base_results_folder, image_file)
        cv2.imwrite(save_path, result["result_img"])
        cv2.imwrite(save_path, cv2.cvtColor(result["result_img"], cv2.COLOR_BGR2RGB))
        # img = Image.fromarray(result["result_img"])
        # img.save(save_path)

    elif result["status"] == "error":
        print(f"ERROR: {result['message']} [{elapsed:.3f}s]")

total_elapsed = time.perf_counter() - start_total
print(f"Total processing time for {total_images} images: {total_elapsed:.3f} seconds")
create_gif(base_results_folder, os.path.join(base_results_folder, 'gif.gif'), fps = 2)