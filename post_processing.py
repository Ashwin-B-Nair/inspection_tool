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
from utils import find_bounding_box

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def detect_defects(image_path, morph_params = None):
    """Your defect detection logic here"""
    # Example: Read image and process
    try:
      image = cv2.imread(image_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      height, width = image.shape[:2]
      image = image[125:, 300:680]

      _, x,y,w,h = find_bounding_box(image_path)
      if x is None:
            return {"status": "skipped", "message": "No bounding box"}

      input_box = np.array([x, y, x+w, y+h])
      x2, y2  = x + w, y + h
      cropped_image = image[y:y2, x:x2]
      predictor.set_image(cropped_image)
      h, w = cropped_image.shape[:2]
      x=0
      y=0

      #Mask 1
      input_box = np.array([x, y, x+w, y+h])
      input_point = np.array([[(x+w)/2 , y+60]])
      input_label = np.array([1])
      masks1, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, box=None, multimask_output=False,)

      #Mask 2
      input_point = np.array([[(x+w)/2  , y + 170]])
      input_label = np.array([1])
      masks2, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, box=None, multimask_output=False,)

      #Mask 3
      input_point = np.array([[x, y + 40]])
      input_label = np.array([1])
      masks3, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, box=None, multimask_output=False,)

      #Mask 4
      input_point = np.array([[x + w, y+40]])
      input_label = np.array([1])
      masks4, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, box=None, multimask_output=False,)

      original = cropped_image.copy()
      # print(original[0, 0])
      # Defect Masking
      gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
      hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
      _, thresh_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
      contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      final_mask = np.zeros_like(thresh_mask)
      min_area = 10

      for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.fillPoly(final_mask, [contour], 255)

      # colored_overlay = np.zeros_like(original)
      # colored_overlay[final_mask == 255] = [255, 0, 0]
      # result = cv2.addWeighted(original, 0.7, colored_overlay, 0.3, 0)

      # Overlaying masks + segments
      bbox_mask = np.zeros_like(final_mask, dtype=np.uint8)
      x_min, y_min, x_max, y_max = input_box
      # bbox_mask[y_min+150:y_max, x_min+25:x_max-25] = 1
      bbox_mask[60:, :] = 1
      masks1 = np.squeeze(masks1)
      masks2 = np.squeeze(masks2)
      masks3 = np.squeeze(masks3)
      masks4 = np.squeeze(masks4)

      exclude_mask = np.logical_or.reduce([masks1, masks2, masks3, masks4])
      restricted_mask = cv2.bitwise_and(final_mask, final_mask, mask=bbox_mask)
      final_clean_mask = restricted_mask.copy()
      final_clean_mask[exclude_mask] = 0

      colored_overlay = np.zeros_like(original)
      colored_overlay[final_clean_mask == 255] = [255, 0, 0]
      result = cv2.addWeighted(original, 0.7, colored_overlay, 0.3, 0)

      # Dilation + Erosion

      if morph_params:
        k_size, seq, iters = morph_params
        kernel = np.ones((k_size, k_size), np.uint8)
        current_mask = final_clean_mask.copy()

        for op in seq:
            if op == 'D':
                current_mask = cv2.dilate(current_mask, kernel, iterations=iters)
            elif op == 'E':
                current_mask = cv2.erode(current_mask, kernel, iterations=iters)

        deed_mask = current_mask

      else:
        kernel = np.ones((3, 3), np.uint8)  # Adjust size as needed

        # DEED sequence: Dilation -> Erosion -> Erosion -> Dilation
        deed_mask = cv2.dilate(final_clean_mask, kernel, iterations=1)    # D
        deed_mask = cv2.erode(deed_mask, kernel, iterations=1)      # E
        deed_mask = cv2.erode(deed_mask, kernel, iterations=1)      # E
        deed_mask = cv2.dilate(deed_mask, kernel, iterations=1)     # D

      cleaned_overlay = np.zeros_like(original)
      cleaned_overlay[deed_mask == 255] = [255, 0, 0]  # Red in BGR
      result_img = cv2.addWeighted(original, 0.7, cleaned_overlay, 0.3, 0)

      return {"status": "processed", "result_img": result_img}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
import itertools
from PIL import Image

ablation_mode = False  # <<<<<<< SET TO False FOR NORMAL OPERATION

# Folders
image_folder = '/content/drive/MyDrive/inspection_tool/prototype1/rgb3'
base_results_folder = '/content/drive/MyDrive/inspection_tool/results/ablation_26_06' if ablation_mode else '/content/drive/MyDrive/inspection_tool/results/prototype1/trial3_rgb3'
os.makedirs(base_results_folder, exist_ok=True)

if ablation_mode:
  gif_folder = os.path.join(base_results_folder, 'gif')
  os.makedirs(gif_folder, exist_ok=True)

# Define parameter ranges
kernel_sizes = [3, 4, 5]
sequences = ['EDEDE', 'DEDED', 'EDE', 'DED', 'DDEED', 'EEDDE']
iterations_options = [1]
default_params = (3, 'DEED', 1)
# Generate all combinations
param_combinations = list(itertools.product(kernel_sizes, sequences, iterations_options))

# Select every 5th image
image_files = sorted([f for f in os.listdir(image_folder) if f.startswith('rgb_') and f.endswith('.png')])
# selected_images = image_files[::3] if ablation_mode else image_files
selected_images = image_files[40:]
total_images = len(selected_images)

# === MAIN LOOP ===
# if ablation_mode:
#     for params in param_combinations:
#         k_size, seq, iters = params
#         param_str = f"k{k_size}_s{seq}_i{iters}"
#         param_folder = os.path.join(base_results_folder, param_str)
#         os.makedirs(param_folder, exist_ok=True)
#         print(f"\n=== Testing params: {param_str} ===")
#         start_total = time.perf_counter()

#         for idx, image_file in enumerate(selected_images, 1):
#             img_path = os.path.join(image_folder, image_file)
#             print(f"Processing img ({idx:03}/{total_images:03}): {image_file} ... ", end="")

#             start_time = time.perf_counter()
#             result = detect_defects(img_path, morph_params=params)
#             elapsed = time.perf_counter() - start_time

#             if result["status"] == "skipped":
#                 print(f"SKIPPED (no bbox) [{elapsed:.3f}s]")

#             elif result["status"] == "processed":
#                 print(f"PROCESSED [{elapsed:.3f}s]")
#                 save_path = os.path.join(param_folder, image_file)
#                 cv2.imwrite(save_path, cv2.cvtColor(result["result_img"], cv2.COLOR_RGB2BGR))

#             elif result["status"] == "error":
#                 print(f"ERROR: {result['message']} [{elapsed:.3f}s]")

#         total_elapsed = time.perf_counter() - start_total
#         print(f"Total processing time for {total_images} images with params {param_str}: {total_elapsed:.3f} seconds")
#         create_gif(param_folder , os.path.join(gif_folder, f'{param_str}.gif'), fps = 2)
# else:
#     # Normal operation: single parameter set, all images
#     # os.makedirs(base_results_folder, exist_ok=True)

print(f"\n=== Running normal operation with default params ===")
start_total = time.perf_counter()

for idx, image_file in enumerate(selected_images, 1):
    img_path = os.path.join(image_folder, image_file)
    print(f"Processing img ({idx:03}/{total_images:03}): {image_file} ... ", end="")
    start_time = time.perf_counter()
    result = detect_defects(img_path, morph_params=default_params)
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