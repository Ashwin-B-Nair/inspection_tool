
from huggingface_hub import hf_hub_download
chkpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth")
print(chkpt_path)
from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_b"](checkpoint=chkpt_path)
sam.to("cpu")
predictor = SamPredictor(sam)

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load the image
img = cv2.imread("C:/Users/ashwi/Desktop/rgb_028.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load the model (replace with your checkpoint path)
sam = sam_model_registry["vit_b"](checkpoint=chkpt_path)
sam.to("cpu")  # or "cuda" if you have a GPU

# Generate masks
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img_rgb)

# Visualize all masks (for inspection)
for m in masks:
    mask = m['segmentation']
    img_vis = img_rgb.copy()
    img_vis[mask] = [255, 0, 0]  # Color this mask red for visualization
    cv2.imshow('Mask', cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
