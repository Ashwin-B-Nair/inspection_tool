# Model Download and Setup
from huggingface_hub import hf_hub_download
chkpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth")
# print(chkpt_path)
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

## Load the model (replace with your checkpoint path)
sam = sam_model_registry["vit_b"](checkpoint=chkpt_path)
sam.to("cpu")# or "cuda" if you have a GPU
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

# Helper Function to display masks
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)   #sort in descending order
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))  # Creating a transparent overlay ( size: H x W x 4 rgba channels)
    img[:,:,3] = 0   # alpha = 0: fully transparent
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # random color - semi transparent
        img[m] = color_mask
    ax.imshow(img)
    
# Load the image
img = cv2.imread("C:/Users/ashwi/Desktop/rgb_028.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Automatic Mask Generation - Default Params

masks = mask_generator.generate(img_rgb)
plt.figure(figsize=(20,20))
plt.imshow(img_rgb)
show_anns(masks)
plt.axis('off')
plt.show() 

#Automatic Mask Generation - Modified Params

''' 
Parameters and their function:
1. points_per_side = 32: Controls the density of initial prompt points across the image. ( creates a 32 x 32 grid and generates masks @ each grid point)
2. pred_iou_thresh=0.86: Masks with predicted IoU < 0.86 are discarded.
3. stability_score_thresh=0.92: Masks with stability score < 0.92 are discarded.
4. crop_n_layers=1:  1 = No multi-scale processing (single pass). Higher values → crops image into smaller tiles at multiple scales.
5. crop_n_points_downscale_factor=2:  Reduces point density in subsequent crop layers. If crop_n_layers>1, each layer uses points_per_side/(2^layer).
6. min_mask_region_area=100: Uses OpenCV to filter masks smaller than 100 pixels.
'''

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  
)
masks2 = mask_generator_2.generate(img_rgb)
plt.figure(figsize=(20,20))
plt.imshow(img_rgb)
show_anns(masks2)
plt.axis('off')
plt.show() 