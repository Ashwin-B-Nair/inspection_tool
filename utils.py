import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def create_gif(image_folder, output_gif_path, fps=5):

    image_files = sorted([f for f in os.listdir(image_folder) if f.startswith('rgb_') and f.endswith('.png')])

    images = []
    widths, heights = [], []
    for filename in image_files:
        img_path = os.path.join(image_folder, filename)
        try:
            img = Image.open(img_path)
            images.append(img)
            widths.append(img.width)
            heights.append(img.height)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    max_width, max_height = max(widths), max(heights)

    # Pad images to the max size
    padded_images = []
    for img in images:
        new_img = Image.new("RGB", (max_width, max_height), color=(0,0,0))  # black background
        offset = ((max_width - img.width) // 2, (max_height - img.height) // 2)
        new_img.paste(img, offset)
        padded_images.append(new_img)

    # Set FPS and duration

    duration = int(1000 / fps)

    if padded_images:
        padded_images[0].save(
            output_gif_path,
            save_all=True,
            append_images=padded_images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved at: {output_gif_path}")
    else:
        print("No images loaded, GIF not created.")

def find_bounding_box (image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None

    height, width = image.shape[:2]
    image = image[125:, 300:680]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define yellow range
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([30, 255, 255])
    # lower_blue = np.array([110, 50, 50])
    # upper_blue = np.array([130, 255, 255])
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Threshold
    mask= cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        bbox = np.array([x - 30, y, w + 60, h + 250])
        return image, x - 20, y, w + 120, h + 200
    else:
        print("No blue contours found")
        return image, None, None, None, None
    
    
