import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt




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
    
def detect_defects(predictor, image_path, morph_params = None):
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
