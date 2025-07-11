import cv2 
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def find_bounding_box (image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define yellow range (tune as needed)
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([30, 255, 255])

    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    # White range
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])


    # Threshold
    mask= cv2.inRange(hsv, lower_blue, upper_blue)
    # mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # mask_white = cv2.inRange(hsv, lower_white, upper_white)
    # mask = cv2.bitwise_or(mask_yellow, mask_white)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # return image, (x - 30, y, w + 60, h + 250)
        return image, (x, y, w, h + 250)
    else:
        print("No yellow contours found")
        return image, None
    

def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    
    if bbox is not None:
        x, y, w, h = bbox
        # cv2.rectangle(image, (x - 30, y ), (x + 30 + w , y + 250 + h), color, thickness)
        cv2.rectangle(image, (x, y ), (x + w , y + h), color, thickness)
        # Optional: Add text label
        cv2.putText(image, f'Blue Sticker', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    return image

def draw_bbox_with_sam_coords(image_path, bbox):
    x, y, w, h = bbox
    bbox_xyxy = n 
    # Open image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Draw bounding box
    draw.rectangle(bbox_xyxy, outline='green', width=3)
    
    # Prepare text
    text = f"{bbox_xyxy}"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Position text slightly above the box
    x, y = bbox_xyxy[0], bbox_xyxy[1]
    text_position = (x, y - 25)
    
    # Draw text
    draw.text(text_position, text, fill='green', font=font)
    
    # Save or display
    image.save('C:/Users/ashwi/Desktop/diffused_img/bbox_with_text_126.png')
    image.show()
    return image


image_path = 'C:/Users/ashwi/Desktop/rgb_001.png'
image, bbox = find_bounding_box(image_path)


if image is not None:
        if bbox is not None:
            
            result_image = draw_bounding_box(image.copy(), bbox)
            # result_image = draw_bbox_with_sam_coords(image_path, bbox)
            
            cv2.imshow('Original with Bounding Box', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            cv2.imwrite('C:/Users/ashwi/Desktop/rgb_001_bbox.png', result_image)