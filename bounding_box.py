import cv2 
import numpy as np

def find_bounding_box (image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define yellow range (tune as needed)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return image, (x, y, w, h)
    else:
        print("No yellow contours found")
        return image, None
    

def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(image, (x - 30, y ), (x + 30 + w , y + 250 + h), color, thickness)
        # Optional: Add text label
        cv2.putText(image, f'Yellow Sticker', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    return image

image_path = 'C:/Users/ashwi/Desktop/rgb_000.png'
image, bbox = find_bounding_box(image_path)

if image is not None:
        if bbox is not None:
            
            result_image = draw_bounding_box(image.copy(), bbox)
            
            cv2.imshow('Original with Bounding Box', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            cv2.imwrite('C:/Users/ashwi/Desktop/rgb_000_bbox.png', result_image)