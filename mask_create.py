import cv2
import numpy as np

def detect_delamination_and_create_mask(image_path, output_path):
    # Load the original image
    img = cv2.imread(image_path)
    original = img.copy()
    
    # Convert to different color spaces for better detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Method 1: HSV-based dark region detection
    # Define range for dark/black regions (delamination)
    lower_dark = np.array([0, 0, 0])      # Lower bound for dark regions
    upper_dark = np.array([180, 0, 5])  # Upper bound (adjust as needed)
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Method 2: Grayscale thresholding for additional detection
    _, thresh_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    
    # Combine both masks
    # combined_mask = cv2.bitwise_or(dark_mask, thresh_mask)
    combined_mask = thresh_mask
    
    # Clean up the mask using morphological operations
    # kernel = np.ones((3,3), np.uint8)
    # # Remove small noise
    # cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    # # # Fill small gaps in delamination lines
    # cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    
    cleaned_mask = combined_mask
    # Filter contours by area to remove very small detections
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(cleaned_mask)
    
    min_area = 10  # Adjust based on your image size
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.fillPoly(final_mask, [contour], 255)
    
    # Create colored overlay (red like your reference)
    colored_overlay = np.zeros_like(original)
    colored_overlay[final_mask == 255] = [0, 0, 255]  # Red in BGR
    
    # Blend original image with colored overlay
    result = cv2.addWeighted(original, 0.7, colored_overlay, 0.3, 0)
    
    # Save results
    cv2.imwrite(output_path.replace('.jpg', '_mask.jpg'), final_mask)
    cv2.imwrite(output_path, result)
    
    return result, final_mask



result, mask = detect_delamination_and_create_mask("C:/Users/ashwi/Desktop/rgb_028.png", 'C:/Users/ashwi/Desktop/output.jpg')
cv2.imshow('Delamination Detection', result)
cv2.imshow('Binary Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
