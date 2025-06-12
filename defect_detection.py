import cv2
import numpy as np

# Load image
img = cv2.imread("C:/Users/ashwi/Desktop/rgb/rgb_014.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred, 10, 100)

# # Apply morphological closing to connect broken edges
# kernel = np.ones((3,3), np.uint8)
# closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# # Find contours to identify crack-like patterns
# contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# original = cv2.imread("C:/Users/ashwi/Desktop/rgb/rgb_014.png")  # Replace with your image filename

# # Draw contours on a copy of the original image
# result = original.copy()
# cv2.drawContours(result, contours, -1, (0, 255, 0), 2)  # Green contours, thickness 2

# Detect lines using Hough transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100)
                        # minLineLength=50, maxLineGap=10)

# Create result image by drawing detected lines
result = img.copy()  # Copy original image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines


cv2.imshow('Contours', result)
cv2.waitKey(0)
cv2.destroyAllWindows()