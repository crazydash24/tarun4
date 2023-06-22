import cv2
import numpy as np

# Load the image
img = cv2.imread('test4.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply median blur to smooth the image and reduce noise
blur = cv2.medianBlur(gray, 5)

# Apply thresholding to convert the image to black and white
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Find contours in the image
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Loop through each contour and identify the blood cells
for contour in contours:
    # Get the area of the contour
    area = cv2.contourArea(contour)
    
    # Ignore small contours
    if area < 100:
        continue
    
    # Draw a bounding box around the contour
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Identify the type of blood cell based on the aspect ratio of the bounding box
    aspect_ratio = float(w) / h
    if aspect_ratio > 1.2:
        cv2.putText(img, "Red Blood Cell", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(img, "White Blood Cell", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()