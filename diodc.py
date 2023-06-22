import cv2
import numpy as np
from skimage import measure
image = cv2.imread('test4.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
labels = measure.label(opened)
wbc_count = 0
rbc_count = 0
for label in np.unique(labels):
    if label == 0:
        continue
    mask = np.zeros(opened.shape, dtype="uint8")
    mask[labels == label] = 255
    cell_pixels = cv2.countNonZero(mask)
    if cell_pixels > 300:  # Adjust the threshold as per your image
        rbc_count += 1
    else:
        wbc_count += 1
print("WBC count:", wbc_count)
print("RBC count:", rbc_count)