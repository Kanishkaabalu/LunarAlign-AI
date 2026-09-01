import cv2
import numpy as np

# Load reference image
image = cv2.imread("data/reference.jpg")

if image is None:
    print("Error: reference.jpg not found!")
    exit()

# Get image dimensions
height, width = image.shape[:2]

# Rotate and scale the image
center = (width // 2, height // 2)
matrix = cv2.getRotationMatrix2D(center, 15, 0.85)

transformed = cv2.warpAffine(
    image,
    matrix,
    (width, height)
)

# Change brightness to simulate different illumination
transformed = cv2.convertScaleAbs(
    transformed,
    alpha=0.8,
    beta=30
)

# Save as source image
cv2.imwrite("data/source.jpg", transformed)

print("Source image created successfully!")