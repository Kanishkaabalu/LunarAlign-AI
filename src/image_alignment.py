import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
reference = cv2.imread("data/reference.jpg")
source = cv2.imread("data/source.jpg")

if reference is None or source is None:
    print("Error: Could not load images.")
    exit()

# Convert to grayscale
gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
gray_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

# Detect SIFT features
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray_ref, None)
kp2, des2 = sift.detectAndCompute(gray_src, None)

# Match descriptors
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Sort and select best matches
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:100]

# Extract corresponding points
ref_pts = np.float32(
    [kp1[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

src_pts = np.float32(
    [kp2[m.trainIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

# Estimate transformation using RANSAC
H, mask = cv2.findHomography(
    src_pts,
    ref_pts,
    cv2.RANSAC,
    5.0
)

# Align source image with reference
aligned = cv2.warpPerspective(
    source,
    H,
    (reference.shape[1], reference.shape[0])
)

# Save aligned image
cv2.imwrite("results/aligned_image.jpg", aligned)

print("Image alignment completed successfully!")
print("Aligned image saved in results/aligned_image.jpg")

# Display comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
plt.title("Reference Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
plt.title("Source Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
plt.title("Aligned Image")
plt.axis("off")

plt.tight_layout()
plt.show()