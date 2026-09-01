import cv2
import numpy as np

# Load images
reference = cv2.imread("data/reference.jpg", cv2.IMREAD_GRAYSCALE)
source = cv2.imread("data/source.jpg", cv2.IMREAD_GRAYSCALE)

if reference is None or source is None:
    print("Error: Could not load images.")
    exit()

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect features
kp1, des1 = sift.detectAndCompute(reference, None)
kp2, des2 = sift.detectAndCompute(source, None)

# Match features
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches
matches = sorted(matches, key=lambda x: x.distance)

# Use the best matches
good_matches = matches[:100]

# Extract matching points
src_pts = np.float32(
    [kp1[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

dst_pts = np.float32(
    [kp2[m.trainIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

# Find homography using RANSAC
H, mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    5.0
)

# Keep only RANSAC inliers
inlier_matches = [
    good_matches[i]
    for i in range(len(good_matches))
    if mask[i]
]

print("Total initial matches:", len(matches))
print("Matches used for RANSAC:", len(good_matches))
print("Reliable matches after RANSAC:", len(inlier_matches))

# Draw reliable matches
result = cv2.drawMatches(
    reference,
    kp1,
    source,
    kp2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Save result
cv2.imwrite("results/ransac_matches.jpg", result)

print("RANSAC verification completed successfully!")
print("Result saved in results/ransac_matches.jpg")