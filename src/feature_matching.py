import cv2
import matplotlib.pyplot as plt

# Load images in grayscale
reference = cv2.imread("data/reference.jpg", cv2.IMREAD_GRAYSCALE)
source = cv2.imread("data/source.jpg", cv2.IMREAD_GRAYSCALE)

if reference is None or source is None:
    print("Error: Could not load one or both images.")
    exit()

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(reference, None)
kp2, des2 = sift.detectAndCompute(source, None)

print("Reference keypoints:", len(kp1))
print("Source keypoints:", len(kp2))

# Match descriptors using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

print("Total matches:", len(matches))

# Draw best matches
result = cv2.drawMatches(
    reference,
    kp1,
    source,
    kp2,
    matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Save result
cv2.imwrite("results/feature_matches.jpg", result)

# Display result
plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Lunar Image Feature Matching")
plt.axis("off")
plt.show()

print("Feature matching completed successfully!")
print("Result saved in results/feature_matches.jpg")