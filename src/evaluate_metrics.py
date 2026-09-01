import cv2
import numpy as np

# Load images
reference = cv2.imread("data/reference.jpg", cv2.IMREAD_GRAYSCALE)
source = cv2.imread("data/source.jpg", cv2.IMREAD_GRAYSCALE)
aligned = cv2.imread("results/aligned_image.jpg", cv2.IMREAD_GRAYSCALE)

if reference is None or source is None or aligned is None:
    print("Error: Could not load one or more images.")
    exit()

# ---------- RMSE ----------
# Resize aligned image if needed
if reference.shape != aligned.shape:
    aligned = cv2.resize(
        aligned,
        (reference.shape[1], reference.shape[0])
    )

rmse = np.sqrt(
    np.mean(
        (reference.astype(np.float32) -
         aligned.astype(np.float32)) ** 2
    )
)

# ---------- SIFT Feature Matching ----------
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(reference, None)
kp2, des2 = sift.detectAndCompute(source, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

# Use up to 100 best matches
good_matches = matches[:min(100, len(matches))]

# ---------- RANSAC ----------
if len(good_matches) >= 4:
    ref_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    src_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        ref_pts,
        cv2.RANSAC,
        5.0
    )

    inliers = int(np.sum(mask)) if mask is not None else 0
    inlier_ratio = (inliers / len(good_matches)) * 100

else:
    inliers = 0
    inlier_ratio = 0

# ---------- Results ----------
print("\n========== LunarAlign-AI Evaluation ==========")
print("Reference keypoints:", len(kp1))
print("Source keypoints:", len(kp2))
print("Total initial matches:", len(matches))
print("Matches used for RANSAC:", len(good_matches))
print("Reliable RANSAC inliers:", inliers)
print(f"Inlier Ratio: {inlier_ratio:.2f}%")
print(f"RMSE after alignment: {rmse:.2f}")
print("==============================================")