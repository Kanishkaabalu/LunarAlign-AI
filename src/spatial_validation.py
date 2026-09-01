import cv2
import numpy as np


def calculate_spatial_coverage(points, image_shape, grid_size=4):
    """
    Measures how well the matched points are distributed
    across the image using a grid-based coverage score.
    """

    height, width = image_shape[:2]

    occupied_cells = set()

    cell_width = width / grid_size
    cell_height = height / grid_size

    for point in points:
        x, y = point.ravel()

        col = min(int(x / cell_width), grid_size - 1)
        row = min(int(y / cell_height), grid_size - 1)

        occupied_cells.add((row, col))

    total_cells = grid_size * grid_size
    occupied_count = len(occupied_cells)

    coverage_score = (
        occupied_count / total_cells
    ) * 100

    return coverage_score, occupied_cells


def main():

    # Load advanced preprocessed images
    reference = cv2.imread(
        "results/advanced_reference.jpg",
        cv2.IMREAD_GRAYSCALE
    )

    source = cv2.imread(
        "results/advanced_source.jpg",
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None or source is None:
        print("Error: Preprocessed images not found.")
        return

    # Use SIFT
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(
        reference,
        None
    )

    kp2, des2 = sift.detectAndCompute(
        source,
        None
    )

    if des1 is None or des2 is None:
        print("Error: Not enough features detected.")
        return

    # KNN matching
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    knn_matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    # Lowe's Ratio Test
    good_matches = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print("Error: Not enough reliable matches.")
        return

    # Extract matching points
    ref_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    src_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # RANSAC geometric verification
    H, mask = cv2.findHomography(
        src_pts,
        ref_pts,
        cv2.RANSAC,
        4.0
    )

    if H is None or mask is None:
        print("Error: Homography estimation failed.")
        return

    # Keep only reliable inlier points
    inlier_ref_pts = ref_pts[
        mask.ravel() == 1
    ]

    inlier_src_pts = src_pts[
        mask.ravel() == 1
    ]

    # Calculate spatial coverage
    ref_coverage, ref_cells = calculate_spatial_coverage(
        inlier_ref_pts,
        reference.shape,
        grid_size=4
    )

    src_coverage, src_cells = calculate_spatial_coverage(
        inlier_src_pts,
        source.shape,
        grid_size=4
    )

    # Reprojection error
    projected_pts = cv2.perspectiveTransform(
        src_pts,
        H
    )

    errors = np.linalg.norm(
        projected_pts - ref_pts,
        axis=2
    )

    inlier_errors = errors[
        mask.ravel() == 1
    ]

    mean_error = float(
        np.mean(inlier_errors)
    )

    median_error = float(
        np.median(inlier_errors)
    )

    max_error = float(
        np.max(inlier_errors)
    )

    # Overall validation confidence
    inlier_ratio = (
        int(mask.ravel().sum())
        / len(good_matches)
    ) * 100

    average_coverage = (
        ref_coverage + src_coverage
    ) / 2

    # Confidence score combines:
    # inlier reliability + spatial distribution + geometric accuracy
    geometric_score = max(
        0,
        100 - (mean_error * 20)
    )

    confidence_score = (
        0.45 * inlier_ratio
        + 0.30 * average_coverage
        + 0.25 * geometric_score
    )

    confidence_score = min(
        confidence_score,
        100
    )

    # Print results
    print("\n========== SPATIAL VALIDATION ==========")

    print(
        "Good Matches:",
        len(good_matches)
    )

    print(
        "RANSAC Inliers:",
        int(mask.ravel().sum())
    )

    print(
        f"Inlier Ratio: {inlier_ratio:.2f}%"
    )

    print("\n--- Spatial Coverage ---")

    print(
        f"Reference Coverage: "
        f"{ref_coverage:.2f}%"
    )

    print(
        f"Source Coverage: "
        f"{src_coverage:.2f}%"
    )

    print(
        f"Average Coverage: "
        f"{average_coverage:.2f}%"
    )

    print("\n--- Reprojection Error ---")

    print(
        f"Mean Error: "
        f"{mean_error:.4f} pixels"
    )

    print(
        f"Median Error: "
        f"{median_error:.4f} pixels"
    )

    print(
        f"Maximum Error: "
        f"{max_error:.4f} pixels"
    )

    print("\n--- Alignment Confidence ---")

    print(
        f"Confidence Score: "
        f"{confidence_score:.2f}%"
    )

    if confidence_score >= 85:
        print("Status: HIGH CONFIDENCE")

    elif confidence_score >= 60:
        print("Status: MODERATE CONFIDENCE")

    else:
        print("Status: LOW CONFIDENCE")

    print(
        "========================================\n"
    )


if __name__ == "__main__":
    main()