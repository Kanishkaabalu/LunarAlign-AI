import cv2
import numpy as np


def refine_points(image, points):
    """
    Refine feature point locations using cornerSubPix.
    """

    # cornerSubPix expects float32 coordinates
    points = np.asarray(
        points,
        dtype=np.float32
    ).reshape(-1, 1, 2)

    # Termination criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001
    )

    refined = cv2.cornerSubPix(
        image,
        points,
        (5, 5),
        (-1, -1),
        criteria
    )

    return refined


def main():

    # Load preprocessed images
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

    # Detect features using SIFT
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

    # Original correspondence points
    ref_pts = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    src_pts = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in good_matches
        ]
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

    # Keep only reliable inliers
    inlier_ref = ref_pts[
        mask.ravel() == 1
    ]

    inlier_src = src_pts[
        mask.ravel() == 1
    ]

    print(
        "\n========== SUB-PIXEL REFINEMENT ==========\n"
    )

    print(
        "Reliable correspondence points:",
        len(inlier_ref)
    )

    # Refine correspondence coordinates
    refined_ref = refine_points(
        reference,
        inlier_ref
    )

    refined_src = refine_points(
        source,
        inlier_src
    )

    # Calculate how much each point moved
    ref_shift = np.linalg.norm(
        refined_ref - inlier_ref,
        axis=2
    )

    src_shift = np.linalg.norm(
        refined_src - inlier_src,
        axis=2
    )

    mean_ref_shift = float(np.mean(ref_shift))
    mean_src_shift = float(np.mean(src_shift))

    # Calculate refined homography
    H_refined, refined_mask = cv2.findHomography(
        refined_src,
        refined_ref,
        cv2.RANSAC,
        3.0
    )

    if H_refined is None:
        print("Refined homography estimation failed.")
        return

    # Calculate reprojection error after refinement
    projected_pts = cv2.perspectiveTransform(
        refined_src,
        H_refined
    )

    errors = np.linalg.norm(
        projected_pts - refined_ref,
        axis=2
    )

    mean_error = float(np.mean(errors))
    median_error = float(np.median(errors))

    print(
        f"Mean Reference Point Refinement: "
        f"{mean_ref_shift:.4f} pixels"
    )

    print(
        f"Mean Source Point Refinement: "
        f"{mean_src_shift:.4f} pixels"
    )

    print(
        f"Mean Reprojection Error After Refinement: "
        f"{mean_error:.4f} pixels"
    )

    print(
        f"Median Reprojection Error After Refinement: "
        f"{median_error:.4f} pixels"
    )

    if mean_error < 1.0:
        print(
            "\nStatus: SUB-PIXEL LEVEL REPROJECTION "
            "ERROR ACHIEVED"
        )
    else:
        print(
            "\nStatus: Refinement completed. "
            "Error is above 1 pixel."
        )

    # Save refined points for later use
    np.save(
        "results/refined_reference_points.npy",
        refined_ref
    )

    np.save(
        "results/refined_source_points.npy",
        refined_src
    )

    np.save(
        "results/refined_homography.npy",
        H_refined
    )

    print(
        "\nRefined correspondence data saved "
        "in the results folder."
    )

    print(
        "==========================================\n"
    )


if __name__ == "__main__":
    main()