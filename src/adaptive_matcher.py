import cv2
import numpy as np


def get_detector(name):

    if name == "SIFT":
        return cv2.SIFT_create(), cv2.NORM_L2

    elif name == "AKAZE":

        # Check whether AKAZE is available
        if hasattr(cv2, "AKAZE_create"):
            return cv2.AKAZE_create(), cv2.NORM_HAMMING

        else:
            print("AKAZE is not available. Skipping AKAZE.")
            return None, None

    elif name == "ORB":
        return (
            cv2.ORB_create(nfeatures=3000),
            cv2.NORM_HAMMING
        )

    return None, None


def evaluate_detector(name, reference, source):

    # Get feature detector
    detector, norm_type = get_detector(name)

    # Skip detector if unavailable
    if detector is None:
        return None

    # Detect keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(
        reference,
        None
    )

    kp2, des2 = detector.detectAndCompute(
        source,
        None
    )

    # Check descriptors
    if des1 is None or des2 is None:
        print(f"{name}: Not enough descriptors.")
        return None

    # Create matcher
    matcher = cv2.BFMatcher(norm_type)

    # KNN matching
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

    # Need minimum 4 points for homography
    if len(good_matches) < 4:

        print(
            f"{name}: Not enough good matches "
            "for homography."
        )

        return None

    # Extract reference points
    ref_pts = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    # Extract source points
    src_pts = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    # Robust geometric verification
    H, mask = cv2.findHomography(
        src_pts,
        ref_pts,
        cv2.RANSAC,
        4.0
    )

    # Check homography
    if H is None or mask is None:

        print(
            f"{name}: Homography estimation failed."
        )

        return None

    # Count RANSAC inliers
    inliers = int(mask.ravel().sum())

    # Calculate inlier ratio
    inlier_ratio = (
        inliers / len(good_matches)
    )

    # Project source points using homography
    projected_pts = cv2.perspectiveTransform(
        src_pts,
        H
    )

    # Calculate reprojection errors
    errors = np.linalg.norm(
        projected_pts - ref_pts,
        axis=2
    )

    # Keep only inlier errors
    inlier_errors = errors[
        mask.ravel() == 1
    ]

    # Mean reprojection error
    if len(inlier_errors) > 0:

        reprojection_error = float(
            np.mean(inlier_errors)
        )

    else:

        reprojection_error = float("inf")

    # Adaptive quality score
    # Higher inlier ratio = better
    # More inliers = better
    # Lower reprojection error = better
    quality_score = (
        inlier_ratio
        * np.log1p(inliers)
        / (1 + reprojection_error)
    )

    return {

        "name": name,

        "keypoints_reference": len(kp1),

        "keypoints_source": len(kp2),

        "good_matches": len(good_matches),

        "inliers": inliers,

        "inlier_ratio": inlier_ratio * 100,

        "reprojection_error":
            reprojection_error,

        "quality_score":
            quality_score,

        "homography": H,

        "mask": mask,

        "matches": good_matches,

        "kp1": kp1,

        "kp2": kp2
    }


if __name__ == "__main__":

    # Load advanced preprocessed images
    reference = cv2.imread(
        "results/advanced_reference.jpg",
        cv2.IMREAD_GRAYSCALE
    )

    source = cv2.imread(
        "results/advanced_source.jpg",
        cv2.IMREAD_GRAYSCALE
    )

    # Check images
    if reference is None or source is None:

        print(
            "Error: Advanced preprocessed "
            "images not found."
        )

        exit()

    # Feature methods to compare
    methods = [
        "SIFT",
        "AKAZE",
        "ORB"
    ]

    results = []

    print(
        "\n========== ADAPTIVE FEATURE "
        "EVALUATION ==========\n"
    )

    # Evaluate each detector
    for method in methods:

        result = evaluate_detector(
            method,
            reference,
            source
        )

        if result is not None:

            results.append(result)

            print(
                f"Method: {result['name']}"
            )

            print(
                "Reference Keypoints:",
                result[
                    "keypoints_reference"
                ]
            )

            print(
                "Source Keypoints:",
                result[
                    "keypoints_source"
                ]
            )

            print(
                "Good Matches:",
                result[
                    "good_matches"
                ]
            )

            print(
                "RANSAC Inliers:",
                result[
                    "inliers"
                ]
            )

            print(
                f"Inlier Ratio: "
                f"{result['inlier_ratio']:.2f}%"
            )

            print(
                f"Reprojection Error: "
                f"{result['reprojection_error']:.4f} "
                "pixels"
            )

            print(
                f"Quality Score: "
                f"{result['quality_score']:.4f}"
            )

            print(
                "-" * 45
            )

    # Check whether any detector succeeded
    if not results:

        print(
            "\nNo detector produced enough "
            "reliable matches."
        )

        exit()

    # Automatically select best detector
    best_result = max(
        results,
        key=lambda x: x["quality_score"]
    )

    print(
        "\n========== SELECTED DETECTOR =========="
    )

    print(
        "Best Feature Method:",
        best_result["name"]
    )

    print(
        f"Best Quality Score: "
        f"{best_result['quality_score']:.4f}"
    )

    print(
        "=======================================\n"
    )