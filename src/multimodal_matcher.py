import cv2
import numpy as np

from multimodal_preprocess import create_multimodal_views


def extract_features(image):
    """
    Extract robust SIFT features from a structural
    representation of the lunar surface.
    """

    sift = cv2.SIFT_create(
        nfeatures=5000
    )

    keypoints, descriptors = sift.detectAndCompute(
        image,
        None
    )

    return keypoints, descriptors


def match_features(desc1, desc2):
    """
    Match descriptors using FLANN and Lowe's ratio test.
    """

    if desc1 is None or desc2 is None:
        return []

    FLANN_INDEX_KDTREE = 1

    index_params = dict(
        algorithm=FLANN_INDEX_KDTREE,
        trees=5
    )

    search_params = dict(
        checks=50
    )

    matcher = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    matches = matcher.knnMatch(
        desc1,
        desc2,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches


def geometric_verification(
    kp1,
    kp2,
    matches
):
    """
    Verify correspondences using RANSAC homography.
    """

    if len(matches) < 4:
        return None, [], None

    points1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    points2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        points2,
        points1,
        cv2.RANSAC,
        3.0
    )

    if mask is None:
        return H, [], mask

    inlier_matches = [
        matches[i]
        for i in range(len(matches))
        if mask[i][0] == 1
    ]

    return H, inlier_matches, mask


def evaluate_modality_pair(
    reference_image,
    source_image,
    reference_mode,
    source_mode
):
    """
    Evaluate correspondence between two different
    image representations.
    """

    reference_views = create_multimodal_views(
        reference_image
    )

    source_views = create_multimodal_views(
        source_image
    )

    ref_view = reference_views[
        reference_mode
    ]

    src_view = source_views[
        source_mode
    ]

    kp1, desc1 = extract_features(
        ref_view
    )

    kp2, desc2 = extract_features(
        src_view
    )

    matches = match_features(
        desc1,
        desc2
    )

    H, inliers, mask = geometric_verification(
        kp1,
        kp2,
        matches
    )

    inlier_count = len(inliers)

    if len(matches) > 0:
        inlier_ratio = (
            inlier_count /
            len(matches)
        ) * 100
    else:
        inlier_ratio = 0

    return {
        "reference_mode": reference_mode,
        "source_mode": source_mode,
        "kp1": kp1,
        "kp2": kp2,
        "matches": matches,
        "inliers": inliers,
        "homography": H,
        "mask": mask,
        "good_matches": len(matches),
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio
    }


def adaptive_multimodal_matching(
    reference_image,
    source_image
):
    """
    Automatically evaluate correspondence across
    intensity, structure and edge representations.
    """

    modes = [
        "intensity",
        "structure",
        "edges"
    ]

    results = []

    for reference_mode in modes:

        for source_mode in modes:

            result = evaluate_modality_pair(
                reference_image,
                source_image,
                reference_mode,
                source_mode
            )

            results.append(result)

    valid_results = [
        r for r in results
        if r["homography"] is not None
        and r["inlier_count"] >= 4
    ]

    if not valid_results:
        return None, results

    # Adaptive quality score
    for result in valid_results:

        result["quality_score"] = (
            0.6 * result["inlier_ratio"]
            + 0.4 * min(
                result["inlier_count"] / 50 * 100,
                100
            )
        )

    best_result = max(
        valid_results,
        key=lambda x: x["quality_score"]
    )

    return best_result, results


if __name__ == "__main__":

    reference = cv2.imread(
        "data/reference.jpg"
    )

    source = cv2.imread(
        "data/source.jpg"
    )

    if reference is None or source is None:

        print(
            "Error: Could not load lunar images."
        )

    else:

        best, all_results = (
            adaptive_multimodal_matching(
                reference,
                source
            )
        )

        print(
            "\n========== MULTI-MODAL "
            "CORRESPONDENCE =========="
        )

        if best is not None:

            print(
                f"Best Reference Representation: "
                f"{best['reference_mode']}"
            )

            print(
                f"Best Source Representation: "
                f"{best['source_mode']}"
            )

            print(
                f"Good Matches: "
                f"{best['good_matches']}"
            )

            print(
                f"Geometric Inliers: "
                f"{best['inlier_count']}"
            )

            print(
                f"Inlier Ratio: "
                f"{best['inlier_ratio']:.2f}%"
            )

            print(
                f"Adaptive Quality Score: "
                f"{best['quality_score']:.2f}"
            )

        else:

            print(
                "No reliable multi-modal "
                "correspondence found."
            )

        print(
            "===================================="
        )