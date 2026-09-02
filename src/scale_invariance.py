import cv2
import numpy as np

from multimodal_matcher import (
    extract_features,
    match_features,
    geometric_verification
)


def resize_image(image, scale):
    """
    Resize an image according to the specified scale.
    """

    height, width = image.shape[:2]

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_CUBIC
    )


def evaluate_scale(
    reference_image,
    source_image,
    scale
):
    """
    Evaluate correspondence performance at one scale.
    """

    scaled_source = resize_image(
        source_image,
        scale
    )

    kp1, desc1 = extract_features(
        reference_image
    )

    kp2, desc2 = extract_features(
        scaled_source
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

    good_matches = len(matches)
    inlier_count = len(inliers)

    if good_matches > 0:

        inlier_ratio = (
            inlier_count /
            good_matches
        ) * 100

    else:

        inlier_ratio = 0

    return {
        "scale": scale,
        "good_matches": good_matches,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "homography": H
    }


def evaluate_scale_invariance(
    reference_image,
    source_image
):
    """
    Evaluate feature correspondence across
    multiple source-image scales.
    """

    scales = [
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0
    ]

    results = []

    for scale in scales:

        print(
            f"Evaluating scale: {scale}x"
        )

        result = evaluate_scale(
            reference_image,
            source_image,
            scale
        )

        results.append(result)

    return results


if __name__ == "__main__":

    reference = cv2.imread(
        "data/reference.jpg",
        cv2.IMREAD_GRAYSCALE
    )

    source = cv2.imread(
        "data/source.jpg",
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None or source is None:

        print(
            "Error: Could not load lunar images."
        )

    else:

        results = evaluate_scale_invariance(
            reference,
            source
        )

        print(
            "\n========== SCALE INVARIANCE "
            "EVALUATION =========="
        )

        for result in results:

            print(
                f"\nScale: {result['scale']}x"
            )

            print(
                f"Good Matches: "
                f"{result['good_matches']}"
            )

            print(
                f"Geometric Inliers: "
                f"{result['inliers']}"
            )

            print(
                f"Inlier Ratio: "
                f"{result['inlier_ratio']:.2f}%"
            )

        print(
            "\n===================================="
        )