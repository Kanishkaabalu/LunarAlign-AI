import cv2
import numpy as np
import sys
import os

# Allow imports from the src folder
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from multimodal_matcher import adaptive_multimodal_matching
from scale_invariance import evaluate_scale_invariance


def calculate_confidence(best_result, scale_results):
    """
    Calculate final confidence using:
    - Multi-modal correspondence quality
    - Geometric inlier ratio
    - Consistency across multiple scales
    """

    multimodal_score = best_result["inlier_ratio"]

    scale_ratios = [
        result["inlier_ratio"]
        for result in scale_results
    ]

    scale_consistency = np.mean(scale_ratios)

    final_confidence = (
        0.6 * multimodal_score
        + 0.4 * scale_consistency
    )

    return min(final_confidence, 100.0)


def calculate_scale_stability(scale_results):
    """
    Measure how stable correspondence performance is
    across different image scales.
    """

    ratios = np.array([
        result["inlier_ratio"]
        for result in scale_results
    ])

    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)

    if mean_ratio == 0:
        return 0.0

    stability = 100 * (
        1 - (std_ratio / mean_ratio)
    )

    return max(0.0, stability)


def draw_final_matches(
    reference,
    source,
    best_result
):
    """
    Create visualization of reliable multi-modal
    correspondence points.
    """

    if (
        best_result is None
        or len(best_result["inliers"]) == 0
    ):
        return None

    match_image = cv2.drawMatches(
        reference,
        best_result["kp1"],
        source,
        best_result["kp2"],
        best_result["inliers"],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return match_image


def align_image(
    reference,
    source,
    homography
):
    """
    Warp source image into the reference coordinate system.
    """

    height, width = reference.shape[:2]

    aligned = cv2.warpPerspective(
        source,
        homography,
        (width, height)
    )

    return aligned


def run_sih_pipeline(
    reference_path="data/reference.jpg",
    source_path="data/source.jpg"
):

    print("\n🌕 ===========================================")
    print("       SIH26166 LUNAR CORRESPONDENCE PIPELINE")
    print("=========================================== 🌕\n")

    reference = cv2.imread(reference_path)
    source = cv2.imread(source_path)

    if reference is None or source is None:
        print("❌ Error: Could not load input lunar images.")
        return

    print("✓ Chandrayaan-2 lunar image inputs loaded")

    # ------------------------------------------------
    # STEP 1: ADAPTIVE MULTI-REPRESENTATION MATCHING
    # ------------------------------------------------

    print("\n[1/4] Running adaptive cross-representation matching...")

    best_result, all_results = (
        adaptive_multimodal_matching(
            reference,
            source
        )
    )

    if best_result is None:
        print("❌ No reliable correspondence found.")
        return

    print("✓ Best representation pair selected")

    # ------------------------------------------------
    # STEP 2: SCALE INVARIANCE EVALUATION
    # ------------------------------------------------

    print("\n[2/4] Evaluating scale robustness...")

    gray_reference = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    gray_source = cv2.cvtColor(
        source,
        cv2.COLOR_BGR2GRAY
    )

    scale_results = evaluate_scale_invariance(
        gray_reference,
        gray_source
    )

    print("✓ Multi-scale evaluation completed")

    # ------------------------------------------------
    # STEP 3: FINAL CONFIDENCE + STABILITY
    # ------------------------------------------------

    print("\n[3/4] Computing confidence metrics...")

    confidence = calculate_confidence(
        best_result,
        scale_results
    )

    scale_stability = calculate_scale_stability(
        scale_results
    )

    print("✓ Confidence evaluation completed")

    # ------------------------------------------------
    # STEP 4: ALIGNMENT + VISUALIZATION
    # ------------------------------------------------

    print("\n[4/4] Generating aligned output...")

    aligned = align_image(
        reference,
        source,
        best_result["homography"]
    )

    match_visualization = draw_final_matches(
        reference,
        source,
        best_result
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    cv2.imwrite(
        "results/sih_aligned_image.jpg",
        aligned
    )

    if match_visualization is not None:

        cv2.imwrite(
            "results/sih_multimodal_matches.jpg",
            match_visualization
        )

    # ------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------

    print("\n========== SIH26166 FINAL RESULTS ==========")

    print(
        f"\nSelected Reference Representation: "
        f"{best_result['reference_mode']}"
    )

    print(
        f"Selected Source Representation: "
        f"{best_result['source_mode']}"
    )

    print(
        f"\nInitial Correspondences: "
        f"{best_result['good_matches']}"
    )

    print(
        f"Geometric Inliers: "
        f"{best_result['inlier_count']}"
    )

    print(
        f"Inlier Ratio: "
        f"{best_result['inlier_ratio']:.2f}%"
    )

    print(
        f"\nScale Stability: "
        f"{scale_stability:.2f}%"
    )

    print(
        f"Final Correspondence Confidence: "
        f"{confidence:.2f}%"
    )

    print("\n✓ Aligned image saved:")
    print("  results/sih_aligned_image.jpg")

    print("\n✓ Correspondence visualization saved:")
    print("  results/sih_multimodal_matches.jpg")

    print("\n🌕 SIH26166 PIPELINE COMPLETED SUCCESSFULLY!")
    print("============================================\n")


if __name__ == "__main__":

    run_sih_pipeline()