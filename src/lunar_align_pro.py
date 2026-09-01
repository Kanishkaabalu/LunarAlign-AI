import cv2
import numpy as np


# =========================================================
# 1. ADVANCED PREPROCESSING
# =========================================================

def gamma_correction(image, gamma=1.2):

    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)


def illumination_normalization(image):

    background = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=25
    )

    # Avoid division problems
    background[background == 0] = 1

    normalized = cv2.divide(
        image,
        background,
        scale=255
    )

    return normalized


def preprocess_image(image):

    # Edge-preserving denoising
    denoised = cv2.bilateralFilter(
        image,
        9,
        75,
        75
    )

    # Illumination normalization
    normalized = illumination_normalization(
        denoised
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(normalized)

    # Gamma correction
    final_image = gamma_correction(
        enhanced,
        gamma=1.2
    )

    return final_image


# =========================================================
# 2. FEATURE DETECTORS
# =========================================================

def get_detector(name):

    if name == "SIFT":
        return cv2.SIFT_create(), cv2.NORM_L2

    elif name == "ORB":
        return (
            cv2.ORB_create(nfeatures=4000),
            cv2.NORM_HAMMING
        )

    return None, None


# =========================================================
# 3. ADAPTIVE FEATURE EVALUATION
# =========================================================

def evaluate_detector(
    name,
    reference,
    source
):

    detector, norm_type = get_detector(name)

    if detector is None:
        return None

    kp1, des1 = detector.detectAndCompute(
        reference,
        None
    )

    kp2, des2 = detector.detectAndCompute(
        source,
        None
    )

    if des1 is None or des2 is None:
        return None

    # KNN matching
    matcher = cv2.BFMatcher(norm_type)

    knn_matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    # Lowe's ratio test
    good_matches = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return None

    # Correspondence points
    ref_pts = np.float32([
        kp1[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    src_pts = np.float32([
        kp2[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    # Robust homography estimation
    H, mask = cv2.findHomography(
        src_pts,
        ref_pts,
        cv2.RANSAC,
        4.0
    )

    if H is None or mask is None:
        return None

    inliers = int(mask.ravel().sum())

    if inliers < 4:
        return None

    inlier_ratio = (
        inliers / len(good_matches)
    )

    # Reprojection error
    projected = cv2.perspectiveTransform(
        src_pts,
        H
    )

    errors = np.linalg.norm(
        projected - ref_pts,
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

    # Adaptive quality score
    quality_score = (
        inlier_ratio
        * np.log1p(inliers)
        / (1 + median_error)
    )

    return {

        "name": name,

        "kp1": kp1,
        "kp2": kp2,

        "matches": good_matches,

        "ref_pts": ref_pts,
        "src_pts": src_pts,

        "homography": H,
        "mask": mask,

        "good_matches": len(good_matches),

        "inliers": inliers,

        "inlier_ratio":
            inlier_ratio * 100,

        "mean_error":
            mean_error,

        "median_error":
            median_error,

        "quality_score":
            quality_score
    }


# =========================================================
# 4. SPATIAL COVERAGE
# =========================================================

def calculate_spatial_coverage(
    points,
    image_shape,
    grid_size=4
):

    height, width = image_shape[:2]

    occupied_cells = set()

    cell_width = width / grid_size
    cell_height = height / grid_size

    for point in points:

        x, y = point.ravel()

        col = min(
            int(x / cell_width),
            grid_size - 1
        )

        row = min(
            int(y / cell_height),
            grid_size - 1
        )

        occupied_cells.add(
            (row, col)
        )

    total_cells = grid_size * grid_size

    coverage = (
        len(occupied_cells)
        / total_cells
    ) * 100

    return coverage


# =========================================================
# 5. ROBUST REFINEMENT
# =========================================================

def robust_refine(result):

    H = result["homography"]

    ref_pts = result["ref_pts"]
    src_pts = result["src_pts"]

    mask = result["mask"].ravel()

    # Keep only RANSAC inliers
    inlier_ref = ref_pts[
        mask == 1
    ]

    inlier_src = src_pts[
        mask == 1
    ]

    # Reproject
    projected = cv2.perspectiveTransform(
        inlier_src,
        H
    )

    errors = np.linalg.norm(
        projected - inlier_ref,
        axis=2
    ).ravel()

    # Robust threshold using median + MAD
    median_error = np.median(errors)

    mad = np.median(
        np.abs(errors - median_error)
    )

    threshold = (
        median_error + 2.5 * max(mad, 0.1)
    )

    # Keep geometrically reliable points
    reliable = errors <= threshold

    reliable_ref = inlier_ref[
        reliable
    ]

    reliable_src = inlier_src[
        reliable
    ]

    if len(reliable_ref) < 4:

        return (
            H,
            inlier_ref,
            inlier_src,
            errors
        )

    # Recalculate homography
    H_refined, _ = cv2.findHomography(
        reliable_src,
        reliable_ref,
        0
    )

    if H_refined is None:
        H_refined = H

    # Final reprojection error
    projected_final = cv2.perspectiveTransform(
        reliable_src,
        H_refined
    )

    final_errors = np.linalg.norm(
        projected_final - reliable_ref,
        axis=2
    ).ravel()

    return (
        H_refined,
        reliable_ref,
        reliable_src,
        final_errors
    )


# =========================================================
# 6. MAIN LUNARALIGN-AI PIPELINE
# =========================================================

def run_lunar_align(
    reference_path,
    source_path
):

    print(
        "\n🌕 LUNARALIGN-AI PRO STARTED"
    )

    # Load images
    reference_color = cv2.imread(
        reference_path
    )

    source_color = cv2.imread(
        source_path
    )

    if (
        reference_color is None
        or source_color is None
    ):
        print("Error: Images not found.")
        return None

    # Convert to grayscale
    reference_gray = cv2.cvtColor(
        reference_color,
        cv2.COLOR_BGR2GRAY
    )

    source_gray = cv2.cvtColor(
        source_color,
        cv2.COLOR_BGR2GRAY
    )

    print(
        "✓ Images loaded"
    )

    # Preprocessing
    reference = preprocess_image(
        reference_gray
    )

    source = preprocess_image(
        source_gray
    )

    print(
        "✓ Illumination-aware preprocessing completed"
    )

    # Evaluate feature methods
    methods = [
        "SIFT",
        "ORB"
    ]

    results = []

    for method in methods:

        result = evaluate_detector(
            method,
            reference,
            source
        )

        if result is not None:

            results.append(result)

            print(
                f"\n{method} Evaluation:"
            )

            print(
                f"  Good Matches: "
                f"{result['good_matches']}"
            )

            print(
                f"  Inliers: "
                f"{result['inliers']}"
            )

            print(
                f"  Inlier Ratio: "
                f"{result['inlier_ratio']:.2f}%"
            )

            print(
                f"  Median Error: "
                f"{result['median_error']:.4f}px"
            )

            print(
                f"  Quality Score: "
                f"{result['quality_score']:.4f}"
            )

    if not results:

        print(
            "Error: No reliable detector found."
        )

        return None

    # Select best detector
    best = max(
        results,
        key=lambda x: x["quality_score"]
    )

    print(
        f"\n🏆 Selected Detector: "
        f"{best['name']}"
    )

    # Robust refinement
    (
        H_final,
        reliable_ref,
        reliable_src,
        final_errors
    ) = robust_refine(best)

    print(
        "✓ Robust geometric refinement completed"
    )

    # Spatial coverage
    coverage = calculate_spatial_coverage(
        reliable_ref,
        reference.shape,
        grid_size=4
    )

    # Final metrics
    mean_error = float(
        np.mean(final_errors)
    )

    median_error = float(
        np.median(final_errors)
    )

    max_error = float(
        np.max(final_errors)
    )

    reliable_points = len(
        reliable_ref
    )

    # Alignment confidence
    geometric_score = max(
        0,
        100 - mean_error * 20
    )

    confidence = (
        0.40 * best["inlier_ratio"]
        + 0.30 * coverage
        + 0.30 * geometric_score
    )

    confidence = min(
        confidence,
        100
    )

    # Warp source image
    aligned = cv2.warpPerspective(
        source_color,
        H_final,
        (
            reference_color.shape[1],
            reference_color.shape[0]
        )
    )

    # Save results
    cv2.imwrite(
        "results/pro_aligned_image.jpg",
        aligned
    )

    np.save(
        "results/pro_homography.npy",
        H_final
    )

    print(
        "\n========== FINAL RESULTS =========="
    )

    print(
        f"Selected Detector: "
        f"{best['name']}"
    )

    print(
        f"Initial Good Matches: "
        f"{best['good_matches']}"
    )

    print(
        f"RANSAC Inliers: "
        f"{best['inliers']}"
    )

    print(
        f"Reliable Distributed Points: "
        f"{reliable_points}"
    )

    print(
        f"Spatial Coverage: "
        f"{coverage:.2f}%"
    )

    print(
        f"Mean Reprojection Error: "
        f"{mean_error:.4f} pixels"
    )

    print(
        f"Median Reprojection Error: "
        f"{median_error:.4f} pixels"
    )

    print(
        f"Maximum Reprojection Error: "
        f"{max_error:.4f} pixels"
    )

    print(
        f"Alignment Confidence: "
        f"{confidence:.2f}%"
    )

    print(
        "===================================\n"
    )

    # Return everything needed by Streamlit
    return {

        "reference": reference_color,

        "source": source_color,

        "aligned": aligned,

        "detector":
            best["name"],

        "good_matches":
            best["good_matches"],

        "inliers":
            best["inliers"],

        "inlier_ratio":
            best["inlier_ratio"],

        "reliable_points":
            reliable_points,

        "coverage":
            coverage,

        "mean_error":
            mean_error,

        "median_error":
            median_error,

        "max_error":
            max_error,

        "confidence":
            confidence,

        "homography":
            H_final,

        "result":
            best
    }


# =========================================================
# RUN FROM TERMINAL
# =========================================================

if __name__ == "__main__":

    output = run_lunar_align(
        "data/reference.jpg",
        "data/source.jpg"
    )

    if output is not None:

        print(
            "🌕 LunarAlign-AI Pro completed successfully!"
        )