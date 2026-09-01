import cv2
import numpy as np


def gamma_correction(image, gamma=1.2):
    """
    Correct illumination differences using gamma transformation.
    """
    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)


def illumination_normalization(image):
    """
    Normalize uneven illumination using background estimation.
    """
    background = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=25
    )

    normalized = cv2.divide(
        image,
        background,
        scale=255
    )

    return normalized


def advanced_preprocess(image_path):
    """
    Advanced lunar image preprocessing pipeline.
    """

    # Load image
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print(f"Error: Could not load {image_path}")
        return None, None

    # 1. Edge-preserving denoising
    denoised = cv2.bilateralFilter(
        image,
        9,
        75,
        75
    )

    # 2. Illumination normalization
    normalized = illumination_normalization(
        denoised
    )

    # 3. CLAHE local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(normalized)

    # 4. Gamma correction
    final_image = gamma_correction(
        enhanced,
        gamma=1.2
    )

    # 5. Multi-scale image pyramid
    pyramid = [
        final_image,
        cv2.pyrDown(final_image),
        cv2.pyrDown(cv2.pyrDown(final_image))
    ]

    return final_image, pyramid


if __name__ == "__main__":

    # Process reference image
    reference, ref_pyramid = advanced_preprocess(
        "data/reference.jpg"
    )

    # Process source image
    source, src_pyramid = advanced_preprocess(
        "data/source.jpg"
    )

    if reference is not None and source is not None:

        # Save preprocessed images
        cv2.imwrite(
            "results/advanced_reference.jpg",
            reference
        )

        cv2.imwrite(
            "results/advanced_source.jpg",
            source
        )

        # Save multi-scale versions
        for i, img in enumerate(ref_pyramid):
            cv2.imwrite(
                f"results/reference_scale_{i}.jpg",
                img
            )

        for i, img in enumerate(src_pyramid):
            cv2.imwrite(
                f"results/source_scale_{i}.jpg",
                img
            )

        print("Advanced preprocessing completed successfully!")
        print("CLAHE: Applied")
        print("Illumination normalization: Applied")
        print("Gamma correction: Applied")
        print("Edge-preserving denoising: Applied")
        print("Multi-scale pyramid levels: 3")
        print("Results saved in the results folder.")