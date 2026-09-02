import cv2
import numpy as np


def normalize_illumination(image):
    """
    Reduces global illumination differences and enhances
    local lunar surface structures.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Local contrast normalization
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    normalized = clahe.apply(gray)

    # Remove slow illumination variations
    background = cv2.GaussianBlur(
        normalized,
        (0, 0),
        sigmaX=25,
        sigmaY=25
    )

    illumination_corrected = cv2.divide(
        normalized,
        background,
        scale=128
    )

    return illumination_corrected


def structural_representation(image):
    """
    Creates a representation based mainly on surface structure
    rather than absolute brightness.
    """

    image = normalize_illumination(image)

    blurred = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    grad_x = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        grad_x,
        grad_y
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude.astype(np.uint8)


def edge_representation(image):
    """
    Produces an edge-based representation useful when
    two input modalities have different intensity characteristics.
    """

    image = normalize_illumination(image)

    edges = cv2.Canny(
        image,
        50,
        150
    )

    return edges


def create_multimodal_views(image):
    """
    Returns multiple complementary representations of the same input.
    These can be used for cross-representation correspondence.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.copy()

    illumination = normalize_illumination(gray)

    structure = structural_representation(gray)

    edges = edge_representation(gray)

    return {
        "intensity": illumination,
        "structure": structure,
        "edges": edges
    }