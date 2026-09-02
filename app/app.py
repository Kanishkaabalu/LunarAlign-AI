import streamlit as st
import cv2
import numpy as np
import sys
import os

# Add src folder to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src"
        )
    )
)

from multimodal_matcher import adaptive_multimodal_matching
from scale_invariance import evaluate_scale_invariance


st.set_page_config(
    page_title="LunarAlign-AI | SIH26166",
    page_icon="🌕",
    layout="wide"
)


st.title("🌕 LunarAlign-AI")
st.subheader(
    "SIH26166 — Multi-modal, Sun-Angle and Scale-Invariant "
    "Lunar Image Correspondence"
)

st.write(
    "An adaptive computer-vision pipeline for robust lunar image "
    "correspondence, geometric verification and multi-scale evaluation."
)

st.divider()


def load_image(uploaded_file):
    """Convert Streamlit uploaded file into OpenCV image."""

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    return image


def resize_for_display(image, max_width=700):

    height, width = image.shape[:2]

    if width > max_width:

        scale = max_width / width

        image = cv2.resize(
            image,
            (
                int(width * scale),
                int(height * scale)
            )
        )

    return image


st.header("📤 Upload Lunar Images")

col1, col2 = st.columns(2)

with col1:

    reference_file = st.file_uploader(
        "Upload Reference Image",
        type=["jpg", "jpeg", "png"],
        key="reference"
    )

with col2:

    source_file = st.file_uploader(
        "Upload Source Image",
        type=["jpg", "jpeg", "png"],
        key="source"
    )


if reference_file and source_file:

    reference = load_image(reference_file)
    source = load_image(source_file)

    st.success("✓ Both lunar images loaded successfully")

    preview1, preview2 = st.columns(2)

    with preview1:

        st.image(
            resize_for_display(reference),
            caption="Reference Lunar Image",
            channels="BGR"
        )

    with preview2:

        st.image(
            resize_for_display(source),
            caption="Source Lunar Image",
            channels="BGR"
        )

    st.divider()

    if st.button(
        "🚀 Run SIH26166 Correspondence Pipeline",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Running adaptive multi-modal correspondence..."
            ):

                # STEP 1:
                # Adaptive representation matching

                best_result, all_results = (
                    adaptive_multimodal_matching(
                        reference,
                        source
                    )
                )

                if best_result is None:

                    st.error(
                        "No reliable correspondence could be found."
                    )

                    st.stop()

                # STEP 2:
                # Scale invariance evaluation

                gray_reference = cv2.cvtColor(
                    reference,
                    cv2.COLOR_BGR2GRAY
                )

                gray_source = cv2.cvtColor(
                    source,
                    cv2.COLOR_BGR2GRAY
                )

                scale_results = (
                    evaluate_scale_invariance(
                        gray_reference,
                        gray_source
                    )
                )

                # STEP 3:
                # Calculate metrics

                scale_ratios = [
                    r["inlier_ratio"]
                    for r in scale_results
                ]

                scale_consistency = np.mean(
                    scale_ratios
                )

                ratios_array = np.array(
                    scale_ratios
                )

                mean_ratio = np.mean(
                    ratios_array
                )

                std_ratio = np.std(
                    ratios_array
                )

                if mean_ratio > 0:

                    scale_stability = (
                        100 *
                        (
                            1 -
                            std_ratio / mean_ratio
                        )
                    )

                else:

                    scale_stability = 0

                scale_stability = max(
                    0,
                    scale_stability
                )

                final_confidence = (
                    0.6 *
                    best_result["inlier_ratio"]
                    +
                    0.4 *
                    scale_consistency
                )

                final_confidence = min(
                    final_confidence,
                    100
                )

                # STEP 4:
                # Align source image

                height, width = reference.shape[:2]

                aligned_image = (
                    cv2.warpPerspective(
                        source,
                        best_result["homography"],
                        (width, height)
                    )
                )

                # STEP 5:
                # Draw reliable matches

                match_image = cv2.drawMatches(
                    reference,
                    best_result["kp1"],
                    source,
                    best_result["kp2"],
                    best_result["inliers"],
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

            st.success(
                "🌕 SIH26166 Correspondence Pipeline Completed!"
            )

            # -----------------------------------------
            # METRICS
            # -----------------------------------------

            st.header("📊 Correspondence Results")

            metric1, metric2, metric3, metric4 = (
                st.columns(4)
            )

            metric1.metric(
                "Good Matches",
                best_result["good_matches"]
            )

            metric2.metric(
                "Geometric Inliers",
                best_result["inlier_count"]
            )

            metric3.metric(
                "Inlier Ratio",
                f"{best_result['inlier_ratio']:.2f}%"
            )

            metric4.metric(
                "Final Confidence",
                f"{final_confidence:.2f}%"
            )

            st.divider()

            # -----------------------------------------
            # ADAPTIVE REPRESENTATION
            # -----------------------------------------

            st.header(
                "🧠 Adaptive Representation Selection"
            )

            rep1, rep2 = st.columns(2)

            with rep1:

                st.info(
                    "Reference Representation: "
                    f"**{best_result['reference_mode']}**"
                )

            with rep2:

                st.info(
                    "Source Representation: "
                    f"**{best_result['source_mode']}**"
                )

            st.write(
                "The system evaluates intensity, structural and "
                "edge-based representations and selects the "
                "correspondence pair with the strongest geometric "
                "consistency."
            )

            # -----------------------------------------
            # SCALE RESULTS
            # -----------------------------------------

            st.divider()

            st.header("🔍 Multi-Scale Robustness Evaluation")

            scale_data = []

            for result in scale_results:

                scale_data.append(
                    {
                        "Scale": f"{result['scale']}x",
                        "Good Matches":
                            result["good_matches"],
                        "Geometric Inliers":
                            result["inliers"],
                        "Inlier Ratio (%)":
                            round(
                                result[
                                    "inlier_ratio"
                                ],
                                2
                            )
                    }
                )

            st.dataframe(
                scale_data,
                use_container_width=True
            )

            stability_col1, stability_col2 = (
                st.columns(2)
            )

            with stability_col1:

                st.metric(
                    "Scale Stability",
                    f"{scale_stability:.2f}%"
                )

            with stability_col2:

                st.metric(
                    "Scales Evaluated",
                    len(scale_results)
                )

            # -----------------------------------------
            # VISUAL OUTPUT
            # -----------------------------------------

            st.divider()

            st.header("🛰️ Geometric Correspondence")

            st.image(
                resize_for_display(
                    match_image,
                    1200
                ),
                caption=(
                    "RANSAC-Verified Reliable "
                    "Correspondences"
                ),
                channels="BGR"
            )

            st.divider()

            st.header("🌕 Final Aligned Lunar Image")

            st.image(
                resize_for_display(
                    aligned_image
                ),
                caption=(
                    "Source Image Aligned to "
                    "Reference Coordinate System"
                ),
                channels="BGR"
            )

            # -----------------------------------------
            # PROJECT SUMMARY
            # -----------------------------------------

            st.divider()

            st.header("🔬 SIH26166 Pipeline Summary")

            st.markdown(
                """
                **Pipeline capabilities:**

                - 🌗 Illumination-aware preprocessing
                - 🛰️ Multiple structural image representations
                - 🧠 Adaptive representation selection
                - 🔍 Explicit multi-scale correspondence evaluation
                - 📍 SIFT-based feature correspondence
                - 🛡️ RANSAC geometric verification
                - 📐 Homography-based image alignment
                - 📊 Quantitative confidence and stability metrics

                **Note:** Performance metrics are calculated for the
                currently uploaded image pair and should be interpreted
                as experimental results for that evaluation.
                """
            )

        except Exception as error:

            st.error(
                f"Pipeline Error: {error}"
            )

else:

    st.info(
        "👆 Upload both a reference and source lunar image "
        "to start the SIH26166 correspondence pipeline."
    )