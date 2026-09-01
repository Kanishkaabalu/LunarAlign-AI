import streamlit as st
import cv2
import numpy as np
import os
import sys

# Allow Streamlit to access the src folder
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from lunar_align_pro import (
    preprocess_image,
    evaluate_detector,
    robust_refine,
    calculate_spatial_coverage
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LunarAlign-AI Pro",
    page_icon="🌕",
    layout="wide"
)

st.title("🌕 LunarAlign-AI Pro")
st.subheader(
    "Adaptive Multi-Feature Lunar Image Correspondence "
    "and Registration System"
)

st.markdown("""
**Advanced Pipeline:**

Illumination-Aware Preprocessing → Adaptive Feature Selection →
Lowe's Ratio Test → RANSAC Verification → Robust Geometric Refinement →
Spatial Coverage Analysis → Sub-Pixel Error Evaluation → Image Registration
""")

st.divider()


# =========================================================
# IMAGE UPLOAD
# =========================================================

col1, col2 = st.columns(2)

with col1:
    reference_file = st.file_uploader(
        "🌕 Upload Reference Lunar Image",
        type=["jpg", "jpeg", "png"],
        key="reference"
    )

with col2:
    source_file = st.file_uploader(
        "🛰️ Upload Source Lunar Image",
        type=["jpg", "jpeg", "png"],
        key="source"
    )


def read_uploaded_image(uploaded_file):

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    return cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

if reference_file and source_file:

    reference_color = read_uploaded_image(
        reference_file
    )

    source_color = read_uploaded_image(
        source_file
    )

    # Show uploaded images
    st.subheader("📥 Input Lunar Images")

    c1, c2 = st.columns(2)

    with c1:
        st.image(
            cv2.cvtColor(
                reference_color,
                cv2.COLOR_BGR2RGB
            ),
            caption="Reference Image (Fixed)",
            use_container_width=True
        )

    with c2:
        st.image(
            cv2.cvtColor(
                source_color,
                cv2.COLOR_BGR2RGB
            ),
            caption="Source Image (Moving)",
            use_container_width=True
        )

    st.divider()

    if st.button(
        "🚀 Run LunarAlign-AI Pro",
        use_container_width=True
    ):

        with st.spinner(
            "Running advanced lunar correspondence pipeline..."
        ):

            # ---------------------------------------------
            # Convert to grayscale
            # ---------------------------------------------

            reference_gray = cv2.cvtColor(
                reference_color,
                cv2.COLOR_BGR2GRAY
            )

            source_gray = cv2.cvtColor(
                source_color,
                cv2.COLOR_BGR2GRAY
            )

            # ---------------------------------------------
            # Advanced preprocessing
            # ---------------------------------------------

            reference = preprocess_image(
                reference_gray
            )

            source = preprocess_image(
                source_gray
            )

            # ---------------------------------------------
            # Adaptive detector evaluation
            # ---------------------------------------------

            methods = ["SIFT", "ORB"]

            detector_results = []

            for method in methods:

                result = evaluate_detector(
                    method,
                    reference,
                    source
                )

                if result is not None:
                    detector_results.append(
                        result
                    )

            if not detector_results:

                st.error(
                    "No reliable feature detector found."
                )

                st.stop()

            # Select best detector
            best = max(
                detector_results,
                key=lambda x: x["quality_score"]
            )

            # ---------------------------------------------
            # Robust refinement
            # ---------------------------------------------

            (
                H_final,
                reliable_ref,
                reliable_src,
                final_errors
            ) = robust_refine(best)

            # ---------------------------------------------
            # Spatial coverage
            # ---------------------------------------------

            coverage = calculate_spatial_coverage(
                reliable_ref,
                reference.shape,
                grid_size=4
            )

            # ---------------------------------------------
            # Final metrics
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Final registration
            # ---------------------------------------------

            aligned = cv2.warpPerspective(
                source_color,
                H_final,
                (
                    reference_color.shape[1],
                    reference_color.shape[0]
                )
            )

            # ---------------------------------------------
            # Draw reliable correspondence points
            # ---------------------------------------------

            inlier_mask = best["mask"].ravel() == 1

            inlier_matches = [
                best["matches"][i]
                for i in range(
                    len(best["matches"])
                )
                if inlier_mask[i]
            ]

            # Limit visualization for cleaner display
            display_matches = inlier_matches[:100]

            match_result = cv2.drawMatches(
                reference_color,
                best["kp1"],
                source_color,
                best["kp2"],
                display_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

        # =================================================
        # DISPLAY RESULTS
        # =================================================

        st.success(
            "🎉 LunarAlign-AI Pro completed successfully!"
        )

        st.divider()

        # ---------------------------------------------
        # Detector Comparison
        # ---------------------------------------------

        st.subheader("🧠 Adaptive Feature Selection")

        detector_data = []

        for result in detector_results:

            detector_data.append({
                "Detector": result["name"],
                "Good Matches": result["good_matches"],
                "RANSAC Inliers": result["inliers"],
                "Inlier Ratio (%)":
                    round(result["inlier_ratio"], 2),
                "Median Error (px)":
                    round(result["median_error"], 4),
                "Quality Score":
                    round(result["quality_score"], 4)
            })

        st.dataframe(
            detector_data,
            use_container_width=True
        )

        st.success(
            f"🏆 Automatically Selected Detector: "
            f"{best['name']}"
        )

        st.divider()

        # ---------------------------------------------
        # Final Metrics
        # ---------------------------------------------

        st.subheader("📊 Final Registration Performance")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Good Matches",
            best["good_matches"]
        )

        m2.metric(
            "RANSAC Inliers",
            best["inliers"]
        )

        m3.metric(
            "Reliable Points",
            reliable_points
        )

        m4.metric(
            "Spatial Coverage",
            f"{coverage:.1f}%"
        )

        m5, m6, m7, m8 = st.columns(4)

        m5.metric(
            "Mean Error",
            f"{mean_error:.4f} px"
        )

        m6.metric(
            "Median Error",
            f"{median_error:.4f} px"
        )

        m7.metric(
            "Maximum Error",
            f"{max_error:.4f} px"
        )

        m8.metric(
            "Alignment Confidence",
            f"{confidence:.2f}%"
        )

        st.divider()

        # ---------------------------------------------
        # Correspondence Visualization
        # ---------------------------------------------

        st.subheader(
            "🔗 Geometrically Verified Correspondence Points"
        )

        st.image(
            cv2.cvtColor(
                match_result,
                cv2.COLOR_BGR2RGB
            ),
            caption=(
                "Top reliable feature correspondences "
                "after Lowe's Ratio Test and RANSAC"
            ),
            use_container_width=True
        )

        st.divider()

        # ---------------------------------------------
        # Final Alignment
        # ---------------------------------------------

        st.subheader(
            "🌕 Final Lunar Image Registration"
        )

        a1, a2, a3 = st.columns(3)

        with a1:

            st.image(
                cv2.cvtColor(
                    reference_color,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Reference Image",
                use_container_width=True
            )

        with a2:

            st.image(
                cv2.cvtColor(
                    source_color,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Source Image",
                use_container_width=True
            )

        with a3:

            st.image(
                cv2.cvtColor(
                    aligned,
                    cv2.COLOR_BGR2RGB
                ),
                caption="LunarAlign-AI Pro Registered Image",
                use_container_width=True
            )

        st.divider()

        # ---------------------------------------------
        # Final Summary
        # ---------------------------------------------

        st.subheader("🚀 Alignment Summary")

        st.info(
            f"""
**Selected Feature Engine:** {best["name"]}

**Reliable Correspondence Points:** {reliable_points}

**Spatial Distribution Coverage:** {coverage:.2f}%

**Mean Reprojection Error:** {mean_error:.4f} pixels

**Median Reprojection Error:** {median_error:.4f} pixels

**Alignment Confidence Score:** {confidence:.2f}%
"""
        )

        if mean_error < 1.0:

            st.success(
                "🎯 Sub-pixel mean reprojection error achieved "
                "for this evaluated image pair."
            )

        else:

            st.warning(
                "Refinement completed. "
                "Sub-pixel mean error was not achieved "
                "for this image pair."
            )

else:

    st.info(
        "👆 Upload both a reference image and a source "
        "image to start LunarAlign-AI Pro."
    )