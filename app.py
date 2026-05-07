import streamlit as st
from PIL import Image
import numpy as np
import filters


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Image Filters App", layout="wide")

st.title("🖼️ Image Filters App")
st.write("Upload an image and apply digital image processing filters.")


# ─────────────────────────────────────────────────────────────────────────────
#  Filter Objects (OOP)
# ─────────────────────────────────────────────────────────────────────────────

point_filters = filters.PointFilters()
spatial_filters = filters.SpatialFilters()
frequency_filters = filters.FrequencyFilters()


# ─────────────────────────────────────────────────────────────────────────────
#  Upload Image
# ─────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


# ─────────────────────────────────────────────────────────────────────────────
#  Presets
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "── None ──": [],

    # ─────────────────────────────────────────
    # Contrast Enhancement
    # ─────────────────────────────────────────

    "📖 Full Bone-Scan Pipeline": [
        "Laplacian",
        "Sobel Edge Detection",
        "Average Blur",
        "Power Law (Gamma)"
    ],

    # ─────────────────────────────────────────
    # Contrast Enhancement
    # ─────────────────────────────────────────

    # Power Law darkens or brightens the overall tone curve first,
    # then Histogram Equalization redistributes intensities globally
    # so the full 0-255 range is utilised — together they fix both
    # exposure and contrast in one pass.
    "📈 Contrast Enhancement": [
        "Power Law (Gamma)",
        "Histogram Equalization"
    ],

    # Power Law aggressively lifts shadow regions that are too dark
    # to recover by equalization alone, Gaussian then suppresses the
    # noise that gets amplified alongside the lifted shadows, and
    # Histogram Equalization finally stretches the cleaned signal
    # across the full intensity range.
    "🌙 Dark Image Recovery": [
        "Power Law (Gamma)",
        "Gaussian Filter",
        "Histogram Equalization"
    ],

    # ─────────────────────────────────────────
    # Denoising Pipelines
    # ─────────────────────────────────────────

    # Median is the best filter for salt-and-pepper noise because it
    # replaces each pixel with the neighbourhood median, killing spikes
    # without blurring edges. Histogram Equalization then restores the
    # contrast that the median pass may have slightly flattened.
    "🧹 Median Denoise": [
        "Median",
        "Histogram Equalization"
    ],

    # Gaussian smooths continuous (Gaussian) noise by averaging
    # neighbours with a bell-curve weight, reducing random intensity
    # fluctuations. Histogram Equalization follows to recover the
    # global contrast that smoothing tends to reduce.
    "☁️ Gaussian Smoothing": [
        "Gaussian Filter",
        "Histogram Equalization"
    ],

    # Weighted Smoothing uses a Gaussian-like 3×3 kernel that gives
    # more importance to the centre pixel, so it smooths gently
    # without destroying fine structure. Histogram Equalization then
    # lifts the contrast back after the softening effect.
    "🧼 Smooth + Contrast": [
        "Weighted Smoothing",
        "Histogram Equalization"
    ],

    # ─────────────────────────────────────────
    # Sharpening Pipelines
    # ─────────────────────────────────────────

    # Histogram Equalization first spreads intensities so that the
    # Laplacian has balanced contrast to work with — sharpening a
    # low-contrast image only amplifies flat regions. Laplacian then
    # uses its 4-connectivity second-derivative kernel to enhance
    # edges and fine detail on the already well-contrasted image.
    "🔪 Sharpen": [
        "Histogram Equalization",
        "Laplacian"
    ],

    # Median removes salt-and-pepper spikes first so Laplacian does
    # not mistake noise for edges and amplify it. Gaussian then
    # smooths the residual continuous noise that Median leaves behind.
    # Laplacian finally sharpens only the true structural edges that
    # survived both denoising passes.
    "🧹 Denoise then Sharpen": [
        "Median",
        "Gaussian Filter",
        "Laplacian"
    ],

    # ─────────────────────────────────────────
    # Edge Detection Pipelines
    # ─────────────────────────────────────────

    # Sobel computes the first-order gradient, so any noise present
    # appears as false edges. Gaussian pre-smooths the image to
    # suppress that noise before Sobel runs, ensuring only real
    # structural boundaries produce strong gradient responses.
    "🔍 Edge Detection": [
        "Gaussian Filter",
        "Sobel"
    ],

    # Median is chosen over Gaussian here because salt-and-pepper
    # noise creates isolated extreme pixels that Sobel reads as very
    # strong edges. Median eliminates those spikes entirely before
    # Sobel runs, producing cleaner and more accurate edge maps than
    # Gaussian smoothing would in noisy conditions.
    "🧹 Denoise then Edge": [
        "Median",
        "Sobel"
    ],

    # ─────────────────────────────────────────
    # Frequency Domain Pipelines
    # ─────────────────────────────────────────

    # Butterworth Low Pass attenuates high-frequency noise smoothly
    # in the frequency domain without the ringing artifacts of the
    # Ideal filter. Histogram Equalization then restores contrast
    # in the spatial domain after the frequency pass has suppressed
    # the noise energy.
    "🌊 Frequency Smooth": [
        "Butterworth Low Pass",
        "Histogram Equalization"
    ],

    # Butterworth High Pass isolates high-frequency edge components
    # in the frequency domain with a smooth roll-off so mid-frequency
    # transitions are preserved. Laplacian then reinforces those
    # recovered edges spatially, producing a doubly-sharpened result
    # that is stronger than either filter alone.
    "⚡ Frequency Sharpen": [
        "Butterworth High Pass",
        "Laplacian"
    ],

    # Ideal Low Pass cuts all frequencies beyond the radius with a
    # hard boundary, effectively removing high-frequency noise in one
    # step. Median follows in the spatial domain to clean up any
    # residual impulse noise that the frequency filter cannot remove
    # because impulse noise is spread across all frequencies.
    "🔬 Frequency + Spatial Denoise": [
        "Ideal Low Pass",
        "Median"
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Default Parameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "Negative": {},
    "Threshold": {"threshold_val": 128},
    "Logarithmic": {},
    "Power Law (Gamma)": {"gamma": 0.5},
    "Grey Level Slicing": {"a": 100, "b": 150},
    "Bit Plane Slicing": {"bits": 3},
    "Histogram Equalization": {},
    "Average Blur": {"n": 5},
    "Weighted Smoothing": {},
    "Median": {"size": 7},
    "Laplacian": {},
    "Sobel Edge Detection": {},
    "Ideal Low Pass": {"cutoff": 30},
    "Ideal High Pass": {"cutoff": 135},
    "Butterworth Low Pass": {"cutoff": 13, "order": 2},
    "Butterworth High Pass": {"cutoff": 37, "order": 5},
    "Gaussian Filter": {"kernel_size": 5,"sigma": 1.5}
}


ALL_FILTERS = list(DEFAULT_PARAMS.keys())


FILTER_CATEGORY = {
    "Negative": "Point Filters",
    "Threshold": "Point Filters",
    "Logarithmic": "Point Filters",
    "Power Law (Gamma)": "Point Filters",
    "Grey Level Slicing": "Point Filters",
    "Bit Plane Slicing": "Point Filters",
    "Histogram Equalization": "Point Filters",

    "Average Blur": "Spatial Filters",
    "Weighted Smoothing": "Spatial Filters",
    "Median": "Spatial Filters",
    "Laplacian": "Spatial Filters",
    "Sobel Edge Detection": "Spatial Filters",

    "Ideal Low Pass": "Frequency Filters",
    "Ideal High Pass": "Frequency Filters",
    "Butterworth Low Pass": "Frequency Filters",
    "Butterworth High Pass": "Frequency Filters",
    "Butterworth High Pass": "Frequency Filters"
}


# ─────────────────────────────────────────────────────────────────────────────
#  Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def to_grayscale(image_array: np.ndarray) -> np.ndarray:

    if image_array.ndim == 3:
        gray = np.mean(image_array, axis=2).astype(np.uint8)

    else:
        gray = image_array.astype(np.uint8)

    gray = np.abs(gray)
    gray = np.clip(gray, 0, 255)

    return gray.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Filter Application
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def apply_filter(
    gray: np.ndarray,
    filter_name: str,
    params: dict
) -> np.ndarray:

    # ─────────────────────────────────────────
    # Point Filters
    # ─────────────────────────────────────────

    if filter_name == "Negative":

        result = point_filters.negative(gray)

    elif filter_name == "Threshold":

        result = point_filters.threshold(
            gray,
            threshold_val=params.get("threshold_val", 128)
        )

    elif filter_name == "Logarithmic":

        result = point_filters.logarithmic(gray)

    elif filter_name == "Power Law (Gamma)":

        result = point_filters.power_law(
            gray,
            gamma=params.get("gamma", 2.2)
        )

    elif filter_name == "Grey Level Slicing":

        result = point_filters.grey_level_slicing(
            gray,
            a=params.get("a", 100),
            b=params.get("b", 150)
        )

    elif filter_name == "Bit Plane Slicing":

        bits = params.get("bits", 3)

        mask = np.uint8(0xFF) & ~np.uint8((1 << bits) - 1)

        result = point_filters.bit_plane_slicing(
            gray,
            mask=mask
        )
    
    elif filter_name == "Histogram Equalization":

        result = point_filters.hist_equ(gray)

    # ─────────────────────────────────────────
    # Spatial Filters
    # ─────────────────────────────────────────

    elif filter_name == "Average Blur":

        result = spatial_filters.average(
            gray,
            n=params.get("n", 5)
        )

    elif filter_name == "Weighted Smoothing":

        result = spatial_filters.weighted_smoothing(gray)

    elif filter_name == "Median":

        result = spatial_filters.median(
            gray,
            med_size=params.get("size", 7)
        )

    elif filter_name == "Laplacian":

        result = spatial_filters.laplacian(gray)

    elif filter_name == "Sobel Edge Detection":

        result = spatial_filters.sobel(gray)

    # ─────────────────────────────────────────
    # Frequency Filters
    # ─────────────────────────────────────────

    elif filter_name == "Ideal Low Pass":

        result = frequency_filters.ideal_low_pass(
            gray,
            radius=params.get("cutoff", 30)
        )

    elif filter_name == "Ideal High Pass":

        result = frequency_filters.ideal_high_pass(
            gray,
            radius=params.get("cutoff", 135)
        )

    elif filter_name == "Butterworth Low Pass":

        result = frequency_filters.butterworth_low_pass(
            gray,
            radius=params.get("cutoff", 13),
            order=params.get("order", 2)
        )

    elif filter_name == "Butterworth High Pass":

        result = frequency_filters.butterworth_high_pass(
            gray,
            radius=params.get("cutoff", 37),
            order=params.get("order", 5)
        )
    elif filter_name == "Gaussian Filter":

        result = frequency_filters.gaussian_filter(
            gray,
            kernel_size=params.get("kernel_size", 5),
            sigma=params.get("sigma", 1.5)
        )

    else:

        result = gray

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def apply_pipeline(gray: np.ndarray, steps: list):

    intermediates = []

    current = gray.copy()

    for name, params in steps:

        current = apply_filter(current, name, params)

        intermediates.append(
            (name, current.copy())
        )

    return current, intermediates


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar Parameter Widgets
# ─────────────────────────────────────────────────────────────────────────────

def param_widgets(filter_name: str, key_prefix: str = "") -> dict:

    params = {}

    if filter_name == "Threshold":

        params["threshold_val"] = st.sidebar.slider(
            "Threshold Value",
            0,
            255,
            128,
            key=f"{key_prefix}_thr"
        )

    elif filter_name == "Power Law (Gamma)":

        params["gamma"] = st.sidebar.slider(
            "Gamma",
            0.1,
            5.0,
            0.5,
            key=f"{key_prefix}_gam"
        )

    elif filter_name == "Grey Level Slicing":

        params["a"] = st.sidebar.slider(
            "Lower Bound (a)",
            0,
            255,
            100,
            key=f"{key_prefix}_a"
        )

        params["b"] = st.sidebar.slider(
            "Upper Bound (b)",
            0,
            255,
            150,
            key=f"{key_prefix}_b"
        )

    elif filter_name == "Bit Plane Slicing":

        params["bits"] = st.sidebar.slider(
            "LSBs to zero",
            1,
            7,
            3,
            key=f"{key_prefix}_bits"
        )

    elif filter_name == "Average Blur":

        params["n"] = st.sidebar.slider(
            "Kernel Size",
            3,
            21,
            5,
            step=2,
            key=f"{key_prefix}_n"
        )

    elif filter_name == "Median":

        params["size"] = st.sidebar.slider(
            "Neighbourhood Size",
            3,
            21,
            7,
            step=2,
            key=f"{key_prefix}_med"
        )

    elif filter_name in (
        "Ideal Low Pass",
        "Ideal High Pass"
    ):

        default_r = 30 if "Low" in filter_name else 135

        params["cutoff"] = st.sidebar.slider(
            "Cutoff Radius",
            1,
            300,
            default_r,
            key=f"{key_prefix}_cut"
        )

    elif filter_name in (
        "Butterworth Low Pass",
        "Butterworth High Pass"
    ):

        default_r = 13 if "Low" in filter_name else 37
        default_o = 2 if "Low" in filter_name else 5

        params["cutoff"] = st.sidebar.slider(
            "Cutoff Radius",
            1,
            300,
            default_r,
            key=f"{key_prefix}_cut"
        )

        params["order"] = st.sidebar.slider(
            "Filter Order",
            1,
            10,
            default_o,
            key=f"{key_prefix}_ord"
        )

    elif filter_name == "Gaussian Filter":

        params["kernel_size"] = st.sidebar.slider(
            "Kernel Size",
            3,
            21,
            5,
            step=2,
            key=f"{key_prefix}_gk"
        )

        params["sigma"] = st.sidebar.slider(
            "Sigma",
            0.1,
            10.0,
            1.5,
            key=f"{key_prefix}_sig"
        )

    return params


# ─────────────────────────────────────────────────────────────────────────────
#  Main UI
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    if image.mode == "RGBA":
        image = image.convert("RGB")

    image_array = np.array(image)

    gray_base = to_grayscale(image_array)

    # ─────────────────────────────────────────
    # Mode Selection
    # ─────────────────────────────────────────

    st.sidebar.header("⚙️ Mode")

    mode = st.sidebar.radio(
        "",
        ["Single Filter", "Combine Filters"],
        horizontal=True
    )

    # ═════════════════════════════════════════════════════════════════════
    #  SINGLE FILTER MODE
    # ═════════════════════════════════════════════════════════════════════

    if mode == "Single Filter":

        st.sidebar.markdown("---")

        filter_option = st.sidebar.selectbox(
            "Select a Filter",
            ["── Original ──"] + ALL_FILTERS
        )

        params = {}

        if filter_option != "── Original ──":

            st.sidebar.markdown(
                f"*Category: {FILTER_CATEGORY.get(filter_option, '')}*"
            )

            params = param_widgets(
                filter_option,
                key_prefix="single"
            )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            if filter_option == "── Original ──":

                st.subheader("No Filter Applied")

                st.image(
                    image,
                    use_container_width=True
                )

            else:

                st.subheader(
                    f"Filtered — {filter_option}"
                )

                with st.spinner("Applying filter..."):

                    result = apply_filter(
                        gray_base,
                        filter_option,
                        params
                    )

                st.image(
                    result,
                    use_container_width=True,
                    clamp=True
                )

    # ═════════════════════════════════════════════════════════════════════
    #  COMBINE FILTERS MODE
    # ═════════════════════════════════════════════════════════════════════

    else:

        st.sidebar.markdown("---")

        st.sidebar.subheader("📋 Recommended Presets")

        preset_name = st.sidebar.selectbox(
            "Load a preset",
            list(PRESETS.keys())
        )

        if preset_name != "── None ──":

            if st.sidebar.button("✅ Load Preset"):

                st.session_state["pipeline"] = list(
                    PRESETS[preset_name]
                )

                st.rerun()

        st.sidebar.markdown("---")

        st.sidebar.subheader("🛠️ Build Your Pipeline")

        if "pipeline" not in st.session_state:

            st.session_state["pipeline"] = []

        new_filter = st.sidebar.selectbox(
            "Add filter",
            ["── pick ──"] + ALL_FILTERS,
            key="add_sel"
        )

        if st.sidebar.button("➕ Add to Pipeline"):

            if new_filter != "── pick ──":

                st.session_state["pipeline"].append(
                    new_filter
                )

                st.rerun()

        # Show Pipeline

        if st.session_state["pipeline"]:

            st.sidebar.markdown(
                "**Current pipeline:**"
            )

            for i, fname in enumerate(
                list(st.session_state["pipeline"])
            ):

                c1, c2 = st.sidebar.columns([3, 1])

                c1.markdown(f"`{i+1}.` {fname}")

                if c2.button(
                    "✖",
                    key=f"rm_{i}"
                ):

                    st.session_state["pipeline"].pop(i)

                    st.rerun()

            if st.sidebar.button("🗑️ Clear All"):

                st.session_state["pipeline"] = []

                st.rerun()

        else:

            st.sidebar.info(
                "Pipeline empty."
            )

        # Parameters

        step_params = []

        if st.session_state["pipeline"]:

            st.sidebar.markdown("---")

            st.sidebar.subheader("🎛️ Step Parameters")

            needs_widget = {
                "Threshold",
                "Power Law (Gamma)",
                "Grey Level Slicing",
                "Bit Plane Slicing",
                "Average Blur",
                "Median",
                "Ideal Low Pass",
                "Ideal High Pass",
                "Butterworth Low Pass",
                "Butterworth High Pass",
                "Gaussian Filter"
            }

            for i, fname in enumerate(
                st.session_state["pipeline"]
            ):

                if fname in needs_widget:

                    st.sidebar.markdown(
                        f"**Step {i+1} — {fname}**"
                    )

                    p = param_widgets(
                        fname,
                        key_prefix=f"step{i}"
                    )

                else:

                    p = DEFAULT_PARAMS.get(
                        fname,
                        {}
                    )

                step_params.append(
                    (fname, p)
                )

        # Display

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        intermediates = []

        with col2:

            if not st.session_state["pipeline"]:

                st.subheader("No Pipeline Defined")

                st.image(
                    image,
                    use_container_width=True
                )

            else:

                pipeline_label = " → ".join(
                    st.session_state["pipeline"]
                )

                st.subheader("Pipeline Result")

                st.caption(pipeline_label)

                with st.spinner("Running pipeline..."):

                    final, intermediates = apply_pipeline(
                        gray_base,
                        step_params
                    )

                st.image(
                    final,
                    use_container_width=True,
                    clamp=True
                )

        # Intermediate Steps

        if intermediates:

            st.markdown("---")

            st.subheader("🔬 Intermediate Steps")

            all_steps = [
                ("Original", gray_base)
            ] + intermediates

            cols = st.columns(len(all_steps))

            for col, (label, img_arr) in zip(cols, all_steps):

                col.image(
                    img_arr,
                    caption=label,
                    use_container_width=True,
                    clamp=True
                )

else:

    st.info(
        "👆 Upload an image to start."
    )