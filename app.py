import streamlit as st
from PIL import Image
import numpy as np
from filters import PointFilters, SpatialFilters, FrequencyFilters

st.set_page_config(page_title="Image Filters App", layout="wide")
st.title("🖼️ Image Filters App")
st.write("Upload an image and apply various digital image processing filters.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


# ─────────────────────────────────────────────────────────────────────────────
#  Recommended combinations (from lecture slides 29-32)
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "── None ──": [],
    "📖 Full Bone-Scan Pipeline": [
        "Laplacian", "Sobel Edge Detection", "Average Blur", "Power Law (Gamma)"
    ],
    "🔪 Sharpen with Laplacian": [
        "Laplacian"
    ],
    "🔍 Edge Detection (Sobel only)": [
        "Sobel Edge Detection"
    ],
    "🧹 Denoise then Sharpen": [
        "Median", "Laplacian"
    ],
    "🌀 Smooth then Detect Edges": [
        "Average Blur", "Sobel Edge Detection"
    ],
    "🎚️ Gamma Correction + Sharpen": [
        "Power Law (Gamma)", "Laplacian"
    ],
    "🔆 Median + Power Law": [
        "Median", "Power Law (Gamma)"
    ],
    "🏔️ Frequency LPF then Sharpen": [
        "Butterworth Low Pass", "Laplacian"
    ],
}

# Default params for each filter
DEFAULT_PARAMS = {
    "Negative":              {},
    "Threshold":             {"threshold_val": 128},
    "Logarithmic":           {},
    "Power Law (Gamma)":     {"gamma": 0.5},
    "Grey Level Slicing":    {"a": 100, "b": 150},
    "Bit Plane Slicing":     {"bits": 3},
    "Average Blur":          {"n": 5},
    "Weighted Smoothing":    {},
    "Median":                {"size": 7},
    "Laplacian":             {},
    "Sobel Edge Detection":  {},
    "Ideal Low Pass":        {"cutoff": 30},
    "Ideal High Pass":       {"cutoff": 135},
    "Butterworth Low Pass":  {"cutoff": 13, "order": 2},
    "Butterworth High Pass": {"cutoff": 37, "order": 5},
}

ALL_FILTERS = list(DEFAULT_PARAMS.keys())

FILTER_CATEGORY = {
    "Negative":              "Point Filters",
    "Threshold":             "Point Filters",
    "Logarithmic":           "Point Filters",
    "Power Law (Gamma)":     "Point Filters",
    "Grey Level Slicing":    "Point Filters",
    "Bit Plane Slicing":     "Point Filters",
    "Average Blur":          "Spatial Filters",
    "Weighted Smoothing":    "Spatial Filters",
    "Median":                "Spatial Filters",
    "Laplacian":             "Spatial Filters",
    "Sobel Edge Detection":  "Spatial Filters",
    "Ideal Low Pass":        "Frequency Filters",
    "Ideal High Pass":       "Frequency Filters",
    "Butterworth Low Pass":  "Frequency Filters",
    "Butterworth High Pass": "Frequency Filters",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Pure-math filter functions (no cv2 window calls)
# ─────────────────────────────────────────────────────────────────────────────

def to_grayscale(image_array: np.ndarray) -> np.ndarray:
    if image_array.ndim == 3:
        gray = np.mean(image_array, axis=2).astype(np.uint8)
    else:
        gray = image_array.astype(np.uint8)
    return np.clip(np.abs(gray), 0, 255).astype(np.uint8)


def to_displayable(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255
    return np.clip(arr, 0, 255).astype(np.uint8)


@st.cache_data
def apply_filter(gray: np.ndarray, filter_name: str, params: dict) -> np.ndarray:
    """Apply a single named filter to a 2-D grayscale uint8 array."""

    if filter_name == "Negative":
        result = 255 - gray

    elif filter_name == "Threshold":
        result = np.where(gray > params.get("threshold_val", 128), 255, 0).astype(np.uint8)

    elif filter_name == "Logarithmic":
        normed = gray.astype(np.float32) / 255
        result = to_displayable(np.log1p(1 + normed))

    elif filter_name == "Power Law (Gamma)":
        normed = gray.astype(np.float32) / 255
        result = to_displayable(np.power(normed, params.get("gamma", 2.2)))

    elif filter_name == "Grey Level Slicing":
        a, b = params.get("a", 100), params.get("b", 150)
        result = np.where((gray >= a) & (gray <= b), 255, 0).astype(np.uint8)

    elif filter_name == "Bit Plane Slicing":
        bits = params.get("bits", 3)
        mask = np.uint8(0xFF) & ~np.uint8((1 << bits) - 1)
        result = (gray & mask).astype(np.uint8)

    elif filter_name == "Average Blur":
        from scipy.ndimage import convolve
        n = params.get("n", 5)
        kernel = np.ones((n, n)) / n ** 2
        result = to_displayable(convolve(gray.astype(np.float32), kernel, mode="reflect"))

    elif filter_name == "Weighted Smoothing":
        from scipy.ndimage import convolve
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
        kernel = np.tile(kernel, 12)
        kernel /= kernel.sum()
        result = to_displayable(convolve(gray.astype(np.float32), kernel, mode="reflect"))

    elif filter_name == "Median":
        from scipy.ndimage import median_filter
        result = median_filter(gray, size=params.get("size", 7), mode="reflect").astype(np.uint8)

    elif filter_name == "Laplacian":
        from scipy.ndimage import convolve
        w = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        result = to_displayable(convolve(gray.astype(np.float32), w, mode="reflect"))

    elif filter_name == "Sobel Edge Detection":
        from scipy.ndimage import convolve
        Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gx = convolve(gray.astype(np.float32), Sx, mode="reflect")
        Gy = convolve(gray.astype(np.float32), Sy, mode="reflect")
        result = to_displayable(np.sqrt(Gx ** 2 + Gy ** 2))

    elif filter_name in ("Ideal Low Pass", "Ideal High Pass"):
        radius = params.get("cutoff", 30 if "Low" in filter_name else 135)
        F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        rows, cols = gray.shape
        U, V = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        dist = np.sqrt((U - rows // 2) ** 2 + (V - cols // 2) ** 2)
        fmask = (dist <= radius) if "Low" in filter_name else (dist > radius)
        result = to_displayable(np.abs(np.fft.ifft2(np.fft.ifftshift(F * fmask))))

    elif filter_name in ("Butterworth Low Pass", "Butterworth High Pass"):
        radius = params.get("cutoff", 13 if "Low" in filter_name else 37)
        order  = params.get("order",  2  if "Low" in filter_name else 5)
        F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        rows, cols = gray.shape
        U, V = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        dist = np.sqrt((U - rows // 2) ** 2 + (V - cols // 2) ** 2)
        if "Low" in filter_name:
            H = 1 / (1 + (dist / radius) ** (2 * order))
        else:
            dist = np.where(dist == 0, 1e-6, dist)
            H = 1 / (1 + (radius / dist) ** (2 * order))
        result = to_displayable(np.abs(np.fft.ifft2(np.fft.ifftshift(F * H))))

    else:
        result = gray

    return result.astype(np.uint8)


def apply_pipeline(gray: np.ndarray, steps: list) -> tuple:
    """Apply a list of (filter_name, params) tuples sequentially.
    Returns (final_image, list_of_intermediate_images).
    """
    intermediates = []
    current = gray.copy()
    for name, params in steps:
        current = apply_filter(current, name, params)
        intermediates.append((name, current.copy()))
    return current, intermediates


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar param widget helper
# ─────────────────────────────────────────────────────────────────────────────

def param_widgets(filter_name: str, key_prefix: str = "") -> dict:
    params = {}
    if filter_name == "Threshold":
        params["threshold_val"] = st.sidebar.slider(
            "Threshold Value", 0, 255, 128, key=f"{key_prefix}_thr")
    elif filter_name == "Power Law (Gamma)":
        params["gamma"] = st.sidebar.slider(
            "Gamma", 0.1, 5.0, 0.5, key=f"{key_prefix}_gam")
    elif filter_name == "Grey Level Slicing":
        params["a"] = st.sidebar.slider("Lower Bound (a)", 0, 255, 100, key=f"{key_prefix}_a")
        params["b"] = st.sidebar.slider("Upper Bound (b)", 0, 255, 150, key=f"{key_prefix}_b")
    elif filter_name == "Bit Plane Slicing":
        params["bits"] = st.sidebar.slider("LSBs to zero", 1, 7, 3, key=f"{key_prefix}_bits")
    elif filter_name == "Average Blur":
        params["n"] = st.sidebar.slider(
            "Kernel Size (N)", 3, 21, 5, step=2, key=f"{key_prefix}_n")
    elif filter_name == "Median":
        params["size"] = st.sidebar.slider(
            "Neighbourhood Size", 3, 21, 7, step=2, key=f"{key_prefix}_med")
    elif filter_name in ("Ideal Low Pass", "Ideal High Pass"):
        default_r = 30 if "Low" in filter_name else 135
        params["cutoff"] = st.sidebar.slider(
            "Cutoff Radius (px)", 1, 300, default_r, key=f"{key_prefix}_cut")
    elif filter_name in ("Butterworth Low Pass", "Butterworth High Pass"):
        default_r = 13 if "Low" in filter_name else 37
        default_o = 2  if "Low" in filter_name else 5
        params["cutoff"] = st.sidebar.slider(
            "Cutoff Radius (px)", 1, 300, default_r, key=f"{key_prefix}_cut")
        params["order"] = st.sidebar.slider(
            "Filter Order", 1, 10, default_o, key=f"{key_prefix}_ord")
    return params


# ─────────────────────────────────────────────────────────────────────────────
#  Main UI
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image_array = np.array(image)
    gray_base   = to_grayscale(image_array)

    # Mode toggle
    st.sidebar.header("⚙️ Mode")
    mode = st.sidebar.radio("", ["Single Filter", "Combine Filters"], horizontal=True)

    # ══════════════════════════════════════════════════════════════════════
    #  SINGLE FILTER MODE
    # ══════════════════════════════════════════════════════════════════════
    if mode == "Single Filter":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filter")

        filter_option = st.sidebar.selectbox(
            "Select a filter", ["── Original ──"] + ALL_FILTERS)

        params = {}
        if filter_option != "── Original ──":
            st.sidebar.markdown(f"*Category: {FILTER_CATEGORY.get(filter_option, '')}*")
            params = param_widgets(filter_option, key_prefix="single")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with col2:
            if filter_option == "── Original ──":
                st.subheader("No filter applied")
                st.image(image, use_container_width=True)
            else:
                st.subheader(f"Filtered — {filter_option}")
                st.caption(f"Category: {FILTER_CATEGORY.get(filter_option, '')}")
                with st.spinner("Applying filter…"):
                    result = apply_filter(gray_base, filter_option, params)
                st.image(result, use_container_width=True, clamp=True)

    # ══════════════════════════════════════════════════════════════════════
    #  COMBINE FILTERS MODE
    # ══════════════════════════════════════════════════════════════════════
    else:
        st.sidebar.markdown("---")

        # ── Preset picker ──────────────────────────────────────────────────
        st.sidebar.subheader("📋 Recommended Presets")
        preset_name = st.sidebar.selectbox("Load a preset", list(PRESETS.keys()))

        if preset_name != "── None ──":
            if st.sidebar.button("✅ Load Preset"):
                st.session_state["pipeline"] = list(PRESETS[preset_name])
                st.rerun()

        st.sidebar.markdown("---")

        # ── Manual pipeline builder ────────────────────────────────────────
        st.sidebar.subheader("🛠️ Build Your Pipeline")

        if "pipeline" not in st.session_state:
            st.session_state["pipeline"] = []

        new_filter = st.sidebar.selectbox(
            "Add filter to pipeline", ["── pick ──"] + ALL_FILTERS, key="add_sel")
        if st.sidebar.button("➕ Add to Pipeline"):
            if new_filter != "── pick ──":
                st.session_state["pipeline"].append(new_filter)
                st.rerun()

        # Show current pipeline with per-step remove buttons
        if st.session_state["pipeline"]:
            st.sidebar.markdown("**Current pipeline:**")
            for i, fname in enumerate(list(st.session_state["pipeline"])):
                c1, c2 = st.sidebar.columns([3, 1])
                c1.markdown(f"`{i+1}.` {fname}")
                if c2.button("✖", key=f"rm_{i}"):
                    st.session_state["pipeline"].pop(i)
                    st.rerun()
            if st.sidebar.button("🗑️ Clear All"):
                st.session_state["pipeline"] = []
                st.rerun()
        else:
            st.sidebar.info("Pipeline empty — add filters or load a preset.")

        # ── Per-step params ────────────────────────────────────────────────
        step_params = []
        if st.session_state["pipeline"]:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🎛️ Step Parameters")
            needs_widget = {
                "Threshold", "Power Law (Gamma)", "Grey Level Slicing",
                "Bit Plane Slicing", "Average Blur", "Median",
                "Ideal Low Pass", "Ideal High Pass",
                "Butterworth Low Pass", "Butterworth High Pass",
            }
            for i, fname in enumerate(st.session_state["pipeline"]):
                if fname in needs_widget:
                    st.sidebar.markdown(f"**Step {i+1} — {fname}**")
                    p = param_widgets(fname, key_prefix=f"step{i}")
                else:
                    p = DEFAULT_PARAMS.get(fname, {})
                step_params.append((fname, p))

        # ── Display ────────────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        intermediates = []
        with col2:
            if not st.session_state["pipeline"]:
                st.subheader("No pipeline defined")
                st.image(image, use_container_width=True)
            else:
                pipeline_label = " → ".join(st.session_state["pipeline"])
                st.subheader("Pipeline Result")
                st.caption(pipeline_label)
                with st.spinner("Running pipeline…"):
                    final, intermediates = apply_pipeline(gray_base, step_params)
                st.image(final, use_container_width=True, clamp=True)

        # ── Intermediate steps strip ───────────────────────────────────────
        if intermediates:
            st.markdown("---")
            st.subheader("🔬 Intermediate Steps")
            all_steps = [("Original", gray_base)] + intermediates
            cols = st.columns(len(all_steps))
            for col, (label, img_arr) in zip(cols, all_steps):
                col.image(img_arr, caption=label, use_container_width=True, clamp=True)

            # Explanation expander for the lecture preset
            if preset_name == "📖 Lecture: Full Bone-Scan Pipeline (slides 29-32)":
                with st.expander("📖 How this pipeline works (lecture slides 29-32)"):
                    st.markdown("""
**Step-by-step breakdown — Gonzalez & Woods slides 29-32:**

| Step | Filter | Purpose |
|------|--------|---------|
| 1 | **Laplacian** | 2nd-derivative sharpening → highlights edges & fine detail; result subtracted from original |
| 2 | **Sobel Edge Detection** | 1st-derivative gradient magnitude → strong edge map used as a mask |
| 3 | **Average Blur (5×5)** | Smooth the Sobel mask so it blends softly |
| 4 | **Power Law (Gamma)** | Final contrast stretch on the combined result |

**Key formula from the lecture:**

`g(x,y) = f(x,y) − ∇²f`  → Laplacian sharpening  
then multiply with Sobel mask → add back to sharpened → gamma stretch

The lecture shows that combining these operations produces far more detail than any single filter alone.
                    """)

else:
    st.info("👆 Upload an image using the file uploader above to get started.")