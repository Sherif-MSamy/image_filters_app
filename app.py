import streamlit as st
from PIL import Image
import numpy as np
from filters import (
    BrightnessFilter, ContrastFilter, GammaFilter, HistogramEqualizationFilter,
    BoxBlurFilter, GaussianFilter, SobelFilter,
    IdealLowPassFilter, ButterworthLowPassFilter
)

st.set_page_config(page_title="Image Filters App", layout="wide")

st.title("🖼️ Image Filters App")
st.write("Upload an image and apply various digital image processing filters.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

@st.cache_data
def apply_filter(image_array, filter_name, params):
    if filter_name == "Brightness":
        return BrightnessFilter(params['value']).apply(image_array)
    elif filter_name == "Contrast":
        return ContrastFilter(params['factor']).apply(image_array)
    elif filter_name == "Gamma":
        return GammaFilter(params['gamma']).apply(image_array)
    elif filter_name == "Histogram Equalization":
        return HistogramEqualizationFilter().apply(image_array)
    elif filter_name == "Box Blur":
        return BoxBlurFilter(params['size']).apply(image_array)
    elif filter_name == "Gaussian Blur":
        return GaussianFilter(params['size'], params['sigma']).apply(image_array)
    elif filter_name == "Sobel Edge Detection":
        return SobelFilter().apply(image_array)
    elif filter_name == "Ideal Low Pass":
        return IdealLowPassFilter(params['cutoff']).apply(image_array)
    elif filter_name == "Butterworth Low Pass":
        return ButterworthLowPassFilter(params['cutoff'], params['order']).apply(image_array)
    return image_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Convert to RGB if it's RGBA to avoid issues with some filters
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image_array = np.array(image)

    st.sidebar.header("Filter Settings")
    filter_option = st.sidebar.selectbox(
        "Select a filter",
        [
            "Original",
            "Gamma",
            "Sobel Edge Detection",
            "Brightness",
            "Contrast",
            "Histogram Equalization",
            "Box Blur",
            "Gaussian Blur",
            "Ideal Low Pass",
            "Butterworth Low Pass"
        ]
    )

    params = {}
    if filter_option == "Brightness":
        params['value'] = st.sidebar.slider("Brightness Value", -100, 100, 30)
    elif filter_option == "Contrast":
        params['factor'] = st.sidebar.slider("Contrast Factor", 0.0, 3.0, 1.5)
    elif filter_option == "Gamma":
        params['gamma'] = st.sidebar.slider("Gamma Value", 0.1, 5.0, 1.0)
    elif filter_option == "Box Blur":
        params['size'] = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
    elif filter_option == "Gaussian Blur":
        params['size'] = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
        params['sigma'] = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0)
    elif filter_option == "Ideal Low Pass":
        params['cutoff'] = st.sidebar.slider("Cutoff Frequency", 1.0, 200.0, 30.0)
    elif filter_option == "Butterworth Low Pass":
        params['cutoff'] = st.sidebar.slider("Cutoff Frequency", 1.0, 200.0, 30.0)
        params['order'] = st.sidebar.slider("Order", 1, 10, 2)

    if filter_option != "Original":
        filtered_image_array = apply_filter(image_array, filter_option, params)
        filtered_image = Image.fromarray(filtered_image_array)
    else:
        filtered_image = image

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader(f"Filtered Image ({filter_option})")
        st.image(filtered_image, use_container_width=True)
