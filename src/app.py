# ===============================
# Streamlit App – Vanilla GAN
# ===============================

import sys
import os

# -------------------------------------------------
# FIX PYTHON PATH (CRITICAL FOR STREAMLIT ON WINDOWS)
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------
# Imports
# -------------------------------------------------
import streamlit as st
import numpy as np
import zipfile
from PIL import Image

from src.inference import generate_images

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(
    page_title="Vanilla GAN – Deployment Demo",
    layout="wide"
)

st.title("Vanilla GAN – Real PyTorch Inference")

num_images = st.slider("Number of images", 1, 20, 5)
randomness = st.slider("Randomness", 0.1, 1.5, 1.0)

# -------------------------------------------------
# Generate Images
# -------------------------------------------------
if st.button("Generate Images"):

    images = generate_images(num_images, randomness)

    os.makedirs("generated", exist_ok=True)
    paths = []

    cols = st.columns(5)

    for i, img in enumerate(images):
        # img shape: (1, 64, 64) → FIX IT
        img = img.squeeze()                     # (64, 64)
        img_uint8 = (img * 255).astype(np.uint8)

        pil_img = Image.fromarray(img_uint8, mode="L")

        path = f"generated/image_{i}.png"
        pil_img.save(path)
        paths.append(path)

        cols[i % 5].image(
            pil_img,
            caption=f"Image {i}",
            use_container_width=True
        )

    # -------------------------------------------------
    # ZIP Download
    # -------------------------------------------------
    zip_path = "generated_images.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for p in paths:
            zipf.write(p)

    with open(zip_path, "rb") as f:
        st.download_button(
            "Download Images (ZIP)",
            f,
            file_name="synthetic_images.zip",
            mime="application/zip"
        )
