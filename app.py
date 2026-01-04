import streamlit as st
import numpy as np
import os
import zipfile
from PIL import Image

from inference import generate_images

st.set_page_config(page_title="GAN Deployment Demo", layout="wide")
st.title("Vanilla GAN – Deployment Demo")

num_images = st.slider("Number of images", 1, 20, 5)
randomness = st.slider("Randomness", 0.1, 1.0, 1.0)

if st.button("Generate Images"):
    images = generate_images(num_images, randomness)

    os.makedirs("generated", exist_ok=True)
    paths = []

    cols = st.columns(5)

    for i, img in enumerate(images):
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)

        path = f"generated/image_{i}.png"
        pil_img.save(path)
        paths.append(path)

        cols[i % 5].image(pil_img, caption=f"Image {i}", use_container_width=True)

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
