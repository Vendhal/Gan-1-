import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# -------------------------------------------------
# Resolve project root (Exp-1)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
INPUT_DIR = os.path.join(DATASET_DIR, "images")
OUTPUT_DIR = os.path.join(DATASET_DIR, "preprocessed")

IMAGE_SIZE = 64
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def normalize_tanh(img):
    img = img.astype(np.float32) / 255.0
    return img * 2.0 - 1.0


def preprocess_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    img = np.array(img)
    return normalize_tanh(img)


def get_all_images(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, f))
    return paths


def preprocess_dataset():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = get_all_images(INPUT_DIR)
    print(f"🔍 Found {len(image_paths)} images")

    for path in tqdm(image_paths):
        try:
            img = preprocess_image(path)
            name = os.path.splitext(os.path.basename(path))[0]
            np.save(os.path.join(OUTPUT_DIR, name + ".npy"), img)
        except Exception as e:
            print(f"⚠️ Skipped {path}: {e}")

    print("✅ Preprocessing complete")


if __name__ == "__main__":
    preprocess_dataset()
