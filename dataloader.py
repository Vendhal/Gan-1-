import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# -------------------------------------------------
# Always resolve paths relative to this file
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# You can put images anywhere inside this folder (even subfolders)
INPUT_DIR = os.path.join(DATASET_DIR, "images")

OUTPUT_DIR = os.path.join(DATASET_DIR, "preprocessed")

IMAGE_SIZE = 64
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# -------------------------------------------------
# Normalize to [-1, 1]
# -------------------------------------------------
def normalize_tanh(img):
    img = img.astype(np.float32) / 255.0
    return img * 2.0 - 1.0


# -------------------------------------------------
# Preprocess single image
# -------------------------------------------------
def preprocess_image(path):
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img)
    img = normalize_tanh(img)
    return img


# -------------------------------------------------
# Recursively collect images
# -------------------------------------------------
def get_all_images(root_dir):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, file))
    return image_paths


# -------------------------------------------------
# Main preprocessing logic
# -------------------------------------------------
def preprocess_dataset():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = get_all_images(INPUT_DIR)

    print(f"\n🔍 Found {len(image_paths)} image(s)")

    if len(image_paths) == 0:
        print("\n❌ NO IMAGES FOUND")
        print("➡ Put your images anywhere inside:")
        print(INPUT_DIR)
        print("\nThen run the script again.\n")
        return

    for img_path in tqdm(image_paths, desc="Preprocessing images"):
        processed = preprocess_image(img_path)

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(OUTPUT_DIR, img_name + ".npy")

        np.save(save_path, processed)

    print("\n✅ DONE!")
    print("Preprocessed images saved in:")
    print(OUTPUT_DIR)


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    preprocess_dataset()