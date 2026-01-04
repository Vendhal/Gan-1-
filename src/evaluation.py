import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from scipy.linalg import sqrtm
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model

# ==================================================
# CONFIG
# ==================================================
REAL_DIR = "real_images"
FAKE_DIR = "fake_images"
IMG_SIZE = 75          # InceptionV3 minimum
NUM_DUMMY_IMAGES = 50
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

# ==================================================
# DUMMY IMAGE GENERATOR (FOR PARALLEL TEAMS)
# ==================================================
def generate_dummy_images(folder, label):
    print(f"⚠ No {label} images found. Generating dummy data...")
    for i in range(NUM_DUMMY_IMAGES):
        img = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        Image.fromarray(img).save(f"{folder}/{label}_{i}.png")

# ==================================================
# LOAD IMAGES
# ==================================================
def load_images(folder, label):
    images = []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if len(files) == 0:
        generate_dummy_images(folder, label)
        files = os.listdir(folder)

    for file in files:
        img = Image.open(os.path.join(folder, file)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        images.append(np.array(img))

    return np.array(images)

real_images = load_images(REAL_DIR, "real")
fake_images = load_images(FAKE_DIR, "fake")

print("Real images loaded:", real_images.shape)
print("Fake images loaded:", fake_images.shape)

# Normalize [-1, 1]
real_images = (real_images / 127.5) - 1
fake_images = (fake_images / 127.5) - 1

# ==================================================
# FEATURE EXTRACTOR
# ==================================================
base_model = InceptionV3(include_top=False, pooling="avg",
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))
feature_model = Model(base_model.input, base_model.output)

def extract_features(images):
    images = preprocess_input((images + 1) * 127.5)
    return feature_model.predict(images, verbose=0)

real_features = extract_features(real_images)
fake_features = extract_features(fake_images)

# ==================================================
# METRICS
# ==================================================
realism_score = np.mean(np.linalg.norm(fake_features, axis=1))
print(f"Classifier-based Realism Score: {realism_score:.4f}")

def diversity_score(features):
    return np.mean([
        np.linalg.norm(features[i] - features[j])
        for i in range(len(features))
        for j in range(i + 1, len(features))
    ])

div_score = diversity_score(fake_features[:30])
print(f"Diversity Score: {div_score:.4f}")

def fid_score(real, fake):
    mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
    mu2, sigma2 = fake.mean(axis=0), np.cov(fake, rowvar=False)
    diff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff + np.trace(sigma1 + sigma2 - 2 * covmean)

fid = fid_score(real_features, fake_features)
print(f"FID Proxy Score: {fid:.4f}")

# ==================================================
# IMAGE GRID
# ==================================================
plt.figure(figsize=(5,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow((fake_images[i] + 1) / 2)
    plt.axis("off")
plt.suptitle("Synthetic Images (Evaluation)")
plt.savefig(f"{FIG_DIR}/generated_grid.png")
plt.close()

# ==================================================
# t-SNE
# ==================================================
combined = np.vstack([real_features, fake_features])
labels = np.array([0]*len(real_features) + [1]*len(fake_features))

tsne = TSNE(n_components=2, random_state=42)
emb = tsne.fit_transform(combined)

plt.figure(figsize=(6,6))
plt.scatter(emb[labels==0,0], emb[labels==0,1], label="Real", alpha=0.5)
plt.scatter(emb[labels==1,0], emb[labels==1,1], label="Fake", alpha=0.5)
plt.legend()
plt.title("t-SNE: Real vs Synthetic")
plt.savefig(f"{FIG_DIR}/tsne_real_vs_fake.png")
plt.close()

print("✅ Module-4 Evaluation Completed Independently")
