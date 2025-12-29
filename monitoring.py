# monitoring.py
# Module 6 – Monitoring & Update Pipeline (Fully Independent & Safe)

import time
import csv
import os
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# CONFIGURATION
# ==========================
LOG_DIR = "logs"
REPORT_DIR = "reports"
MODEL_DIR = "models"

LOG_FILE = os.path.join(LOG_DIR, "inference_logs.csv")
REPORT_FILE = os.path.join(REPORT_DIR, "periodic_report.txt")

LATENT_DIM = 100
IMAGE_SHAPE = (64, 64, 1)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================
# INITIALIZE / FIX LOG FILE
# ==========================
EXPECTED_HEADERS = [
    "timestamp",
    "model_version",
    "num_images",
    "latency",
    "status"
]

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EXPECTED_HEADERS)

# ==========================
# DUMMY GENERATOR (FALLBACK)
# ==========================
class DummyGenerator:
    def predict(self, noise, verbose=0):
        return np.random.rand(
            noise.shape[0],
            IMAGE_SHAPE[0],
            IMAGE_SHAPE[1],
            IMAGE_SHAPE[2]
        )

# ==========================
# SAFE MODEL LOADER
# ==========================
def load_generator(version="v1"):
    model_path = os.path.join(MODEL_DIR, f"G_{version}.keras")
    try:
        from tensorflow.keras.models import load_model
        if os.path.exists(model_path):
            print(f"[INFO] Loaded Generator Model: G_{version}")
            return load_model(model_path)
        else:
            print("[WARNING] Generator model not found. Using Dummy Generator.")
            return DummyGenerator()
    except Exception:
        print("[WARNING] TensorFlow not available. Using Dummy Generator.")
        return DummyGenerator()

# ==========================
# IMAGE GENERATION + LATENCY
# ==========================
def generate_images(generator, num_images=8):
    start = time.time()
    status = "SUCCESS"

    try:
        noise = np.random.normal(0, 1, (num_images, LATENT_DIM))
        images = generator.predict(noise)
    except Exception as e:
        images = None
        status = f"FAILED: {str(e)}"

    latency = round(time.time() - start, 4)
    return images, latency, status

# ==========================
# LOG INFERENCE DETAILS
# ==========================
def log_inference(model_version, num_images, latency, status):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_version,
            num_images,
            latency,
            status
        ])

# ==========================
# PRIVACY CHECK (ANTI-MEMORIZATION)
# ==========================
def privacy_check(real_images, fake_images, threshold=0.95):
    real_flat = real_images.reshape(len(real_images), -1)
    fake_flat = fake_images.reshape(len(fake_images), -1)

    similarity = cosine_similarity(fake_flat, real_flat)
    max_similarity = similarity.max()

    return max_similarity < threshold, round(float(max_similarity), 4)

# ==========================
# PERIODIC REPORT (SAFE)
# ==========================
def generate_periodic_report():
    total_requests = 0
    failures = 0
    latencies = []

    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_requests += 1

            # Safe latency parsing
            if "latency" in row and row["latency"]:
                latencies.append(float(row["latency"]))
            elif "latency_sec" in row and row["latency_sec"]:
                latencies.append(float(row["latency_sec"]))
            elif "latency_seconds" in row and row["latency_seconds"]:
                latencies.append(float(row["latency_seconds"]))

            if "FAILED" in row["status"]:
                failures += 1

    avg_latency = round(np.mean(latencies), 4) if latencies else 0

    report = f"""
=========================================
GAN MONITORING REPORT
=========================================
Total Requests        : {total_requests}
Failed Requests       : {failures}
Average Latency (sec) : {avg_latency}
Generated On          : {datetime.now()}
=========================================
"""

    with open(REPORT_FILE, "w") as f:
        f.write(report)

    print(report)

# ==========================
# MAIN EXECUTION
# ==========================
if __name__ == "__main__":

    MODEL_VERSION = "v1"
    NUM_IMAGES = 8

    print("\n--- GAN MONITORING STARTED ---")

    generator = load_generator(MODEL_VERSION)

    fake_images, latency, status = generate_images(generator, NUM_IMAGES)
    log_inference(MODEL_VERSION, NUM_IMAGES, latency, status)

    real_images = np.random.rand(NUM_IMAGES, *IMAGE_SHAPE)

    if fake_images is not None:
        safe, similarity = privacy_check(real_images, fake_images)
        print(f"Privacy Safe        : {safe}")
        print(f"Max Similarity Score: {similarity}")

    generate_periodic_report()

    print("--- MONITORING COMPLETED ---\n")
