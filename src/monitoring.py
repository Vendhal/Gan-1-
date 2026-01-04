import csv
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "inference_logs.csv")

HEADERS = ["timestamp", "model_version", "num_images", "latency_sec", "status"]

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(HEADERS)


def log_inference(model_version, num_images, latency, status):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_version,
            num_images,
            round(latency, 4),
            status
        ])
