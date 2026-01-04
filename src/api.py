import time
from fastapi import FastAPI
from src.inference import generate_images
from src.monitoring import log_inference

app = FastAPI(title="Vanilla GAN API (PyTorch)")

MODEL_VERSION = "final"


@app.get("/generate")
def generate(n: int = 5, randomness: float = 1.0):
    start = time.time()
    status = "SUCCESS"

    try:
        images = generate_images(n, randomness)
    except Exception as e:
        images = None
        status = f"FAILED: {e}"

    latency = time.time() - start
    log_inference(MODEL_VERSION, n, latency, status)

    return {
        "status": status,
        "count": n,
        "latency": latency
    }
