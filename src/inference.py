import os
import time
import torch


from src.generator import Generator
from src.monitoring import log_inference

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")

# -------------------------------------------------
# Config
# -------------------------------------------------
LATENT_DIM = 100
IMG_CHANNELS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Model cache (prevents reloading every call)
# -------------------------------------------------
_LOADED_MODELS = {}

# -------------------------------------------------
# Load Generator (cached)
# -------------------------------------------------
def load_generator(version="final"):
    if version in _LOADED_MODELS:
        return _LOADED_MODELS[version]

    model_path = os.path.join(MODEL_DIR, f"G_{version}.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Generator model not found: {model_path}")

    gen = Generator(
        latent_dim=LATENT_DIM,
        img_channels=IMG_CHANNELS
    )

    gen.load_state_dict(torch.load(model_path, map_location=DEVICE))
    gen.to(DEVICE).eval()

    _LOADED_MODELS[version] = gen
    return gen

# -------------------------------------------------
# Inference + Monitoring
# -------------------------------------------------
def generate_images(n=5, randomness=1.0, version="final"):
    start_time = time.time()
    status = "SUCCESS"

    try:
        gen = load_generator(version)

        z = torch.randn(n, LATENT_DIM, 1, 1, device=DEVICE) * randomness

        with torch.no_grad():
            imgs = gen(z)

        # Normalize from [-1, 1] → [0, 1]
        imgs = (imgs + 1) / 2
        imgs = imgs.clamp(0, 1)

        imgs_np = imgs.cpu().numpy()

    except Exception as e:
        imgs_np = None
        status = f"FAILED: {str(e)}"

    latency = time.time() - start_time

    # -------------------------------------------------
    # 🔴 THIS IS THE CRITICAL LINE YOU WERE MISSING
    # -------------------------------------------------
    log_inference(
        model_version=version,
        num_images=n,
        latency=latency,
        status=status
    )

    if imgs_np is None:
        raise RuntimeError(status)

    return imgs_np
