from fastapi import FastAPI
import numpy as np

app = FastAPI(title="GAN Deployment API (Standalone)")

IMAGE_SIZE = 64


def simulate_gan_output(n_images=5, randomness=1.0):
    """
    Simulated GAN inference for standalone Module-5
    Output shape: (N, 64, 64)
    """
    images = np.random.rand(n_images, IMAGE_SIZE, IMAGE_SIZE)
    images = np.clip(images * randomness, 0, 1)
    return images


@app.get("/generate")
def generate_images(n: int = 5, randomness: float = 1.0):
    images = simulate_gan_output(n, randomness)

    return {
        "status": "success",
        "generated_images": n,
        "image_size": IMAGE_SIZE,
        "range": "[0,1]",
    }
