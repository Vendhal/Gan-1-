import numpy as np

IMAGE_SIZE = 64
CHANNELS = 1


def generate_images(n_images=5, randomness=1.0):
    """
    Simulated GAN inference (standalone for Module-5)
    Output shape: (N, 64, 64)
    Output range: [0, 1]
    """
    images = np.random.rand(n_images, IMAGE_SIZE, IMAGE_SIZE)
    images = np.clip(images * randomness, 0, 1)
    return images


# -------------------------------
# Standalone test
# -------------------------------
if __name__ == "__main__":
    imgs = generate_images(3)
    print("Generated images shape:", imgs.shape)
    print("Min value:", imgs.min())
    print("Max value:", imgs.max())
