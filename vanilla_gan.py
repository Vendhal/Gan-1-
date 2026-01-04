import torch
import torch.nn as nn
import torch.optim as optim

from generator import Generator
from discriminator import Discriminator


# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Hyperparameters (as per spec)
# -------------------------------------------------
LATENT_DIM = 100
LR = 0.0002
BETA1 = 0.5


# -------------------------------------------------
# Models
# -------------------------------------------------
generator = Generator(latent_dim=LATENT_DIM).to(device)
discriminator = Discriminator().to(device)


# -------------------------------------------------
# Loss function
# -------------------------------------------------
criterion = nn.BCELoss()


# -------------------------------------------------
# Optimizers
# -------------------------------------------------
optimizer_G = optim.Adam(
    generator.parameters(),
    lr=LR,
    betas=(BETA1, 0.999)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=LR,
    betas=(BETA1, 0.999)
)


# -------------------------------------------------
# Sanity check (optional)
# -------------------------------------------------
if __name__ == "__main__":
    z = torch.randn(4, LATENT_DIM, 1, 1).to(device)
    fake_images = generator(z)
    preds = discriminator(fake_images)

    print("Generator output shape:", fake_images.shape)
    print("Discriminator output shape:", preds.shape)
