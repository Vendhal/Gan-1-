import os
import csv
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from src.generator import Generator
from src.discriminator import Discriminator
from src.torch_dataset import get_dataloader


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(BASE_DIR, "src", "config.yaml")
DATA_DIR = os.path.join(BASE_DIR, "dataset", "preprocessed")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# -------------------------------------------------
# Load config
# -------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

IMAGE_SIZE = config["image_size"]
CHANNELS = config["channels"]

LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0002
BETA1 = 0.5
SAVE_INTERVAL = 10


# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------
# Models
# -------------------------------------------------
generator = Generator(
    latent_dim=LATENT_DIM,
    img_channels=CHANNELS
).to(device)

discriminator = Discriminator(
    img_channels=CHANNELS
).to(device)


# -------------------------------------------------
# Loss & Optimizers
# -------------------------------------------------
criterion = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))


# -------------------------------------------------
# DataLoader
# -------------------------------------------------
dataloader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)


# -------------------------------------------------
# Logging
# -------------------------------------------------
csv_path = os.path.join(LOG_DIR, "training_log.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["epoch", "D_loss", "G_loss"])


# -------------------------------------------------
# Training Loop
# -------------------------------------------------
for epoch in range(1, EPOCHS + 1):

    d_losses = []
    g_losses = []

    for real_imgs in dataloader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # 🔑 Label smoothing
        real_labels = torch.full((batch_size,), 0.9, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        # =================================================
        # Train Discriminator
        # =================================================
        z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
        fake_imgs = generator(z)

        d_real = discriminator(real_imgs)
        d_fake = discriminator(fake_imgs.detach())

        d_loss_real = criterion(d_real, real_labels)
        d_loss_fake = criterion(d_fake, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # =================================================
        # Train Generator (TWICE)
        # =================================================
        for _ in range(2):
            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_imgs = generator(z)
            d_fake = discriminator(fake_imgs)

            g_loss = criterion(d_fake, real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    d_epoch = sum(d_losses) / len(d_losses)
    g_epoch = sum(g_losses) / len(g_losses)

    csv_writer.writerow([epoch, d_epoch, g_epoch])
    csv_file.flush()

    # -------------------------------------------------
    # Save Samples & Models
    # -------------------------------------------------
    if epoch % SAVE_INTERVAL == 0:
        with torch.no_grad():
            z = torch.randn(16, LATENT_DIM, 1, 1, device=device)
            samples = generator(z)
            save_image(
                (samples + 1) / 2,
                os.path.join(SAMPLE_DIR, f"epoch_{epoch:03d}.png"),
                nrow=4
            )

        torch.save(generator.state_dict(), os.path.join(MODEL_DIR, "G_final.pt"))
        torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, "D_final.pt"))

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"D_loss: {d_epoch:.4f} | "
        f"G_loss: {g_epoch:.4f}"
    )

csv_file.close()
