import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import csv

# =========================
# CONFIG (Module-3)
# =========================
IMAGE_SIZE = 64
CHANNELS = 1
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 200
LR = 0.0002
BETA_1 = 0.5
SAVE_INTERVAL = 10

DATA_DIR = "data/train/images"
CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"
LOG_DIR = "logs/tensorboard"

# =========================
# DIRECTORY SETUP
# =========================
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# =========================
# AUTO DATA GENERATION
# =========================
if len(os.listdir(DATA_DIR)) == 0:
    print("⚠️ No images found. Creating dummy dataset...")
    for i in range(200):
        img = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        plt.imsave(f"{DATA_DIR}/img_{i}.png", img, cmap="gray")
    print("✅ Dummy dataset created.")

# =========================
# DATA LOADER
# =========================
datagen = ImageDataGenerator(rescale=1.0 / 127.5 - 1.0)

dataset = datagen.flow_from_directory(
    "data/train",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    color_mode="grayscale",
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=True
)

steps_per_epoch = len(dataset)

# =========================
# GENERATOR
# =========================
def build_generator():
    model = models.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(128 * 16 * 16),
        layers.ReLU(),
        layers.Reshape((16, 16, 128)),

        layers.Conv2DTranspose(128, 4, strides=2, padding="same"),
        layers.ReLU(),

        layers.Conv2DTranspose(64, 4, strides=2, padding="same"),
        layers.ReLU(),

        layers.Conv2D(CHANNELS, 7, padding="same", activation="tanh")
    ])
    return model

# =========================
# DISCRIMINATOR
# =========================
def build_discriminator():
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),

        layers.Conv2D(64, 4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),

        layers.Conv2D(128, 4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# =========================
# MODEL SETUP
# =========================
generator = build_generator()
discriminator = build_discriminator()

loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = optimizers.Adam(LR, beta_1=BETA_1)

discriminator.compile(
    loss=loss_fn,
    optimizer=optimizer
)

discriminator.trainable = False

z = layers.Input(shape=(LATENT_DIM,))
fake_img = generator(z)
validity = discriminator(fake_img)

gan = models.Model(z, validity)
gan.compile(loss=loss_fn, optimizer=optimizer)

# =========================
# LOGGING
# =========================
csv_log = open("logs/training_log.csv", "w", newline="")
csv_writer = csv.writer(csv_log)
csv_writer.writerow(["epoch", "D_loss", "G_loss"])

tb_gen = tf.summary.create_file_writer(f"{LOG_DIR}/generator")
tb_disc = tf.summary.create_file_writer(f"{LOG_DIR}/discriminator")

# =========================
# SAMPLE IMAGE SAVER
# =========================
def save_samples(epoch):
    noise = np.random.normal(0, 1, (16, LATENT_DIM))
    imgs = generator.predict(noise, verbose=0)
    imgs = (imgs + 1) / 2

    fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    idx = 0
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(imgs[idx, :, :, 0], cmap="gray")
            axs[i, j].axis("off")
            idx += 1

    fig.savefig(f"{SAMPLE_DIR}/epoch_{epoch:03d}.png")
    plt.close()

# =========================
# TRAINING LOOP
# =========================
for epoch in range(1, EPOCHS + 1):

    d_losses = []
    g_losses = []

    for _ in range(steps_per_epoch):

        real_imgs = next(dataset)
        batch_size = real_imgs.shape[0]

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        fake_imgs = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        g_loss = gan.train_on_batch(noise, real_labels)

        d_losses.append(d_loss)
        g_losses.append(g_loss)

    d_loss_epoch = np.mean(d_losses)
    g_loss_epoch = np.mean(g_losses)

    csv_writer.writerow([epoch, d_loss_epoch, g_loss_epoch])
    csv_log.flush()

    with tb_gen.as_default():
        tf.summary.scalar("G_loss", g_loss_epoch, step=epoch)

    with tb_disc.as_default():
        tf.summary.scalar("D_loss", d_loss_epoch, step=epoch)

    if epoch % SAVE_INTERVAL == 0:
        save_samples(epoch)
        generator.save(f"{CHECKPOINT_DIR}/G_epoch_{epoch:03d}.keras")
        discriminator.save(f"{CHECKPOINT_DIR}/D_epoch_{epoch:03d}.keras")

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"D_loss: {d_loss_epoch:.4f} | "
        f"G_loss: {g_loss_epoch:.4f}"
    )

csv_log.close()
