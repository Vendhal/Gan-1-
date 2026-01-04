import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NPYDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [
            f for f in os.listdir(data_dir)
            if f.endswith(".npy")
        ]

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npy files found in {data_dir}. "
                "Run preprocessing first."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.data_dir, self.files[idx]))
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)  # (1, 64, 64)
        return img


def get_dataloader(data_dir, batch_size=32):
    dataset = NPYDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )



