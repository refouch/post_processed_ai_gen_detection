"""Data preprocesssing and DataLoader objects"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

TRANSFORMS    = ["original", "transfer", "redigital"]
REALFAKE_MAP  = {"real": 0, "ai": 1}
TRANSFORM_MAP = {"original": 0, "transfer": 1, "redigital": 2}

class RRDataset(Dataset):
    """Adapting the pytorch dataset to take into input our custom csv pointing to each file,
        as the normal ImageDataset won't work here (splits not in the same folder)"""

    def __init__(self, wide_df, split, apply_transform=None):
        sub = wide_df[wide_df.split == split]
        rows = []
        for _, row in sub.iterrows():
            for transform in TRANSFORMS:
                rows.append((row[transform], row["label"], transform))
        self.samples = rows
        self.transform = apply_transform  # CAUTION: different from the dataset "transform" column

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label, transform = self.samples[idx]
        img = Image.open(filepath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y_label     = REALFAKE_MAP[label]
        y_transform = TRANSFORM_MAP[transform]
        return img, y_label, y_transform