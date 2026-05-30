"""Data preprocesssing and DataLoader objects"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
import os

def collate_fn(batch):
    """Collate that keeps string fields (transform, context) as lists."""
    imgs      = torch.stack([b[0] for b in batch])
    y_rf      = torch.tensor([b[1] for b in batch], dtype=torch.long)
    y_tf      = torch.tensor([b[2] for b in batch], dtype=torch.long)
    transforms_ = [b[3] for b in batch]
    contexts  = [b[4] for b in batch]
    return imgs, y_rf, y_tf, transforms_, contexts


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
                rows.append((row[transform], row["label"], transform, row["context"]))
        self.samples = rows
        self.transform = apply_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label, transform, context = self.samples[idx]
        img = Image.open(filepath).convert("RGB")

        if self.transform:
            img = self.transform(img)
            
        return (img,
                REALFAKE_MAP[label],
                TRANSFORM_MAP[transform],
                transform,          # str, pour le breakdown phase 3
                context)            # str, pour le breakdown phase 4