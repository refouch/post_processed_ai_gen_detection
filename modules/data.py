"""Data preprocesssing and DataLoader objects"""

from torch.utils.data import Dataset
from PIL import Image
import torch

def collate_fn(batch):
    """Collate that keeps fields (transform, context) as lists and not strings
        Avoids crashing the DataLoader"""
    
    imgs = torch.stack([b[0] for b in batch])
    y_rf = torch.tensor([b[1] for b in batch], dtype=torch.long)
    y_tf = torch.tensor([b[2] for b in batch], dtype=torch.long)
    transforms_ = [b[3] for b in batch] # Not to confuse with transform variable in pytorch
    contexts = [b[4] for b in batch]
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
        for _, row in sub.iterrows(): # Wide to long -> one filepath per row
            for transform in TRANSFORMS:
                rows.append((row[transform], row["label"], transform, row["context"]))
        self.samples = rows
        self.transform = apply_transform # CAUTION: confusion possible between torchvision transforms and our "transform" objetive variable!

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label, transform, context = self.samples[idx]
        img = Image.open(filepath).convert("RGB")

        if self.transform:
            img = self.transform(img)
            
        return (img, REALFAKE_MAP[label], TRANSFORM_MAP[transform], transform, context) 