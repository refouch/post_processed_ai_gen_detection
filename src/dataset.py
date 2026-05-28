"""
Dataset utilities for the RRDataset.

Directory layout expected:
  dataset/
    RRDataset_original_train_val/{train,val}/{real,ai}/*.jpg   <- transformation = original
    RRDataset_test/RRDataset_final/{original,transfer,redigital}/{real,ai}/*.jpg

Labels:
  real_fake : 0 = real, 1 = ai
  transform : 0 = original, 1 = transfer, 2 = redigital
"""

import os
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

REAL_FAKE = {"real": 0, "ai": 1}
TRANSFORM = {"original": 0, "transfer": 1, "redigital": 2}

REAL_FAKE_NAMES = {v: k for k, v in REAL_FAKE.items()}
TRANSFORM_NAMES = {v: k for k, v in TRANSFORM.items()}


def _scan_train_val(root: Path) -> list[tuple[str, int, int]]:
    """Scan RRDataset_original_train_val — transformation is always 'original'."""
    samples = []
    base = root / "RRDataset_original_train_val"
    for split in ("train", "val"):
        for cls, rf_label in REAL_FAKE.items():
            folder = base / split / cls
            if not folder.exists():
                continue
            for f in folder.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    samples.append((str(f), rf_label, TRANSFORM["original"]))
    return samples


def _scan_test(root: Path) -> list[tuple[str, int, int]]:
    """Scan RRDataset_test — all three transformation types present."""
    samples = []
    base = root / "RRDataset_test" / "RRDataset_final"
    for tr_name, tr_label in TRANSFORM.items():
        for cls, rf_label in REAL_FAKE.items():
            folder = base / tr_name / cls
            if not folder.exists():
                continue
            for f in folder.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    samples.append((str(f), rf_label, tr_label))
    return samples


def build_samples(
    root: str,
    subset_per_cell: int | None = None,
    seed: int = 42,
) -> list[tuple[str, int, int]]:
    """
    Build a unified sample list from both dataset parts.

    Each "cell" is a (real_fake, transform) combination (6 total).
    subset_per_cell caps how many images are drawn from each cell; None = use all.
    """
    root = Path(root)
    all_samples = _scan_train_val(root) + _scan_test(root)

    if subset_per_cell is None:
        return all_samples

    rng = random.Random(seed)
    buckets: dict[tuple[int, int], list] = defaultdict(list)
    for s in all_samples:
        _, rf, tr = s
        buckets[(rf, tr)].append(s)

    result = []
    for key, items in buckets.items():
        rng.shuffle(items)
        result.extend(items[:subset_per_cell])
    return result


def make_splits(
    samples: list[tuple[str, int, int]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Stratified split by (real_fake, transform) cell."""
    rng = random.Random(seed)
    buckets: dict[tuple[int, int], list] = defaultdict(list)
    for s in samples:
        _, rf, tr = s
        buckets[(rf, tr)].append(s)

    train, val, test = [], [], []
    for items in buckets.values():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    return train, val, test


def get_transforms(image_size: int = 224, augment: bool = False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


class RRDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int, int]],
        image_size: int = 224,
        augment: bool = False,
    ):
        self.samples = samples
        self.transform = get_transforms(image_size, augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, rf_label, tr_label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(rf_label, dtype=torch.long), torch.tensor(tr_label, dtype=torch.long)


def print_split_stats(train, val, test) -> None:
    header = f"{'Split':<8} {'Total':>7}"
    for tr in TRANSFORM_NAMES.values():
        for rf in REAL_FAKE_NAMES.values():
            header += f"  {tr[:4]}/{rf[:4]:>4}"
    print(header)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for _, rf, tr in split:
            counts[(rf, tr)] += 1
        row = f"{name:<8} {len(split):>7}"
        for tr_label in range(3):
            for rf_label in range(2):
                row += f"  {counts[(rf_label, tr_label)]:>9}"
        print(row)
