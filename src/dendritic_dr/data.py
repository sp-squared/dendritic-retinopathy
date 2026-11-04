"""APTOS 2019 fundus data: preprocessing, splits, and a torch Dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def crop_fundus(img: Image.Image, threshold: int = 7) -> Image.Image:
    """Crop away the black border around the circular fundus region.
    Pixels darker than `threshold` (grayscale) are treated as background."""
    gray = np.asarray(img.convert("L"))
    rows = np.where(gray.max(axis=1) > threshold)[0]
    cols = np.where(gray.max(axis=0) > threshold)[0]
    if len(rows) == 0 or len(cols) == 0:  # fully dark image: leave as is
        return img
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def preprocess_image(src: Path, size: int) -> Image.Image:
    img = Image.open(src).convert("RGB")
    img = pad_to_square(crop_fundus(img))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def make_splits(labels: pd.DataFrame, seed: int, val_frac: float, test_frac: float) -> pd.DataFrame:
    """Stratified train/val/test split on the labelled APTOS training set
    (the official Kaggle test set has no public labels)."""
    trainval, test = train_test_split(
        labels, test_size=test_frac, stratify=labels["diagnosis"], random_state=seed
    )
    rel_val = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        trainval, test_size=rel_val, stratify=trainval["diagnosis"], random_state=seed
    )
    out = pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    )
    return out.sort_values("id_code").reset_index(drop=True)


class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.image_dir / f"{row['id_code']}.png").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["diagnosis"]), row["id_code"]
