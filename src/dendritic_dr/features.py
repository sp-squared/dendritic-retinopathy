"""Loading cached backbone features (shared by training and analysis scripts)."""
from __future__ import annotations

import numpy as np
import torch

from .utils import DATA_DIR

SPLITS = ("train", "val", "test")


def smoke_features(seed: int = 0, dim: int = 512) -> dict:
    """Random features with a weak grade signal, for offline end-to-end checks."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=dim).astype(np.float32)
    out = {}
    for split, n in zip(SPLITS, (600, 150, 150)):
        y = rng.choice(5, size=n, p=[0.45, 0.12, 0.25, 0.08, 0.10])
        X = rng.normal(size=(n, dim)).astype(np.float32) + 0.15 * y[:, None] * direction
        out[f"X_{split}"], out[f"y_{split}"] = X, y
        out[f"ids_{split}"] = np.array([f"smoke_{split}_{i:04d}" for i in range(n)])
    return out


def load_features(name: str, smoke: bool = False) -> dict:
    if smoke:
        return smoke_features()
    return dict(np.load(DATA_DIR / "features" / f"{name}.npz", allow_pickle=True))


def standardization(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mean/SD from the training split only (no val/test leakage)."""
    return d["X_train"].mean(0), d["X_train"].std(0) + 1e-6


def to_tensors(d: dict, device, mu=None, sd=None) -> dict:
    if mu is None or sd is None:
        mu, sd = standardization(d)
    return {
        s: (
            torch.tensor((d[f"X_{s}"] - mu) / sd, dtype=torch.float32, device=device),
            torch.tensor(d[f"y_{s}"], dtype=torch.long, device=device),
        )
        for s in SPLITS
    }
