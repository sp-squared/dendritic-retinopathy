"""Loading cached backbone features (shared by training and analysis scripts)."""
from __future__ import annotations

import numpy as np
import torch

from .utils import DATA_DIR

SPLITS = ("train", "val", "test")


def smoke_features(seed: int = 0, dim: int = 512, spatial: bool = False) -> dict:
    """Random features with a weak grade signal, for offline end-to-end checks.
    spatial=True gives (N, 32, 7, 7) maps whose signal sits in a few local regions."""
    rng = np.random.default_rng(seed)
    shape = (32, 7, 7) if spatial else (dim,)
    direction = np.zeros(shape, np.float32)
    if spatial:  # signal confined to two 2x2 regions, on a subset of channels
        direction[:8, 1:3, 1:3] = rng.normal(size=(8, 2, 2))
        direction[8:16, 4:6, 4:6] = rng.normal(size=(8, 2, 2))
        direction *= 3.0
    else:
        direction[:] = rng.normal(size=shape)
    out = {}
    for split, n in zip(SPLITS, (600, 150, 150)):
        y = rng.choice(5, size=n, p=[0.45, 0.12, 0.25, 0.08, 0.10])
        X = rng.normal(size=(n, *shape)).astype(np.float32) + 0.15 * y.reshape(-1, *[1] * len(shape)) * direction
        out[f"X_{split}"], out[f"y_{split}"] = X, y
        out[f"ids_{split}"] = np.array([f"smoke_{split}_{i:04d}" for i in range(n)])
    return out


def load_features(name: str, smoke: bool = False) -> dict:
    """Load cached features. Spatial feature maps (N, C, H, W) are flattened to
    (N, C*H*W) and their shape is returned under 'image_shape' (None for vectors)."""
    if smoke:
        d = smoke_features(spatial="spatial" in name)
    else:
        d = dict(np.load(DATA_DIR / "features" / f"{name}.npz", allow_pickle=True))
    d["image_shape"] = None
    if d["X_train"].ndim == 4:
        d["image_shape"] = tuple(int(v) for v in d["X_train"].shape[1:])
        for s in SPLITS:
            d[f"X_{s}"] = d[f"X_{s}"].reshape(len(d[f"X_{s}"]), -1)
    for s in SPLITS:
        d[f"X_{s}"] = d[f"X_{s}"].astype(np.float32, copy=False)
    return d


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
