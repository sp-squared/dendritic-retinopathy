"""Loading cached backbone features (shared by training and analysis scripts)."""
from __future__ import annotations

import numpy as np
import torch

from .utils import DATA_DIR

SPLITS = ("train", "val", "test")


def smoke_features(seed: int = 0, dim: int = 512, spatial: bool = False, spread: int = 0) -> dict:
    """Random features with a weak grade signal, for offline end-to-end checks.
    spatial=True gives (N, 32, 7, 7) maps whose signal sits in local regions; `spread`
    widens those regions (used to mimic later ResNet stages in the sweep smoke test)."""
    rng = np.random.default_rng(seed)
    shape = (32, 7, 7) if spatial else (dim,)
    direction = np.zeros(shape, np.float32)
    if spatial:  # signal on a subset of channels, in two regions that widen with `spread`
        a0, a1 = max(0, 1 - spread), min(7, 3 + spread)
        b0, b1 = max(0, 4 - spread), min(7, 6 + spread)
        direction[:8, a0:a1, a0:a1] = rng.normal(size=(8, a1 - a0, a1 - a0))
        direction[8:16, b0:b1, b0:b1] = rng.normal(size=(8, b1 - b0, b1 - b0))
        direction *= 3.0 * 2 / (a1 - a0)  # keep total signal roughly constant as it spreads
    else:
        direction[:] = rng.normal(size=shape)
    out = {}
    for split, n in zip(SPLITS, (600, 150, 150)):
        y = rng.choice(5, size=n, p=[0.45, 0.12, 0.25, 0.08, 0.10])
        X = rng.normal(size=(n, *shape)).astype(np.float32) + 0.15 * y.reshape(-1, *[1] * len(shape)) * direction
        out[f"X_{split}"], out[f"y_{split}"] = X, y
        out[f"ids_{split}"] = np.array([f"smoke_{split}_{i:04d}" for i in range(n)])
    return out


def smoke_pixels(seed: int = 0, size: int = 16) -> dict:
    """Synthetic single-channel 'images' (N, 1, size, size) whose grade signal sits in two
    small bright spots, mimicking local lesions, for offline checks of the raw-pixel step."""
    rng = np.random.default_rng(seed)
    direction = np.zeros((1, size, size), np.float32)
    for y0, x0 in [(3, 3), (size - 6, size - 7)]:
        direction[0, y0 : y0 + 3, x0 : x0 + 3] = 2.0
    out = {}
    for split, n in zip(SPLITS, (600, 150, 150)):
        y = rng.choice(5, size=n, p=[0.45, 0.12, 0.25, 0.08, 0.10])
        X = rng.normal(size=(n, 1, size, size)).astype(np.float32) + 0.3 * y.reshape(-1, 1, 1, 1) * direction
        out[f"X_{split}"], out[f"y_{split}"] = X, y
        out[f"ids_{split}"] = np.array([f"smoke_{split}_{i:04d}" for i in range(n)])
    return out


def load_features(name: str, smoke: bool = False) -> dict:
    """Load cached features (or raw pixels from step 12, stored as uint8).
    Spatial feature maps and images (N, C, H, W) are flattened to
    (N, C*H*W) and their shape is returned under 'image_shape' (None for vectors)."""
    if smoke:
        import re

        m = re.search(r"_l(\d)", name)
        if "pixels" in name:
            d = smoke_pixels()
        else:
            d = smoke_features(spatial="spatial" in name, spread=int(m.group(1)) - 1 if m else 0)
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
