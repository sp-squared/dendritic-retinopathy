"""IDRiD lesion data (Porwal et al., 2018) and lesion-selectivity helpers.

Shared by scripts 17 (selectivity of grade-trained CNNs) and 18 (CNNs trained on lesion
masks). Images and masks get the same crop / pad / resize as APTOS in step 2, and lesion
masks are mapped onto the stride-4 grid of the first CNN layer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from .cnn import DendriticConv2d

LESIONS = ["MA", "HE", "EX", "SE"]
NAMES = {"MA": "microaneurysms", "HE": "haemorrhages", "EX": "hard exudates", "SE": "soft exudates"}
GRID_STRIDE = 4


def find_idrid(root: Path):
    """{image id: (jpg path, {lesion: tif path})} for images that have lesion masks."""
    jpgs = {p.stem: p for p in root.rglob("*.jpg")}
    out = {}
    for t in LESIONS + ["OD"]:
        for m in root.rglob(f"*_{t}.tif"):
            stem = m.stem[: -len(t) - 1]
            if stem in jpgs:
                out.setdefault(stem, (jpgs[stem], {}))[1][t] = m
    return dict(sorted(out.items()))


def prep_pair(img_path, mask_paths, size, threshold=7):
    """Crop the black border, pad to square and resize, exactly as step 2 does for APTOS,
    applying the same geometry to every mask. Returns green uint8 (size, size) and
    boolean masks (size, size) per lesion type (all-False if the type has no mask)."""
    img = Image.open(img_path).convert("RGB")
    gray = np.asarray(img.convert("L"))
    rows, cols = np.where(gray.max(1) > threshold)[0], np.where(gray.max(0) > threshold)[0]
    box = (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)
    w, h = box[2] - box[0], box[3] - box[1]
    s = max(w, h)
    off = ((s - w) // 2, (s - h) // 2)

    def square(im, mode, fill):
        canvas = Image.new(mode, (s, s), fill)
        canvas.paste(im.crop(box), off)
        return canvas

    green = np.asarray(square(img, "RGB", (0, 0, 0)).resize((size, size), Image.Resampling.LANCZOS))[..., 1]
    masks = {}
    for t in LESIONS + ["OD"]:
        if t in mask_paths:
            m = Image.fromarray((np.asarray(Image.open(mask_paths[t]).convert("L")) > 0).astype(np.uint8) * 255)
            if m.size != img.size:
                m = m.resize(img.size, Image.Resampling.NEAREST)
            masks[t] = np.asarray(square(m, "L", 0).resize((size, size), Image.Resampling.BOX)) > 0
        else:
            masks[t] = np.zeros((size, size), bool)
    return green, masks


def smoke_idrid(n=12, size=64, seed=1):
    """Synthetic images with dark small 'MA'/'HE' spots and bright 'EX'/'SE' blobs, plus masks."""
    rng = np.random.default_rng(seed)
    items = {}
    for i in range(n):
        g = rng.normal(100, 15, (size, size))
        masks = {t: np.zeros((size, size), bool) for t in LESIONS + ["OD"]}
        for t, r, delta, k in [("MA", 1, -50, 6), ("HE", 2, -40, 3), ("EX", 2, 70, 4), ("SE", 3, 40, 2)]:
            for _ in range(k):
                cy, cx = rng.integers(r + 2, size - r - 2, size=2)
                g[cy - r:cy + r + 1, cx - r:cx + r + 1] += delta
                masks[t][cy - r:cy + r + 1, cx - r:cx + r + 1] = True
        items[f"smoke_{i:02d}"] = (np.clip(g, 0, 255).astype(np.uint8), masks)
    return items


def to_grid(mask: np.ndarray, shape) -> np.ndarray:
    """Boolean pixel mask -> grid of cells whose central 4 x 4 block contains it."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    g = F.max_pool2d(t, GRID_STRIDE, GRID_STRIDE, padding=GRID_STRIDE // 2)[0, 0].numpy() > 0
    return g[: shape[0], : shape[1]]


def grid_mean(img: np.ndarray, shape, kernel=12) -> np.ndarray:
    t = torch.from_numpy(img.astype(np.float32))[None, None]
    g = F.avg_pool2d(t, kernel, GRID_STRIDE, padding=kernel // 2, count_include_pad=False)[0, 0].numpy()
    return g[: shape[0], : shape[1]]


@torch.no_grad()
def stem_maps(model, x):
    """(unit maps, dendrite maps or None), each (n_units, H, W) for one image x (1, 1, S, S)."""
    s = model.stem
    if isinstance(s, DendriticConv2d):
        d = F.relu(F.conv2d(x, s.weight * s.mask, s.bias, stride=s.stride, padding=s.padding))
        return F.relu(s.soma(d))[0].float().cpu().numpy(), d[0].float().cpu().numpy()
    return s(x)[0].float().cpu().numpy(), None


def fit(grid, a):
    """Crop a grid to an activation map's (H, W); 5 x 5 and 12 x 12 layers differ by one cell
    at the bottom/right edge but share cell centres."""
    return grid[: a.shape[-2], : a.shape[-1]]


def unit_aucs(acts, pos, neg, rng, max_neg=50_000):
    """AUC per unit; acts: list over images of (n_units, H, W); pos/neg: lists of bool grids."""
    pos = [fit(p, a) for a, p in zip(acts, pos)]
    neg = [fit(n, a) for a, n in zip(acts, neg)]
    P = np.concatenate([a[:, p] for a, p in zip(acts, pos)], axis=1)
    N = np.concatenate([a[:, n] for a, n in zip(acts, neg)], axis=1)
    if P.shape[1] == 0 or N.shape[1] == 0:
        return np.full(acts[0].shape[0], np.nan)
    if N.shape[1] > max_neg:
        N = N[:, rng.choice(N.shape[1], max_neg, replace=False)]
    y = np.r_[np.ones(P.shape[1]), np.zeros(N.shape[1])]
    out = []
    for u in range(P.shape[0]):
        v = np.r_[P[u], N[u]]
        out.append(roc_auc_score(y, v) if v.std() > 0 else 0.5)
    return np.array(out)


def idrid_split(path: Path) -> str:
    """'train' or 'test' from the IDRiD folder names (a. Training Set / b. Testing Set)."""
    return "test" if "test" in str(path).lower() else "train"


def load_idrid(root: Path, cache: Path, size: int = 512) -> dict:
    """Preprocessed IDRiD lesion set, cached as .npz: ids, split ('train'/'test'),
    green (N, size, size) uint8 and masks (N, 5, size, size) bool in LESIONS + ['OD'] order."""
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    found = find_idrid(Path(root))
    if not found:
        raise SystemExit(f"no IDRiD images with *_MA.tif etc. masks under {root}; see data/README.md")
    ids, split, green, masks = [], [], [], []
    for k, (jpg, mpaths) in found.items():
        g, m = prep_pair(jpg, mpaths, size)
        ids.append(k)
        split.append(idrid_split(jpg))
        green.append(g)
        masks.append(np.stack([m[t] for t in LESIONS + ["OD"]]))
    out = {"ids": np.array(ids), "split": np.array(split), "green": np.stack(green), "masks": np.stack(masks)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def cell_grids(green: np.ndarray, masks: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For one image: lesion grids (4, H, W), valid cells (inside the fundus, outside the
    dilated optic disc) and lesion-free valid cells (no lesion within one cell)."""
    fundus = grid_mean((green > 7).astype(np.float32), shape, kernel=GRID_STRIDE) > 0.9
    od = ndimage.binary_dilation(to_grid(masks[4], shape), iterations=2)
    valid = fundus & ~od
    les = np.stack([to_grid(masks[i], shape) for i in range(4)])
    free = valid & ~ndimage.binary_dilation(les.any(0), iterations=1)
    return les & valid, valid, free
