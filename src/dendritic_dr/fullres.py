"""Near-native-resolution lesion training (step 4i): IDRiD + DDR data, patches, Tversky loss, FROC.

Step 18 trained on 512 px images, where a microaneurysm is one to three pixels wide. Here the
fundus is resized to a 2048 px diameter (about DDR's native scale, 0.6x IDRiD's), so a
microaneurysm spans roughly 4-12 px, and the network is trained on random patches.

* Data: IDRiD (Porwal et al., 2018; 54 training / 27 test images) and DDR (Li et al., 2019;
  383 training / 149 validation / 225 test images with MA, HE, EX and SE masks). Each image
  is cropped, padded and resized exactly as in step 2 (``idrid.prep_pair``) and cached once
  as green channel (uint8) plus a lesion bitfield (bit i = LESIONS[i], bit 4 = optic disc).
* Patches: half of the training patches are centred near a lesion pixel (lesion type drawn
  uniformly among the types present in the image, so rare microaneurysms are seen often),
  the rest anywhere in the fundus.
* Loss: Tversky (Salehi et al., 2017) with beta > alpha weights missed lesion cells more
  than false alarms, i.e. favours recall.
* Scoring: FROC per lesion type, the standard for microaneurysm detection (Niemeijer et al.,
  2010): a candidate is a local maximum of the predicted map; it is a true positive if it
  falls on a lesion (one cell tolerance), each lesion counts once, every other candidate is
  a false positive. Score = mean sensitivity at 1/8 ... 8 false positives per image.
  Cell-level AP and AUC are kept for comparison, computed from logit histograms.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

from .idrid import GRID_STRIDE, LESIONS, find_idrid, idrid_split, prep_pair, smoke_idrid

ALL = LESIONS + ["OD"]  # bit order
FROC_RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SPLIT_WORDS = {"train": "train", "training": "train", "valid": "val", "val": "val", "validation": "val",
               "test": "test", "testing": "test"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ----------------------------------------------------------------------------- data

def split_from_path(path: Path, root: Path) -> str | None:
    """'train' / 'val' / 'test' from the first folder name that says so (DDR: train, valid
    or val, test; IDRiD: 'a. Training Set', 'b. Testing Set')."""
    for part in Path(path).relative_to(root).parts[:-1]:
        for word in re.split(r"[^a-z]+", part.lower()):
            if word in SPLIT_WORDS:
                return SPLIT_WORDS[word]
    return None


def find_ddr(root: Path) -> dict:
    """{image id: (image path, {lesion: mask path}, split)} for DDR's lesion-segmentation set.

    Works with the original release (lesion_segmentation/<split>/image/*.jpg, .../label/EX/*.tif)
    and the Hugging Face mirror (lesion_segmentation/images/<split>/*.jpg,
    .../annotations/<split>/EX/*.tif): masks are files in a folder named after the lesion."""
    root = Path(root)
    masks, images = {}, {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
            continue
        t = p.parent.name.upper()
        if t in LESIONS:
            stem = p.stem[: -len(t) - 1] if p.stem.upper().endswith(f"_{t}") else p.stem
            masks.setdefault(stem, {})[t] = p
        elif p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images[p.stem] = p
    out = {}
    for stem, m in sorted(masks.items()):
        if stem in images:
            split = split_from_path(images[stem], root) or split_from_path(next(iter(m.values())), root)
            out[stem] = (images[stem], m, split)
    return out


def pack(masks: np.ndarray) -> np.ndarray:
    """(5, S, S) bool in ALL order -> (S, S) uint8 bitfield."""
    bits = np.zeros(masks.shape[1:], np.uint8)
    for i, m in enumerate(masks):
        bits |= m.astype(np.uint8) << i
    return bits


def unpack(bits: np.ndarray) -> np.ndarray:
    """(..., S, S) uint8 bitfield -> (..., 5, S, S) bool in ALL order."""
    return ((bits[..., None, :, :] >> np.arange(len(ALL), dtype=np.uint8)[:, None, None]) & 1).astype(bool)


def build_index(idrid_dir: Path, ddr_dir: Path, size: int, cache_dir: Path, n_val_idrid: int = 10,
                n_val_ddr: int = 30) -> pd.DataFrame:
    """Preprocess every image once into cache_dir/<dataset>/<id>.npz (green, bits) and return
    the index: key, dataset, id, split ('train' / 'val' / 'test').

    IDRiD has no validation split: the same 10 training images as step 18 are held out.
    DDR's validation split has 149 images; a fixed subset of n_val_ddr is used for early
    stopping (all 149 would make every epoch's check slow at 2048 px)."""
    rows = []
    idrid = find_idrid(Path(idrid_dir)) if Path(idrid_dir).exists() else {}
    for k, (jpg, mp) in idrid.items():
        rows.append(("idrid", k, idrid_split(jpg), jpg, mp))
    ddr = find_ddr(Path(ddr_dir)) if Path(ddr_dir).exists() else {}
    for k, (img, mp, split) in ddr.items():
        if split is None:
            raise SystemExit(f"cannot tell the split of DDR image {img}; expected train / valid / test folders")
        rows.append(("ddr", k, split, img, mp))
    if not rows:
        raise SystemExit(f"no lesion data under {idrid_dir} or {ddr_dir}; see data/README.md")
    index = pd.DataFrame([r[:3] for r in rows], columns=["dataset", "id", "split"])
    index["key"] = index["dataset"] + "/" + index["id"]

    # fixed validation images (IDRiD: identical to step 18's choice)
    rng = np.random.default_rng(42)
    tr = np.flatnonzero((index["dataset"] == "idrid") & (index["split"] == "train"))
    if len(tr):
        index.loc[np.sort(rng.choice(tr, min(n_val_idrid, len(tr)), replace=False)), "split"] = "val"
    dv = np.flatnonzero((index["dataset"] == "ddr") & (index["split"] == "val"))
    if len(dv) > n_val_ddr:
        keep = np.random.default_rng(42).choice(dv, n_val_ddr, replace=False)
        index.loc[np.setdiff1d(dv, keep), "split"] = "val-unused"

    todo = [(r, p) for r, p in zip(rows, cache_paths(index, cache_dir)) if not p.exists()]
    for i, ((ds, k, _, img, mp), path) in enumerate(todo):
        if i % 50 == 0:
            print(f"  preprocessing {ds} images at {size} px: {i}/{len(todo)}", flush=True)
        green, m = prep_pair(img, mp, size)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, green=green, bits=pack(np.stack([m[t] for t in ALL])))
    return index


def cache_paths(index: pd.DataFrame, cache_dir: Path) -> list[Path]:
    return [Path(cache_dir) / ds / f"{k}.npz" for ds, k in zip(index["dataset"], index["id"])]


class Store:
    """Images by key, from the .npz cache or (smoke test) from memory."""

    def __init__(self, index: pd.DataFrame, cache_dir: Path | None = None, memory: dict | None = None):
        self.index, self.cache_dir, self.memory = index.reset_index(drop=True), cache_dir, memory

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if self.memory is not None:
            return self.memory[key]
        ds, k = key.split("/", 1)
        d = np.load(Path(self.cache_dir) / ds / f"{k}.npz")
        return d["green"], d["bits"]

    def keys(self, split: str, datasets=None) -> list[str]:
        sel = self.index["split"] == split
        if datasets is not None:
            sel &= self.index["dataset"].isin(list(datasets))
        return list(self.index.loc[sel, "key"])


def smoke_store(size: int = 128) -> Store:
    """Synthetic IDRiD- and DDR-like sets for an offline smoke test."""
    memory, rows = {}, []
    for ds, seed, splits in [("idrid", 1, ["train"] * 12 + ["val"] * 2 + ["test"] * 4),
                             ("ddr", 2, ["train"] * 12 + ["val"] * 3 + ["test"] * 5)]:
        items = smoke_idrid(n=len(splits), size=size, seed=seed)
        for (k, (g, m)), split in zip(items.items(), splits):
            memory[f"{ds}/{k}"] = (g, pack(np.stack([m[t] for t in ALL])))
            rows.append((ds, k, split, f"{ds}/{k}"))
    return Store(pd.DataFrame(rows, columns=["dataset", "id", "split", "key"]), memory=memory)


class PatchSampler:
    """Random square training patches; top-left corners on the stride-4 grid so that patch
    grids line up with full-image grids."""

    def __init__(self, greens: list, bits: list, patch: int, lesion_frac: float = 0.5, seed: int = 0,
                 max_coords: int = 4000):
        self.greens, self.bits, self.patch, self.lesion_frac = greens, bits, patch, lesion_frac
        self.rng = np.random.default_rng(seed)
        self.fundus, self.lesions = [], []
        for g, b in zip(greens, bits):
            if min(g.shape) < patch:
                raise ValueError(f"patch {patch} larger than image {g.shape}")
            self.fundus.append(self._coords(g > 7, max_coords))
            self.lesions.append({j: c for j in range(len(LESIONS))
                                 if len(c := self._coords((b >> j) & 1 > 0, max_coords))})

    def _coords(self, mask, n):
        ys, xs = np.nonzero(mask)
        if len(ys) > n:
            keep = self.rng.choice(len(ys), n, replace=False)
            ys, xs = ys[keep], xs[keep]
        return np.stack([ys, xs], 1)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """green (n, 1, P, P) uint8 and masks (n, 5, P, P) bool."""
        P = self.patch
        g_out = np.empty((n, 1, P, P), np.uint8)
        b_out = np.empty((n, P, P), np.uint8)
        for i in range(n):
            k = int(self.rng.integers(len(self.greens)))
            les = self.lesions[k]
            if les and self.rng.random() < self.lesion_frac:
                c = les[list(les)[self.rng.integers(len(les))]]
            else:
                c = self.fundus[k] if len(self.fundus[k]) else np.array([[s // 2 for s in self.greens[k].shape]])
            cy, cx = c[self.rng.integers(len(c))]
            jy, jx = self.rng.integers(-P // 4, P // 4 + 1, size=2)
            S0, S1 = self.greens[k].shape
            y0 = int(np.clip(cy - P // 2 + jy, 0, S0 - P)) // GRID_STRIDE * GRID_STRIDE
            x0 = int(np.clip(cx - P // 2 + jx, 0, S1 - P)) // GRID_STRIDE * GRID_STRIDE
            g_out[i, 0] = self.greens[k][y0:y0 + P, x0:x0 + P]
            b_out[i] = self.bits[k][y0:y0 + P, x0:x0 + P]
        return g_out, unpack(b_out)


# ----------------------------------------------------------------------------- training

def grid_targets(green_u8: torch.Tensor, masks: torch.Tensor, shape) -> tuple[torch.Tensor, torch.Tensor]:
    """Same targets as step 18 (scripts/18_train_lesions.torch_targets): lesions (B, 4, H, W)
    on the stride-4 grid and valid cells (B, 1, H, W): inside the fundus, outside the dilated
    optic disc (DDR has no optic-disc masks, so there every fundus cell is valid)."""
    s = GRID_STRIDE

    def pool(m):
        return F.max_pool2d(m.float(), s, s, padding=s // 2)[..., : shape[0], : shape[1]]

    fundus = F.avg_pool2d((green_u8 > 7).float(), s, s, padding=s // 2,
                          count_include_pad=False)[..., : shape[0], : shape[1]] > 0.9
    od = F.max_pool2d(pool(masks[:, 4:5]), 5, 1, padding=2) > 0
    valid = fundus & ~od
    return pool(masks[:, :4]) * valid, valid.float()


def augment(x: torch.Tensor, m: torch.Tensor, gen: torch.Generator):
    """Same random flips / 90-degree rotation for the patch batch and its masks."""
    if torch.rand(1, generator=gen) < 0.5:
        x, m = x.flip(-1), m.flip(-1)
    if torch.rand(1, generator=gen) < 0.5:
        x, m = x.flip(-2), m.flip(-2)
    k = int(torch.randint(0, 4, (1,), generator=gen))
    return torch.rot90(x, k, (-2, -1)), torch.rot90(m, k, (-2, -1))


def tversky_loss(logits: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, alpha: float = 0.3,
                 beta: float = 0.7, gamma: float = 1.0, eps: float = 1.0) -> torch.Tensor:
    """Mean over lesion types of (1 - Tversky index) ** gamma, pooled over the batch.

    TI = TP / (TP + alpha FP + beta FN) on soft predictions over valid cells; beta > alpha
    penalises missed lesion cells more than false alarms. gamma < 1 is the focal Tversky
    variant (Abraham & Khan, 2019). Pooling over the batch keeps patches without a given
    lesion from producing an undefined index."""
    p = torch.sigmoid(logits.float()) * valid
    y = y * valid
    dims = (0, 2, 3)
    tp = (p * y).sum(dims)
    fp = (p * (1 - y)).sum(dims)
    fn = ((1 - p) * y).sum(dims)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1 - ti).clamp(min=0) ** gamma).mean()


# ----------------------------------------------------------------------------- scoring

class Curve:
    """Cell-level AP and ROC AUC from a histogram of logits (memory stays fixed however many
    cells: DDR's test set has ~60 million). Ties within a bin share one precision value; with
    8192 bins over [-16, 16] the difference from exact AP/AUC is < 0.001 in the unit tests."""

    def __init__(self, bins: int = 8192, lim: float = 16.0):
        self.bins, self.lim = bins, lim
        self.pos = np.zeros(bins, np.int64)
        self.neg = np.zeros(bins, np.int64)

    def add(self, y: np.ndarray, logit: np.ndarray) -> None:
        idx = np.clip(((np.asarray(logit, np.float64) + self.lim) / (2 * self.lim) * self.bins).astype(np.int64),
                      0, self.bins - 1)
        y = np.asarray(y, bool)
        self.pos += np.bincount(idx[y], minlength=self.bins)
        self.neg += np.bincount(idx[~y], minlength=self.bins)

    def prevalence(self) -> float:
        n = self.pos.sum() + self.neg.sum()
        return float(self.pos.sum() / n) if n else np.nan

    def ap(self) -> float:
        P = self.pos.sum()
        if P == 0 or self.neg.sum() == 0:
            return np.nan
        pos, neg = self.pos[::-1], self.neg[::-1]
        tp, fp = np.cumsum(pos), np.cumsum(neg)
        prec = np.divide(tp, tp + fp, out=np.zeros(len(tp)), where=(tp + fp) > 0)
        return float((pos * prec).sum() / P)

    def auc(self) -> float:
        P, N = self.pos.sum(), self.neg.sum()
        if P == 0 or N == 0:
            return np.nan
        pos, neg = self.pos[::-1], self.neg[::-1]
        above = np.cumsum(pos) - pos
        return float((neg * (above + 0.5 * pos)).sum() / (P * N))


def candidates(logit: torch.Tensor, valid: torch.Tensor, k: int = 300, min_logit: float = -6.0):
    """Lesion candidates of one map: local maxima (3 x 3) on valid cells above min_logit,
    at most the k highest. Returns ys, xs, scores (numpy)."""
    mp = F.max_pool2d(logit[None, None], 3, 1, padding=1)[0, 0]
    peak = (logit == mp) & valid & (logit > min_logit)
    ys, xs = torch.nonzero(peak, as_tuple=True)
    sc = logit[ys, xs]
    if len(sc) > k:
        top = torch.topk(sc, k).indices
        ys, xs, sc = ys[top], xs[top], sc[top]
    return ys.cpu().numpy(), xs.cpu().numpy(), sc.float().cpu().numpy()


def lesion_labels(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected lesions (8-connectivity) of a boolean grid, each label grown by one cell
    into the background so that a candidate next to a lesion still hits it."""
    lab, n = ndimage.label(grid, structure=np.ones((3, 3), int))
    grown = ndimage.grey_dilation(lab, size=(3, 3))
    return np.where(lab > 0, lab, grown), n


class Froc:
    """Free-response ROC pooled over images."""

    def __init__(self):
        self.scores, self.ids = [], []
        self.n_lesions = self.n_images = 0

    def add(self, scores: np.ndarray, hits: np.ndarray, n_lesions: int) -> None:
        """hits: lesion label (1..n_lesions) per candidate, 0 = false positive."""
        self.scores.append(np.asarray(scores, np.float64))
        self.ids.append(np.where(hits > 0, hits + self.n_lesions, 0))
        self.n_lesions += n_lesions
        self.n_images += 1

    def curve(self) -> tuple[np.ndarray, np.ndarray]:
        """(false positives per image, sensitivity) after each candidate, best first."""
        if not self.scores:
            return np.zeros(1), np.zeros(1)
        s, ids = np.concatenate(self.scores), np.concatenate(self.ids)
        order = np.argsort(-s, kind="stable")
        ids = ids[order]
        first = np.zeros(len(ids), bool)
        tp_ids = np.flatnonzero(ids > 0)
        _, where = np.unique(ids[tp_ids], return_index=True)
        first[tp_ids[where]] = True  # a lesion counts once; later hits on it are ignored
        fppi = np.cumsum(ids == 0) / max(self.n_images, 1)
        sens = np.cumsum(first) / max(self.n_lesions, 1)
        return np.r_[0.0, fppi], np.r_[0.0, sens]

    def sensitivity_at(self, rate: float) -> float:
        if self.n_lesions == 0:
            return np.nan
        fppi, sens = self.curve()
        ok = fppi <= rate
        return float(sens[ok].max()) if ok.any() else 0.0

    def score(self) -> float:
        if self.n_lesions == 0:
            return np.nan
        return float(np.mean([self.sensitivity_at(r) for r in FROC_RATES]))
