"""Analysis helpers: treat hidden-unit activations like a gene-expression matrix,
plus resampling statistics for comparing models.

Conventions: an activation matrix A has shape (n_images, n_units), mirroring a
samples x genes expression matrix; y holds the DR grade (0-4) of each image.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .metrics import N_GRADES, quadratic_kappa
from .models import build_model


# --- Multiple testing --------------------------------------------------------

def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values). NaNs are passed through."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]  # enforce monotonicity
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


# --- Loading trained heads and extracting activations ------------------------

def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["model_args"])
    model = build_model(args, ckpt["in_features"], ckpt["n_out"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def hidden_activations(model, x: torch.Tensor) -> np.ndarray:
    """Output of the last hidden layer: soma activations for a dANN, hidden
    units for the vanilla MLP. Shape (n_images, n_units)."""
    model.eval()
    return model.body(x).cpu().numpy()


# --- Differential activation (the DE-analysis analogue) ----------------------

def grade_means(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(n_units, 5) mean activation of each unit for each grade (NaN if a grade is absent)."""
    out = np.full((A.shape[1], N_GRADES), np.nan)
    for g in range(N_GRADES):
        if (y == g).any():
            out[:, g] = A[y == g].mean(axis=0)
    return out


def differential_activation(A: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per-unit tests of association with DR grade.

    kruskal_p : Kruskal-Wallis H test across the grade groups (any difference)
    spearman_rho / spearman_p : monotonic trend with grade (direction of change)
    q values  : Benjamini-Hochberg across units
    Units with no variance ("dead" ReLU units) get NaN and are excluded from FDR.
    """
    n_units = A.shape[1]
    kw_p = np.full(n_units, np.nan)
    rho = np.full(n_units, np.nan)
    sp_p = np.full(n_units, np.nan)
    groups_present = [g for g in range(N_GRADES) if (y == g).sum() > 0]
    for u in range(n_units):
        a = A[:, u]
        if np.ptp(a) == 0:
            continue
        try:
            kw_p[u] = stats.kruskal(*[a[y == g] for g in groups_present]).pvalue
        except ValueError:  # all values identical within the tested groups
            pass
        r = stats.spearmanr(a, y)
        rho[u], sp_p[u] = r.statistic, r.pvalue
    gm = grade_means(A, y)
    df = pd.DataFrame(
        {
            "unit": np.arange(n_units),
            "active": np.ptp(A, axis=0) > 0,
            "mean_activation": A.mean(axis=0),
            "kruskal_p": kw_p,
            "kruskal_q": bh_fdr(kw_p),
            "spearman_rho": rho,
            "spearman_p": sp_p,
            "spearman_q": bh_fdr(sp_p),
        }
    )
    for g in range(N_GRADES):
        df[f"mean_grade{g}"] = gm[:, g]
    sel = selectivity(gm)
    df["selectivity_index"] = sel["selectivity_index"]
    df["n_grades_responsive"] = sel["n_grades_responsive"]
    df["preferred_grade"] = sel["preferred_grade"]
    return df


# --- Selectivity (tests the paper's "mixed selectivity" claim) ----------------

def selectivity(gm: np.ndarray, threshold: float = 0.5) -> dict:
    """Per-unit selectivity from grade-mean activations (ReLU outputs, >= 0).

    selectivity_index  = (max - mean of other grades) / (max + mean of other grades);
                         1 = responds to a single grade, 0 = equal response to all.
    n_grades_responsive = number of grades whose mean response is >= `threshold`
                         x the unit's maximum grade mean (1 = class-specific,
                         5 = responds to every grade, i.e. mixed selectivity).
    Units that never activate get NaN.
    """
    gm = np.nan_to_num(gm, nan=0.0)
    mx = gm.max(axis=1)
    others = (gm.sum(axis=1) - mx) / (gm.shape[1] - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        si = (mx - others) / (mx + others)
    resp = (gm >= threshold * mx[:, None]).sum(axis=1).astype(float)
    dead = mx <= 0
    si[dead] = np.nan
    resp[dead] = np.nan
    pref = gm.argmax(axis=1).astype(float)
    pref[dead] = np.nan
    return {"selectivity_index": si, "n_grades_responsive": resp, "preferred_grade": pref}


# --- Representation-level summaries ------------------------------------------

def zscore_units(A: np.ndarray) -> np.ndarray:
    """Z-score each unit across images, dropping units with zero variance."""
    sd = A.std(axis=0)
    keep = sd > 0
    return (A[:, keep] - A[:, keep].mean(axis=0)) / sd[keep]


def pca(A: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """PCA via SVD on z-scored units. Returns (scores, explained_variance_ratio)."""
    Z = zscore_units(A)
    if Z.shape[1] == 0:
        return np.zeros((A.shape[0], n_components)), np.zeros(n_components)
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    var = S**2 / (S**2).sum()
    k = min(n_components, len(S))
    return U[:, :k] * S[:k], var[:k]


def progression_score(A: np.ndarray, y: np.ndarray) -> float:
    """|Spearman rho| between PC1 and grade: how strongly the dominant axis of
    the representation follows disease severity (0 = not at all, 1 = perfectly)."""
    scores, _ = pca(A, 1)
    if np.ptp(scores[:, 0]) == 0:
        return 0.0
    return float(abs(stats.spearmanr(scores[:, 0], y).statistic))


# --- Resampling statistics for model comparison -------------------------------

def bootstrap_qwk_ci(y, pred, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for QWK, resampling test images with replacement."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = quadratic_kappa(y[idx], pred[idx])
    vals = vals[~np.isnan(vals)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def _mean_qwk(y, preds_2d, idx):
    return float(np.mean([quadratic_kappa(y[idx], p[idx]) for p in preds_2d]))


def paired_bootstrap_diff(y, preds_a, preds_b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Paired bootstrap of mean-over-seeds QWK(A) - QWK(B) on the same test images.

    preds_a, preds_b: arrays of shape (n_seeds, n_images), aligned to y.
    Returns (observed_diff, ci_low, ci_high, two_sided_p).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    preds_a, preds_b = np.atleast_2d(preds_a), np.atleast_2d(preds_b)
    n = len(y)
    full = np.arange(n)
    obs = _mean_qwk(y, preds_a, full) - _mean_qwk(y, preds_b, full)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = _mean_qwk(y, preds_a, idx) - _mean_qwk(y, preds_b, idx)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # Two-sided p: how often the bootstrap distribution crosses zero.
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(p, 1.0))
