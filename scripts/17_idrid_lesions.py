"""Step 17: do individual dendrites respond to individual lesions? (IDRiD lesion masks)

The end-to-end CNNs of step 15 were trained on APTOS grades only. IDRiD (Porwal et al.,
2018) provides 81 fundus images with pixel-level masks for four DR lesion types:
microaneurysms (MA), haemorrhages (HE), hard exudates (EX) and soft exudates (SE), plus
the optic disc (OD). Without any retraining, this script runs every trained CNN on the
IDRiD images and asks how well the first-layer units pick out lesions:

  units      somas (dendritic CNNs) or channels (dense CNNs): 16 per model, comparable
             across first-layer variants; and, for dendritic CNNs, the 128 dendrites
  locations  the stride-4 grid of the first layer (one cell = 4 x 4 px at 512 px);
             a cell is lesion-positive if its central 4 x 4 block contains that lesion,
             lesion-free if no lesion lies within one cell of it; cells outside the
             fundus or inside the optic disc are left out
  AUC        per unit and lesion type: activation at lesion cells vs. lesion-free cells
             (pooled over all 81 images); 0.5 = no preference
  lesions    every connected lesion component on the grid counts as one lesion; a unit
             detects it if its peak activation inside the component exceeds the unit's
             99th percentile over lesion-free cells of the same image. The same
             components moved to random lesion-free places give the chance rate
             ("pseudo-lesions").
  baselines  local green intensity and local contrast (green minus a 31 x 31 blur) as
             "units", to tell lesion detectors from plain brightness detectors

Prediction (stated before running): clustered (LRF) dendrites, each confined to a 5 x 5 px
patch, are more lesion-selective than scattered dendrites, most for small lesions (MA, HE):
higher best-dendrite AUC and a larger share of dendrites with AUC >= 0.75.

Data: download "A. Segmentation.zip" (free IEEE account) from
https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
and unzip it under data/raw/idrid/ (any folder layout; files are found by name:
IDRiD_xx.jpg and IDRiD_xx_MA.tif, _HE, _EX, _SE, _OD).

python scripts/17_idrid_lesions.py
python scripts/17_idrid_lesions.py --smoke        # synthetic images + smoke CNNs (step 15 --smoke)

Outputs (results/analysis/): idrid_unit_auc.csv, idrid_detection.csv, idrid_summary.csv,
idrid_tests.csv, fig_idrid.png, fig_idrid_example.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage, stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dendritic_dr.analysis import bh_fdr
from dendritic_dr.cnn import DendriticConv2d, build_cnn
from dendritic_dr.utils import DATA_DIR, get_device, results_root

LESIONS = ["MA", "HE", "EX", "SE"]
NAMES = {"MA": "microaneurysms", "HE": "haemorrhages", "EX": "hard exudates", "SE": "soft exudates"}
STEM_ORDER = [("dend-lrf", 12), ("dend-r", 12), ("conv", 12), ("conv", 5)]
STEM_LABEL = {("dend-lrf", 12): "dendritic, clustered", ("dend-r", 12): "dendritic, scattered",
              ("conv", 12): "dense 12x12", ("conv", 5): "dense 5x5"}
COLORS = {"dend-lrf": "#0072B2", "dend-r": "#E69F00", "conv": "#009E73"}
GRID_STRIDE = 4


# ---------------------------------------------------------------- data

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


# ---------------------------------------------------------------- grid helpers

def to_grid(mask: np.ndarray, shape) -> np.ndarray:
    """Boolean pixel mask -> grid of cells whose central 4 x 4 block contains it."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    g = F.max_pool2d(t, GRID_STRIDE, GRID_STRIDE, padding=GRID_STRIDE // 2)[0, 0].numpy() > 0
    return g[: shape[0], : shape[1]]


def grid_mean(img: np.ndarray, shape, kernel=12) -> np.ndarray:
    t = torch.from_numpy(img.astype(np.float32))[None, None]
    g = F.avg_pool2d(t, kernel, GRID_STRIDE, padding=kernel // 2, count_include_pad=False)[0, 0].numpy()
    return g[: shape[0], : shape[1]]


# ---------------------------------------------------------------- models

def load_cnn(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = build_cnn(a["stem"], in_ch=1, somas=a["somas"], dendrites=a["dendrites"], synapses=a["synapses"],
                      kernel=a["kernel"], cluster=a["cluster"], stride=a["stride"], seed=a["seed"],
                      dropout=a["dropout"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), a


@torch.no_grad()
def stem_maps(model, x):
    """(unit maps, dendrite maps or None), each (n_units, H, W) for one image x (1, 1, S, S)."""
    s = model.stem
    if isinstance(s, DendriticConv2d):
        d = F.relu(F.conv2d(x, s.weight * s.mask, s.bias, stride=s.stride, padding=s.padding))
        return F.relu(s.soma(d))[0].float().cpu().numpy(), d[0].float().cpu().numpy()
    return s(x)[0].float().cpu().numpy(), None


# ---------------------------------------------------------------- analysis

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


def components(grid_mask):
    lab, n = ndimage.label(grid_mask, structure=np.ones((3, 3)))
    return [np.argwhere(lab == k) for k in range(1, n + 1)]


def pseudo(comp, free, rng, tries=50):
    """The same component shape moved to a random place where every cell is lesion-free."""
    H, W = free.shape
    span = comp.max(0) - comp.min(0)
    for _ in range(tries):
        oy, ox = rng.integers(0, H - span[0]), rng.integers(0, W - span[1])
        moved = comp - comp.min(0) + [oy, ox]
        if free[moved[:, 0], moved[:, 1]].all():
            return moved
    return None


def detection(acts, comps_per_image, neg, rng, q=99):
    """Per unit: share of lesion components (and of their pseudo-lesions) whose peak
    activation exceeds the unit's q-th percentile over lesion-free cells of that image."""
    hit, hit_p, n, n_p = 0, 0, 0, 0
    for a, comps, ng in zip(acts, comps_per_image, neg):
        ng = fit(ng, a)
        comps = [c for c in comps if (c[:, 0] < a.shape[1]).all() and (c[:, 1] < a.shape[2]).all()]
        if not comps or not ng.any():
            continue
        thr = np.percentile(a[:, ng], q, axis=1)
        for c in comps:
            hit = hit + (a[:, c[:, 0], c[:, 1]].max(1) > thr)
            n += 1
            pc = pseudo(c, ng, rng)
            if pc is not None:
                hit_p = hit_p + (a[:, pc[:, 0], pc[:, 1]].max(1) > thr)
                n_p += 1
    if n == 0:
        return None
    return hit / n, (hit_p / n_p if n_p else np.full_like(hit / n, np.nan)), n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--idrid-dir", default=str(DATA_DIR / "raw" / "idrid"))
    p.add_argument("--pixels", default=str(DATA_DIR / "features" / "pixels_green_512.npz"),
                   help="APTOS 512 px pixels: their training-split mean/SD normalise the IDRiD input as in training")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    root = results_root(args.smoke)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    rng = np.random.default_rng(0)

    # --- images and masks
    if args.smoke:
        items, size = smoke_idrid(), 64
        mean, sd = 100.0, 25.0
    else:
        found = find_idrid(Path(args.idrid_dir))
        if not found:
            raise SystemExit(f"no IDRiD images with *_MA.tif etc. masks under {args.idrid_dir}; see the docstring")
        size = 512
        items = {k: prep_pair(j, m, size) for k, (j, m) in tqdm(found.items(), desc="IDRiD images", leave=False)}
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location("train_cnn", Path(__file__).with_name("15_train_cnn.py"))
        tc = module_from_spec(spec)
        spec.loader.exec_module(tc)
        mean, sd = tc.pixel_stats(np.load(args.pixels, allow_pickle=True)["X_train"])
    print(f"{len(items)} images; lesion masks per type: "
          + ", ".join(f"{t} {sum(m[t].any() for _, m in items.values())}" for t in LESIONS))

    runs = pd.read_csv(root / "cnn_runs.csv").drop_duplicates("run_id", keep="last")
    shape = (size // GRID_STRIDE + 1, size // GRID_STRIDE + 1)  # largest first-layer map (12 x 12 kernels)

    # --- per image: grids of lesions, lesion-free and fundus cells
    ids = list(items)
    grids = {}
    for k in ids:
        g, m = items[k]
        fundus = grid_mean((g > 7).astype(np.float32), shape, kernel=GRID_STRIDE) > 0.9
        od = ndimage.binary_dilation(to_grid(m["OD"], shape), iterations=2)
        les = {t: to_grid(m[t], shape) & fundus & ~od for t in LESIONS}
        any_les = ndimage.binary_dilation(np.logical_or.reduce([to_grid(m[t], shape) for t in LESIONS]), iterations=1)
        grids[k] = {"pos": les, "neg": fundus & ~od & ~any_les, "comps": {t: components(les[t]) for t in LESIONS}}

    # --- baselines: local intensity and local contrast as single "units"
    base = {}
    for k in ids:
        g = items[k][0].astype(np.float32)
        blur = ndimage.uniform_filter(g, size=31)
        base[k] = np.stack([grid_mean(g, shape), grid_mean(g - blur, shape)])
    rows, det_rows = [], []
    for t in LESIONS:
        pos = [grids[k]["pos"][t] for k in ids]
        neg = [grids[k]["neg"] for k in ids]
        auc = unit_aucs([base[k] for k in ids], pos, neg, rng)
        for name, a in zip(["intensity", "contrast"], auc):
            rows.append({"model": f"baseline:{name}", "stem": "baseline", "kernel": 0, "seed": -1,
                         "level": "unit", "unit": 0, "lesion": t, "auc": a, "selectivity": abs(a - 0.5) * 2})

    # --- every trained CNN
    for r in tqdm(list(runs.itertuples()), desc="models"):
        model, a = load_cnn(root / "models" / f"{r.run_id}.pt", device)
        units, dends = [], []
        for k in ids:
            x = (torch.from_numpy(items[k][0]).float().to(device)[None, None] - mean) / sd
            u, d = stem_maps(model, x)
            units.append(u)
            dends.append(d)
        for t in LESIONS:
            pos = [grids[k]["pos"][t] for k in ids]
            neg = [grids[k]["neg"] for k in ids]
            levels = [("unit", units)] + ([("dendrite", dends)] if dends[0] is not None else [])
            for level, acts in levels:
                for i, au in enumerate(unit_aucs(acts, pos, neg, rng)):
                    rows.append({"model": r.config, "stem": r.stem, "kernel": int(r.kernel), "seed": int(r.seed),
                                 "level": level, "unit": i, "lesion": t, "auc": au, "selectivity": abs(au - 0.5) * 2})
            det = detection(units, [grids[k]["comps"][t] for k in ids], [grids[k]["neg"] for k in ids], rng)
            if det is not None:
                rate, prate, n = det
                for i in range(len(rate)):
                    det_rows.append({"model": r.config, "stem": r.stem, "kernel": int(r.kernel), "seed": int(r.seed),
                                     "unit": i, "lesion": t, "n_lesions": n, "detect": rate[i], "pseudo": prate[i]})
    auc_df, det_df = pd.DataFrame(rows), pd.DataFrame(det_rows)
    auc_df.to_csv(out_dir / "idrid_unit_auc.csv", index=False)
    det_df.to_csv(out_dir / "idrid_detection.csv", index=False)

    # --- per-model summary: best unit, share of selective units
    m = auc_df[auc_df["stem"] != "baseline"]
    summ = (m.groupby(["stem", "kernel", "seed", "level", "lesion"])
            .agg(best_auc=("auc", "max"), best_selectivity=("selectivity", "max"),
                 frac_auc75=("auc", lambda v: float((v >= 0.75).mean())), n_units=("auc", "size")).reset_index())
    if not det_df.empty:
        d = det_df.assign(excess=det_df["detect"] - det_df["pseudo"])
        best = d.loc[d.groupby(["stem", "kernel", "seed", "lesion"])["excess"].idxmax()]
        best = best[["stem", "kernel", "seed", "lesion", "detect", "pseudo", "n_lesions"]].assign(level="unit")
        summ = summ.merge(best, on=["stem", "kernel", "seed", "lesion", "level"], how="left")
    summ.to_csv(out_dir / "idrid_summary.csv", index=False)

    # --- tests over seeds (paired by seed), BH across all
    tests = []

    def paired(level, metric, ka, kb, t):
        A = summ[(summ.stem == ka[0]) & (summ.kernel == ka[1]) & (summ.level == level) & (summ.lesion == t)]
        B = summ[(summ.stem == kb[0]) & (summ.kernel == kb[1]) & (summ.level == level) & (summ.lesion == t)]
        j = A.set_index("seed")[metric].to_frame("a").join(B.set_index("seed")[metric].to_frame("b"), how="inner").dropna()
        if len(j) < 2:
            return
        dd = j["a"] - j["b"]
        pval = float(stats.wilcoxon(j["a"], j["b"]).pvalue) if (dd != 0).any() else 1.0
        tests.append({"level": level, "metric": metric, "lesion": t, "a": STEM_LABEL[ka], "b": STEM_LABEL[kb],
                      "n_seeds": len(j), "mean_a": j["a"].mean(), "mean_b": j["b"].mean(),
                      "median_diff": float(dd.median()), "wilcoxon_p": pval})

    for t in LESIONS:
        paired("dendrite", "best_auc", ("dend-lrf", 12), ("dend-r", 12), t)
        paired("dendrite", "frac_auc75", ("dend-lrf", 12), ("dend-r", 12), t)
        paired("unit", "best_auc", ("dend-lrf", 12), ("dend-r", 12), t)
        paired("unit", "best_auc", ("dend-lrf", 12), ("conv", 12), t)
        paired("unit", "best_auc", ("dend-r", 12), ("conv", 12), t)
    tests = pd.DataFrame(tests)
    if not tests.empty:
        tests["q"] = bh_fdr(tests["wilcoxon_p"])
    tests.to_csv(out_dir / "idrid_tests.csv", index=False)

    bl = auc_df[auc_df["stem"] == "baseline"].pivot(index="lesion", columns="model", values="auc")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        view = (summ.groupby(["level", "stem", "kernel", "lesion"])[["best_auc", "frac_auc75"]
                + (["detect", "pseudo"] if "detect" in summ else [])].mean().round(3))
        print("\nMean over seeds (best unit / share of units with AUC >= 0.75 / best-unit lesion detection vs. chance):")
        print(view.to_string())
        print("\nBaselines (AUC):")
        print(bl.round(3).to_string())
        if not tests.empty:
            print("\nPaired tests over seeds:")
            print(tests.round(4).to_string(index=False))

    # --- figure: (a) best-unit AUC per lesion and variant, (b) dendrite level, (c) detection vs. chance
    keys = [k for k in STEM_ORDER if ((summ.stem == k[0]) & (summ.kernel == k[1])).any()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    xs = np.arange(len(LESIONS))
    for j, k in enumerate(keys):
        s = summ[(summ.stem == k[0]) & (summ.kernel == k[1]) & (summ.level == "unit")]
        g = s.groupby("lesion")["best_auc"].agg(["mean", "std"]).reindex(LESIONS)
        off = (j - (len(keys) - 1) / 2) * 0.18
        mk = "o" if k[0].startswith("dend") else ("^" if k[1] == 12 else "v")
        axes[0].errorbar(xs + off, g["mean"], yerr=g["std"], fmt=mk, color=COLORS[k[0]], capsize=3,
                         label=STEM_LABEL[k], mfc="white" if k == ("conv", 5) else COLORS[k[0]])
    for name, ls in [("baseline:intensity", ":"), ("baseline:contrast", "--")]:
        if name in bl:
            v = bl[name].reindex(LESIONS).to_numpy()
            v = np.maximum(v, 1 - v)
            axes[0].hlines(v, xs - 0.4, xs + 0.4, colors="#555555", linestyles=ls, lw=1,
                           label=f"{name.split(':')[1]} baseline (either sign)")
    axes[0].set_xticks(xs, LESIONS)
    axes[0].set_ylabel("Best unit AUC (mean ± SD over seeds)")
    axes[0].set_title("(a) somas / channels vs. lesion-free fundus", fontsize=9)
    axes[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    axes[0].grid(alpha=0.3, axis="y")

    for j, k in enumerate([("dend-lrf", 12), ("dend-r", 12)]):
        s = summ[(summ.stem == k[0]) & (summ.kernel == k[1]) & (summ.level == "dendrite")]
        if s.empty:
            continue
        g = s.groupby("lesion")["best_auc"].agg(["mean", "std"]).reindex(LESIONS)
        f = s.groupby("lesion")["frac_auc75"].mean().reindex(LESIONS)
        axes[1].errorbar(xs + (j - 0.5) * 0.2, g["mean"], yerr=g["std"], fmt="o", color=COLORS[k[0]], capsize=3,
                         label=STEM_LABEL[k])
        for xi, yi, fi in zip(xs + (j - 0.5) * 0.2, g["mean"], f):
            axes[1].annotate(f"{fi:.0%}", (xi, yi), xytext=(0, -12), textcoords="offset points", ha="center",
                             fontsize=6, color=COLORS[k[0]])
    axes[1].set_xticks(xs, LESIONS)
    axes[1].set_ylabel("Best dendrite AUC (labels: share with AUC ≥ 0.75)")
    axes[1].set_title("(b) individual dendrites", fontsize=9)
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    axes[1].grid(alpha=0.3, axis="y")

    if "detect" in summ:
        for j, k in enumerate(keys):
            s = summ[(summ.stem == k[0]) & (summ.kernel == k[1]) & (summ.level == "unit")].groupby("lesion")[
                ["detect", "pseudo"]].mean().reindex(LESIONS)
            off = (j - (len(keys) - 1) / 2) * 0.18
            axes[2].bar(xs + off, s["detect"], width=0.17, color=COLORS[k[0]],
                        alpha=1.0 if k != ("conv", 5) else 0.45, label=STEM_LABEL[k])
            axes[2].scatter(xs + off, s["pseudo"], marker="_", s=60, color="black", zorder=3,
                            label="same unit, pseudo-lesions" if j == 0 else None)
        axes[2].set_xticks(xs, LESIONS)
        axes[2].set_ylabel("Share of lesions detected (best unit)")
        axes[2].set_title("(c) individual lesions vs. chance", fontsize=9)
        axes[2].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        axes[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_idrid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- example: the most microaneurysm-rich image, best MA soma of one clustered-dendrite model
    ex = summ[(summ.stem == "dend-lrf") & (summ.level == "unit") & (summ.lesion == "MA")].sort_values("best_auc")
    if len(ex):
        seed = int(ex.iloc[-1]["seed"])
        rid = runs[(runs.stem == "dend-lrf") & (runs.seed == seed)].iloc[0]
        model, _ = load_cnn(root / "models" / f"{rid.run_id}.pt", device)
        ua = auc_df[(auc_df.model == rid.config) & (auc_df.seed == seed) & (auc_df.level == "unit") & (auc_df.lesion == "MA")]
        u = int(ua.loc[ua["auc"].idxmax(), "unit"])
        k = max(ids, key=lambda i: grids[i]["pos"]["MA"].sum())
        x = (torch.from_numpy(items[k][0]).float().to(device)[None, None] - mean) / sd
        amap = np.pad(stem_maps(model, x)[0][u], ((0, 1), (0, 1)))
        cy, cx = np.argwhere(grids[k]["pos"]["MA"]).mean(0).astype(int)
        h = min(shape[0] // 4, 32)
        y0, x0 = np.clip([cy - h, cx - h], 0, np.array(shape) - 2 * h)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        g = items[k][0][y0 * 4:(y0 + 2 * h) * 4, x0 * 4:(x0 + 2 * h) * 4]
        ax[0].imshow(g, cmap="gray")
        ax[0].contour(items[k][1]["MA"][y0 * 4:(y0 + 2 * h) * 4, x0 * 4:(x0 + 2 * h) * 4], levels=[0.5],
                      colors="#E69F00", linewidths=0.8)
        ax[0].set_title(f"{k}: green channel, microaneurysms outlined", fontsize=8)
        ax[1].imshow(amap[y0:y0 + 2 * h, x0:x0 + 2 * h], cmap="magma")
        ax[1].contour(grids[k]["pos"]["MA"][y0:y0 + 2 * h, x0:x0 + 2 * h], levels=[0.5], colors="cyan", linewidths=0.6)
        ax[1].set_title(f"soma {u} of clustered-dendrite CNN (seed {seed}), AUC {ua['auc'].max():.2f}", fontsize=8)
        for a_ in ax:
            a_.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_idrid_example.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"\nwrote idrid_unit_auc.csv, idrid_detection.csv, idrid_summary.csv, idrid_tests.csv, "
          f"fig_idrid.png, fig_idrid_example.png in {out_dir}")


if __name__ == "__main__":
    main()
