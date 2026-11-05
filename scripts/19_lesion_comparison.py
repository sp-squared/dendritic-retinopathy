"""Step 19: compare lesion-trained first layers (step 18) and contrast them with grade-trained ones.

1. Lesion detection on the 27 IDRiD test images, per readout depth: test AP per lesion type
   and mean AP, paired by seed: clustered vs. scattered dendrites, each vs. the dense 12 x 12
   layer, clustered vs. dense 5 x 5 (Wilcoxon signed-rank, BH within each depth).
2. Did the units become lesion detectors? Best first-layer unit / dendrite AUC and share of
   dendrites with AUC >= 0.75 at depth 0, clustered vs. scattered (Wilcoxon, BH).
3. Reference: the same unit measures for the grade-trained CNNs of step 15 on the same test
   images (lesion- vs. grade-trained, Mann-Whitney, BH).

python scripts/19_lesion_comparison.py
python scripts/19_lesion_comparison.py --smoke

Outputs (results/analysis/): lesion_by_config.csv, lesion_tests.csv, lesion_grade_reference.csv,
fig_lesions.png
"""
from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from dendritic_dr.analysis import bh_fdr
from dendritic_dr.cnn import build_cnn
from dendritic_dr.idrid import LESIONS, cell_grids, load_idrid, stem_maps, unit_aucs
from dendritic_dr.utils import DATA_DIR, get_device, results_root

KEYS = [("dend-lrf", 12), ("dend-r", 12), ("conv", 12), ("conv", 5)]
LABEL = {("dend-lrf", 12): "dendritic, clustered", ("dend-r", 12): "dendritic, scattered",
         ("conv", 12): "dense 12x12", ("conv", 5): "dense 5x5"}
COLORS = {"dend-lrf": "#0072B2", "dend-r": "#E69F00", "conv": "#009E73"}
PAIRS = [(KEYS[0], KEYS[1]), (KEYS[0], KEYS[2]), (KEYS[1], KEYS[2]), (KEYS[0], KEYS[3])]


def script(name):
    spec = spec_from_file_location(name.split("_", 1)[1], Path(__file__).with_name(f"{name}.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade_reference(root, smoke, device):
    """Best unit / dendrite AUC per lesion of the step-15 (grade-trained) CNNs on IDRiD test images."""
    runs_csv = root / "cnn_runs.csv"
    if not runs_csv.exists():
        return pd.DataFrame()
    if smoke:
        data = script("18_train_lesions").smoke_data()
        mean, sd = 100.0, 25.0
    else:
        data = load_idrid(DATA_DIR / "raw" / "idrid", DATA_DIR / "features" / "idrid_512.npz")
        X = np.load(DATA_DIR / "features" / "pixels_green_512.npz", allow_pickle=True)["X_train"]
        mean, sd = script("15_train_cnn").pixel_stats(X)
    te = np.flatnonzero(data["split"] == "test")
    size = data["green"].shape[-1]
    rows = []
    runs = pd.read_csv(runs_csv).drop_duplicates("run_id", keep="last")
    for r in runs.itertuples():
        ck = torch.load(root / "models" / f"{r.run_id}.pt", map_location="cpu", weights_only=False)
        a = ck["args"]
        model = build_cnn(a["stem"], somas=a["somas"], dendrites=a["dendrites"], synapses=a["synapses"],
                          kernel=a["kernel"], cluster=a["cluster"], stride=a["stride"], seed=a["seed"])
        model.load_state_dict(ck["state_dict"])
        model = model.to(device).eval()
        with torch.no_grad():
            shape = tuple(model.stem(torch.zeros(1, 1, size, size, device=device)).shape[-2:])
        grids = [cell_grids(data["green"][i], data["masks"][i], shape) for i in te]
        units, dends = [], []
        for i in te:
            x = (torch.from_numpy(data["green"][i].copy()).float().to(device)[None, None] - mean) / sd
            u, d = stem_maps(model, x)
            units.append(u)
            dends.append(d)
        rng = np.random.default_rng(0)
        row = {"stem": r.stem, "kernel": int(r.kernel), "seed": int(r.seed)}
        for j, t in enumerate(LESIONS):
            for level, acts in [("unit", units), ("dend", dends)]:
                if acts[0] is None:
                    continue
                au = unit_aucs(acts, [g[0][j] for g in grids], [g[2] for g in grids], rng)
                row |= {f"{level}_best_{t}": float(np.nanmax(au)), f"{level}_frac75_{t}": float(np.nanmean(au >= 0.75))}
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-grade-reference", action="store_true", help="skip the step-15 comparison")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    root = results_root(args.smoke)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(root / "lesion_runs.csv").drop_duplicates("run_id", keep="last")
    runs["key"] = list(zip(runs["stem"], runs["kernel"].astype(int)))
    metrics = ["test_map"] + [f"ap_{t}" for t in LESIONS]
    unit_cols = [c for c in runs if c.startswith(("unit_", "dend_"))]

    cfg = runs.groupby(["key", "depth"])[metrics + unit_cols + [f"prev_{t}" for t in LESIONS]].mean().reset_index()
    cfg.insert(0, "label", [LABEL.get(k, str(k)) for k in cfg["key"]])
    cfg.drop(columns="key").to_csv(out_dir / "lesion_by_config.csv", index=False)

    tests = []

    def paired(depth, metric, ka, kb, family):
        A = runs[(runs.key == ka) & (runs.depth == depth)].set_index("seed")[metric]
        B = runs[(runs.key == kb) & (runs.depth == depth)].set_index("seed")[metric]
        j = pd.concat([A.rename("a"), B.rename("b")], axis=1, join="inner").dropna()
        if len(j) < 2:
            return
        d = j["a"] - j["b"]
        pv = float(stats.wilcoxon(j["a"], j["b"]).pvalue) if (d != 0).any() else 1.0
        tests.append({"family": family, "depth": depth, "metric": metric, "a": LABEL[ka], "b": LABEL[kb],
                      "n": len(j), "mean_a": j["a"].mean(), "mean_b": j["b"].mean(),
                      "median_diff": float(d.median()), "p": pv, "test": "Wilcoxon (paired by seed)"})

    for depth in sorted(runs["depth"].unique()):
        for m in metrics:
            for ka, kb in PAIRS:
                paired(depth, m, ka, kb, f"AP, depth {depth}")
    for t in LESIONS:
        for m in [f"dend_best_{t}", f"dend_frac75_{t}"]:
            if m in runs:
                paired(0, m, KEYS[0], KEYS[1], "dendrites, depth 0")

    ref = pd.DataFrame() if args.no_grade_reference else grade_reference(root, args.smoke, get_device(args.device))
    if not ref.empty:
        ref.to_csv(out_dir / "lesion_grade_reference.csv", index=False)
        for k in KEYS:
            for t in LESIONS:
                c = f"unit_best_{t}"
                a = runs[(runs.key == k) & (runs.depth == 0)][c].dropna()
                b = ref[(ref.stem == k[0]) & (ref.kernel == k[1])][c].dropna()
                if len(a) >= 2 and len(b) >= 2:
                    tests.append({"family": "lesion- vs. grade-trained units", "depth": 0, "metric": c,
                                  "a": LABEL[k] + " (lesion-trained)", "b": LABEL[k] + " (grade-trained)",
                                  "n": min(len(a), len(b)), "mean_a": a.mean(), "mean_b": b.mean(),
                                  "median_diff": float(a.median() - b.median()),
                                  "p": float(stats.mannwhitneyu(a, b).pvalue), "test": "Mann-Whitney"})
    tests = pd.DataFrame(tests)
    if not tests.empty:
        tests["q"] = np.nan
        for fam, g in tests.groupby("family"):
            tests.loc[g.index, "q"] = bh_fdr(g["p"])
    tests.to_csv(out_dir / "lesion_tests.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        show = ["label", "depth", "test_map"] + [f"ap_{t}" for t in LESIONS] + [f"unit_best_{t}" for t in LESIONS]
        print("Mean over seeds (test AP per lesion; best first-layer unit AUC):")
        print(cfg[show].round(3).to_string(index=False))
        print("\nChance AP (lesion share of test cells): "
              + ", ".join(f"{t} {runs[f'prev_{t}'].mean():.4f}" for t in LESIONS))
        dcols = ["label", "depth"] + [c for c in cfg if c.startswith("dend_")]
        if len(dcols) > 2:
            print("\nDendrites (best AUC, share with AUC >= 0.75):")
            print(cfg[cfg["label"].str.startswith("dendritic")][dcols].round(3).to_string(index=False))
        if not ref.empty:
            print("\nGrade-trained reference (step 15) on the same test images, best unit AUC:")
            print(ref.groupby(["stem", "kernel"])[[f"unit_best_{t}" for t in LESIONS]].mean().round(3).to_string())
        if not tests.empty:
            print("\nTests (q: BH within each family):")
            print(tests.drop(columns="test").round(4).to_string(index=False))

    # --- figure
    depths = sorted(runs["depth"].unique())
    fig, axes = plt.subplots(1, len(depths) + 1, figsize=(4.4 * (len(depths) + 1), 3.9))
    xs = np.arange(len(LESIONS))
    keys = [k for k in KEYS if (runs.key == k).any()]
    for ax, depth in zip(axes, depths):
        for j, k in enumerate(keys):
            s = runs[(runs.key == k) & (runs.depth == depth)]
            mu = s[[f"ap_{t}" for t in LESIONS]].mean().to_numpy()
            sdv = s[[f"ap_{t}" for t in LESIONS]].std().to_numpy()
            off = (j - (len(keys) - 1) / 2) * 0.18
            mk = "o" if k[0].startswith("dend") else ("^" if k[1] == 12 else "v")
            ax.errorbar(xs + off, mu, yerr=sdv, fmt=mk, color=COLORS[k[0]], capsize=3, label=LABEL[k],
                        mfc="white" if k == ("conv", 5) else COLORS[k[0]])
        prev = runs[[f"prev_{t}" for t in LESIONS]].mean().to_numpy()
        ax.hlines(prev, xs - 0.4, xs + 0.4, colors="#555555", linestyles=":", lw=1, label="chance (lesion share)")
        ax.set_xticks(xs, LESIONS)
        ax.set_ylabel("Test average precision (mean ± SD over seeds)")
        ax.set_title("(%s) %s" % ("ab"[depths.index(depth)] if len(depths) <= 2 else depth,
                                  "linear readout on first layer" if depth == 0 else f"{depth} conv blocks before readout"),
                     fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    ax = axes[-1]
    for j, k in enumerate(keys):
        off = (j - (len(keys) - 1) / 2) * 0.18
        s = runs[(runs.key == k) & (runs.depth == 0)]
        ax.errorbar(xs + off, s[[f"unit_best_{t}" for t in LESIONS]].mean(), yerr=s[[f"unit_best_{t}" for t in LESIONS]].std(),
                    fmt="o", color=COLORS[k[0]], capsize=3, label=f"{LABEL[k]}, lesion-trained",
                    mfc="white" if k == ("conv", 5) else COLORS[k[0]])
        if not ref.empty:
            r = ref[(ref.stem == k[0]) & (ref.kernel == k[1])]
            ax.scatter(xs + off, r[[f"unit_best_{t}" for t in LESIONS]].mean(), marker="x", color=COLORS[k[0]], s=25,
                       label="grade-trained (step 15)" if j == 0 else None)
    ax.axhline(0.75, color="#555555", ls="--", lw=0.8)
    ax.set_xticks(xs, LESIONS)
    ax.set_ylabel("Best first-layer unit AUC (depth 0)")
    ax.set_title(f"({'abc'[len(depths)]}) did the units become lesion detectors?", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_lesions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote lesion_by_config.csv, lesion_tests.csv, lesion_grade_reference.csv, fig_lesions.png in {out_dir}")


if __name__ == "__main__":
    main()
