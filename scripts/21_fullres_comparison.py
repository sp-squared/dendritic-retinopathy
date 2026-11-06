"""Step 21: compare first-layer variants trained at near-native resolution (step 20).

For each training set (IDRiD + DDR, or IDRiD alone), readout depth and loss, and each test
set (IDRiD's 27 test images, DDR's 225): FROC score per lesion type, microaneurysm
sensitivity at one false positive per image, mean FROC and cell-level mean AP, paired by
seed (Wilcoxon signed-rank, two-sided; Benjamini-Hochberg within each family below).

Family "main" (the step 4i comparison):
  clustered (12 x 12, 5 x 5 clusters) vs. scattered (12 x 12)   clustered vs. dense 5 x 5
  scattered vs. dense 5 x 5   (and each vs. dense 12 x 12 if those runs exist)

Family "window" (step 4j, only if those runs exist): every other clustered configuration
(cluster size c, kernel k) vs. scattered dendrites with the same kernel, vs. the default
clustered layer (k 12, c 5), and vs. dense 5 x 5; scattered k 24 vs. scattered k 12. The
question: at 2048 px a microaneurysm is ~4-12 px wide, larger than the default 5 x 5 cluster;
does a cluster scaled to the lesion bring back the clustered advantage seen at 512 px?

Family "ddr" (if IDRiD-only runs exist): does adding DDR help? (same variant and seed).

python scripts/21_fullres_comparison.py
python scripts/21_fullres_comparison.py --smoke

Outputs (results/analysis/): fullres_by_config.csv, fullres_tests.csv, fig_fullres.png,
fig_fullres_window.png (window sweep, if present)
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from dendritic_dr.analysis import bh_fdr
from dendritic_dr.idrid import LESIONS
from dendritic_dr.utils import results_root

LRF, R, D5, D12 = ("dend-lrf", 12, 5), ("dend-r", 12, 0), ("conv", 5, 0), ("conv", 12, 0)
MAIN_KEYS = [LRF, R, D5, D12]
MAIN_PAIRS = [(LRF, R), (LRF, D5), (R, D5), (LRF, D12), (R, D12)]
COLORS = {LRF: "#0072B2", R: "#E69F00", D5: "#009E73", D12: "#8C8C8C"}
GROUP = ["train_on", "depth", "loss"]


def label(key):
    stem, k, c = key
    if stem == "conv":
        return f"dense {k}x{k}"
    if stem == "dend-r":
        return f"dendritic, scattered (k {k})"
    return f"dendritic, clustered (k {k}, c {c})"


def metrics(ds):
    return [f"{ds}_froc_{t}" for t in LESIONS] + [f"{ds}_sens1_MA", f"{ds}_mfroc", f"{ds}_map"]


def wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 2 or np.allclose(d, 0):
        return np.nan
    return float(stats.wilcoxon(a, b).pvalue)


def add_keys(runs):
    """(stem, kernel, cluster) per run; cluster 0 for scattered / dense. Older rows have no
    cluster column: it is read from the config name (..._c5)."""
    c = runs["config"].str.extract(r"_c(\d+)$")[0].astype(float)
    if "cluster" in runs:
        c = runs["cluster"].where(runs["cluster"].notna(), c)
    c = c.where(runs["stem"] == "dend-lrf", 0).fillna(0).astype(int)
    runs["cluster"] = c
    runs["key"] = list(zip(runs["stem"], runs["kernel"].astype(int), c))
    return runs


def paired_block(by, pairs, ds, extra):
    block = []
    for a, b in pairs:
        if a not in by or b not in by:
            continue
        seeds = by[a].index.intersection(by[b].index)
        for m in metrics(ds):
            x, y = by[a].loc[seeds, m].to_numpy(float), by[b].loc[seeds, m].to_numpy(float)
            block.append({**extra, "test_set": ds, "a": label(a), "b": label(b), "metric": m.removeprefix(f"{ds}_"),
                          "n": len(seeds), "mean_a": x.mean(), "mean_b": y.mean(), "diff": (x - y).mean(),
                          "p": wilcoxon(x, y)})
    for r, q in zip(block, bh_fdr([r["p"] for r in block]) if block else []):
        r["q"] = q
    return block


def window_pairs(keys):
    """Step 4j pairs among the keys present."""
    pairs = []
    for k in sorted(keys):
        stem, kern, c = k
        if stem == "dend-lrf" and k != LRF:
            pairs += [(k, ("dend-r", kern, 0)), (k, LRF), (k, D5)]
        if stem == "dend-r" and kern != 12:
            pairs += [(k, R), (k, D5)]
    return pairs


def window_figure(sub, test_sets, path):
    """MA FROC (and mean FROC) vs. cluster size, one line per kernel; scattered at the right
    end (cluster = whole kernel); dense 5 x 5 as a dashed reference line."""
    dend = sub[sub["stem"].str.startswith("dend")]
    kernels = sorted(dend["kernel"].unique())
    fig, axes = plt.subplots(2, len(test_sets), figsize=(4.6 * len(test_sets), 6.2), squeeze=False, sharex=True)
    kcolor = dict(zip(kernels, ["#0072B2", "#CC79A7", "#56B4E9", "#D55E00"]))
    for col, ds in enumerate(test_sets):
        for row, (m, name) in enumerate([(f"{ds}_froc_MA", "microaneurysm FROC"), (f"{ds}_mfroc", "mean FROC (4 lesions)")]):
            ax = axes[row, col]
            for k in kernels:
                s = dend[dend["kernel"] == k].copy()
                s["c"] = np.where(s["stem"] == "dend-r", k, s["cluster"])  # scattered = cluster spanning the kernel
                g = s.groupby("c")[m].agg(["mean", "std"]).sort_index()
                ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=5, lw=2, capsize=3,
                            color=kcolor[k], label=f"kernel {k} x {k}")
                if (s["stem"] == "dend-r").any():
                    ax.scatter([k], [g.loc[k, "mean"]], s=90, facecolors="none", edgecolors=kcolor[k], zorder=4)
            d5 = sub[(sub["stem"] == "conv") & (sub["kernel"] == 5)][m]
            if len(d5):
                ax.axhline(d5.mean(), color="#009E73", ls="--", lw=1.5, label="dense 5 x 5")
            ax.set_ylabel(name)
            ax.grid(color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"{'IDRiD' if ds == 'idrid' else 'DDR'} test", fontsize=10)
            else:
                ax.set_xlabel("cluster size c (px); open circle = scattered")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Does a cluster scaled to the lesion help? (2048 px; mean +/- SD over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    root = results_root(args.smoke)
    runs = add_keys(pd.read_csv(root / "fullres_runs.csv").drop_duplicates("run_id", keep="last"))
    runs = runs[~runs["config"].str.contains(r"_n\d+_", regex=True)]  # learning-curve subsets: script 22
    test_sets = [ds for ds in ["idrid", "ddr"] if f"{ds}_mfroc" in runs]
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    # 1. mean / SD per configuration
    cols = [m for ds in test_sets for m in metrics(ds)] + ["best_epoch", "epochs_run"]
    keys = GROUP + ["stem", "kernel", "cluster"]
    agg = runs.groupby(keys)[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.insert(0, "n_seeds", runs.groupby(keys).size())
    agg.reset_index().to_csv(out / "fullres_by_config.csv", index=False)

    # 2. paired tests between variants, per family
    rows = []
    for g, sub in runs.groupby(GROUP):
        by = {k: s.set_index("seed") for k, s in sub.groupby("key")}
        extra = dict(zip(GROUP, g))
        for ds in test_sets:
            rows += [r | {"family": "main"} for r in paired_block(by, MAIN_PAIRS, ds, extra)]
            rows += [r | {"family": "window"} for r in paired_block(by, window_pairs(by), ds, extra)]

    # 3. does adding DDR help? (same variant, depth, loss and seed)
    if {"idrid", "idrid+ddr"} <= set(runs["train_on"]):
        block = []
        allm = [m for ds in test_sets for m in metrics(ds)]
        for (depth, loss, key), sub in runs.groupby(["depth", "loss", "key"]):
            both = sub.pivot_table(index="seed", columns="train_on", values=allm)
            for m in allm:
                if (m, "idrid") not in both or (m, "idrid+ddr") not in both:
                    continue
                d = both[[(m, "idrid+ddr"), (m, "idrid")]].dropna()
                x, y = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
                block.append({"family": "ddr", "train_on": "idrid+ddr vs idrid", "depth": depth, "loss": loss,
                              "test_set": m.split("_", 1)[0], "a": label(key) + " (+DDR)",
                              "b": label(key) + " (IDRiD only)", "metric": m.split("_", 1)[1], "n": len(d),
                              "mean_a": x.mean() if len(d) else np.nan, "mean_b": y.mean() if len(d) else np.nan,
                              "diff": (x - y).mean() if len(d) else np.nan, "p": wilcoxon(x, y)})
        for r, q in zip(block, bh_fdr([r["p"] for r in block]) if block else []):
            r["q"] = q
        rows += block
    tests = pd.DataFrame(rows)
    if len(tests):
        tests = tests[["family"] + [c for c in tests.columns if c != "family"]]
    tests.to_csv(out / "fullres_tests.csv", index=False)

    show = tests[tests["metric"].isin(["froc_MA", "sens1_MA", "mfroc"])] if len(tests) else tests
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(show[["family", "train_on", "depth", "test_set", "a", "b", "metric", "n", "mean_a", "mean_b", "diff", "p", "q"]]
              .round(4).to_string(index=False) if len(show) else "no paired comparisons (need >= 2 variants)")

    # 4. figure: FROC per lesion type for the main variants, main training set / depth / loss
    main_g = runs.groupby(GROUP).size().sort_values().index[-1]
    sub = runs[(runs[GROUP] == pd.Series(main_g, index=GROUP)).all(axis=1)]
    keys = [k for k in MAIN_KEYS if k in set(sub["key"])]
    fig, axes = plt.subplots(1, len(test_sets), figsize=(5.2 * len(test_sets), 3.6), squeeze=False, sharey=True)
    w = 0.8 / max(len(keys), 1)
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes[0], test_sets):
        for i, k in enumerate(keys):
            s = sub[sub["key"] == k]
            for j, t in enumerate(LESIONS + ["mean"]):
                v = s[f"{ds}_mfroc" if t == "mean" else f"{ds}_froc_{t}"].to_numpy(float)
                xpos = j + (i - (len(keys) - 1) / 2) * w
                ax.bar(xpos, np.nanmean(v), w * 0.9, color=COLORS[k], label=label(k) if j == 0 else None)
                ax.scatter(xpos + rng.uniform(-w * 0.2, w * 0.2, len(v)), v, s=6, color="#222222", zorder=3, linewidths=0)
        n = sub[f"{ds}_n_images"].iloc[0] if f"{ds}_n_images" in sub else None
        ax.set_title(f"{'IDRiD' if ds == 'idrid' else 'DDR'} test" + (f" ({int(n)} images)" if n is not None else ""), fontsize=10)
        ax.set_xticks(range(len(LESIONS) + 1), LESIONS + ["mean"])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("FROC score\n(mean sensitivity, 1/8-8 FP per image)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Lesion detection at {int(sub['size'].iloc[0])} px, trained on {main_g[0]} "
                 f"(depth {main_g[1]}, {main_g[2]} loss; bars: mean, dots: seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_fullres.png", dpi=200)
    written = [out / "fullres_by_config.csv", out / "fullres_tests.csv", out / "fig_fullres.png"]

    if window_pairs(set(sub["key"])):
        window_figure(sub, test_sets, out / "fig_fullres_window.png")
        written.append(out / "fig_fullres_window.png")
    print("wrote " + ", ".join(str(w) for w in written))


if __name__ == "__main__":
    main()
