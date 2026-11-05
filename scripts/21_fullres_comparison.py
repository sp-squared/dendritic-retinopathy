"""Step 21: compare first-layer variants trained at near-native resolution (step 20).

For each training set (IDRiD + DDR, or IDRiD alone), readout depth and loss, and each test
set (IDRiD's 27 test images, DDR's 225): FROC score per lesion type, microaneurysm
sensitivity at one false positive per image, mean FROC and cell-level mean AP, paired by
seed (Wilcoxon signed-rank, two-sided; Benjamini-Hochberg within each training set / depth /
loss / test set):

  clustered vs. scattered dendrites   clustered vs. dense 5 x 5   scattered vs. dense 5 x 5
  (and each dendritic variant vs. dense 12 x 12 if those runs exist)

If IDRiD-only runs exist, also: does adding DDR help? (same variant and seed, Wilcoxon).

python scripts/21_fullres_comparison.py
python scripts/21_fullres_comparison.py --smoke

Outputs (results/analysis/): fullres_by_config.csv, fullres_tests.csv, fig_fullres.png
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

KEYS = [("dend-lrf", 12), ("dend-r", 12), ("conv", 5), ("conv", 12)]
LABEL = {("dend-lrf", 12): "dendritic, clustered", ("dend-r", 12): "dendritic, scattered",
         ("conv", 5): "dense 5x5", ("conv", 12): "dense 12x12"}
COLORS = {("dend-lrf", 12): "#0072B2", ("dend-r", 12): "#E69F00", ("conv", 5): "#009E73", ("conv", 12): "#8C8C8C"}
PAIRS = [(KEYS[0], KEYS[1]), (KEYS[0], KEYS[2]), (KEYS[1], KEYS[2]), (KEYS[0], KEYS[3]), (KEYS[1], KEYS[3])]
GROUP = ["train_on", "depth", "loss"]


def metrics(ds):
    return [f"{ds}_froc_{t}" for t in LESIONS] + [f"{ds}_sens1_MA", f"{ds}_mfroc", f"{ds}_map"]


def wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 2 or np.allclose(d, 0):
        return np.nan
    return float(stats.wilcoxon(a, b).pvalue)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    root = results_root(args.smoke)
    runs = pd.read_csv(root / "fullres_runs.csv").drop_duplicates("run_id", keep="last")
    runs["key"] = list(zip(runs["stem"], runs["kernel"]))
    test_sets = [ds for ds in ["idrid", "ddr"] if f"{ds}_mfroc" in runs]
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    # 1. mean / SD per configuration
    cols = [m for ds in test_sets for m in metrics(ds)] + ["best_epoch", "epochs_run"]
    agg = runs.groupby(GROUP + ["stem", "kernel"])[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.insert(0, "n_seeds", runs.groupby(GROUP + ["stem", "kernel"]).size())
    agg.reset_index().to_csv(out / "fullres_by_config.csv", index=False)

    # 2. paired tests between variants
    rows = []
    for g, sub in runs.groupby(GROUP):
        by = {k: s.set_index("seed") for k, s in sub.groupby("key")}
        for ds in test_sets:
            block = []
            for a, b in PAIRS:
                if a not in by or b not in by:
                    continue
                seeds = by[a].index.intersection(by[b].index)
                for m in metrics(ds):
                    x, y = by[a].loc[seeds, m].to_numpy(float), by[b].loc[seeds, m].to_numpy(float)
                    block.append({**dict(zip(GROUP, g)), "test_set": ds, "a": LABEL[a], "b": LABEL[b],
                                  "metric": m.removeprefix(f"{ds}_"), "n": len(seeds), "mean_a": x.mean(),
                                  "mean_b": y.mean(), "diff": (x - y).mean(), "p": wilcoxon(x, y)})
            if block:
                q = bh_fdr([r["p"] for r in block])
                for r, qq in zip(block, q):
                    r["q"] = qq
                rows += block

    # 3. does adding DDR help? (same variant, depth, loss and seed)
    if runs["train_on"].nunique() > 1 and {"idrid", "idrid+ddr"} <= set(runs["train_on"]):
        block = []
        for (depth, loss, key), sub in runs.groupby(["depth", "loss", "key"]):
            both = sub.pivot_table(index="seed", columns="train_on", values=[m for ds in test_sets for m in metrics(ds)])
            for m in [m for ds in test_sets for m in metrics(ds)]:
                if (m, "idrid") not in both or (m, "idrid+ddr") not in both:
                    continue
                d = both[[(m, "idrid+ddr"), (m, "idrid")]].dropna()
                x, y = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
                block.append({"train_on": "idrid+ddr vs idrid", "depth": depth, "loss": loss,
                              "test_set": m.split("_", 1)[0], "a": LABEL.get(key, str(key)) + " (+DDR)",
                              "b": LABEL.get(key, str(key)) + " (IDRiD only)", "metric": m.split("_", 1)[1],
                              "n": len(d), "mean_a": x.mean() if len(d) else np.nan,
                              "mean_b": y.mean() if len(d) else np.nan,
                              "diff": (x - y).mean() if len(d) else np.nan, "p": wilcoxon(x, y)})
        if block:
            for r, qq in zip(block, bh_fdr([r["p"] for r in block])):
                r["q"] = qq
            rows += block
    tests = pd.DataFrame(rows)
    tests.to_csv(out / "fullres_tests.csv", index=False)

    show = tests[tests["metric"].isin(["froc_MA", "sens1_MA", "mfroc"])] if len(tests) else tests
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(show[["train_on", "depth", "loss", "test_set", "a", "b", "metric", "n", "mean_a", "mean_b", "diff", "p", "q"]]
              .round(4).to_string(index=False) if len(show) else "no paired comparisons (need >= 2 variants)")

    # 4. figure: FROC per lesion type, main training set / depth / loss, one panel per test set
    main_g = runs.groupby(GROUP).size().sort_values().index[-1]
    sub = runs[(runs[GROUP] == pd.Series(main_g, index=GROUP)).all(axis=1)]
    keys = [k for k in KEYS if k in set(sub["key"])]
    fig, axes = plt.subplots(1, len(test_sets), figsize=(5.2 * len(test_sets), 3.6), squeeze=False, sharey=True)
    w = 0.8 / max(len(keys), 1)
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes[0], test_sets):
        for i, k in enumerate(keys):
            s = sub[sub["key"] == k]
            for j, t in enumerate(LESIONS + ["mean"]):
                col = f"{ds}_mfroc" if t == "mean" else f"{ds}_froc_{t}"
                v = s[col].to_numpy(float)
                xpos = j + (i - (len(keys) - 1) / 2) * w
                ax.bar(xpos, np.nanmean(v), w * 0.9, color=COLORS[k], label=LABEL[k] if j == 0 else None)
                ax.scatter(xpos + rng.uniform(-w * 0.2, w * 0.2, len(v)), v, s=6, color="#222222", zorder=3, linewidths=0)
        n = sub[f"{ds}_n_images"].iloc[0] if f"{ds}_n_images" in sub else ""
        ax.set_title(f"{'IDRiD' if ds == 'idrid' else 'DDR'} test ({int(n)} images)" if n != "" else ds, fontsize=10)
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
    print(f"wrote {out / 'fullres_by_config.csv'}, {out / 'fullres_tests.csv'}, {out / 'fig_fullres.png'}")


if __name__ == "__main__":
    main()
