"""Step 16: compare the first-layer variants of the end-to-end CNN (step 15).

Pairs (same seeds, same test images):
  dend-lrf vs. dend-r      locality: identical parameters and 12 x 12 field, clustered vs. scattered
  dend-lrf vs. conv k12    dendritic vs. dense at identical first-layer parameters
  dend-r   vs. conv k12
  dend-lrf vs. conv k5     vs. a conventional small dense kernel
Each pair: Wilcoxon signed-rank over seeds, paired bootstrap over test images, BH across pairs.

python scripts/16_cnn_comparison.py
python scripts/16_cnn_comparison.py --smoke

Outputs (results/analysis/): cnn_by_config.csv, cnn_paired.csv, fig_cnn.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from dendritic_dr.analysis import bh_fdr, paired_bootstrap_diff
from dendritic_dr.utils import results_root

LABELS = {"dend-lrf": "Dendritic, clustered (LRF)", "dend-r": "Dendritic, scattered (R)", "conv": "Dense conv"}
COLORS = {"dend-lrf": "#0072B2", "dend-r": "#E69F00", "conv": "#009E73"}
PAIRS = [(("dend-lrf", 12), ("dend-r", 12)), (("dend-lrf", 12), ("conv", 12)),
         (("dend-r", 12), ("conv", 12)), (("dend-lrf", 12), ("conv", 5))]


def label(stem, kernel):
    return f"{LABELS[stem]} {kernel}x{kernel}" if stem == "conv" else LABELS[stem]


def load_preds(root, run_ids):
    frames = [pd.read_csv(root / "predictions" / f"{r}.csv").set_index("id_code") for r in run_ids]
    for f in frames[1:]:
        if not f.index.equals(frames[0].index):
            raise SystemExit("prediction files are not aligned on the same test images")
    return frames[0]["true"].to_numpy(), np.stack([f["pred"].to_numpy() for f in frames])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    root = results_root(args.smoke)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(root / "cnn_runs.csv")
    runs["key"] = list(zip(runs["stem"], runs["kernel"].astype(int)))

    cfg = (runs.groupby("key").agg(config=("config", "first"), stem_params=("stem_params", "first"),
                                   total_params=("effective_params", "first"), mean=("test_qwk", "mean"),
                                   sd=("test_qwk", "std"), n=("test_qwk", "size")).reset_index())
    cfg["label"] = [label(*k) for k in cfg["key"]]
    cfg.drop(columns="key").to_csv(out_dir / "cnn_by_config.csv", index=False)

    rows = []
    for j, (ka, kb) in enumerate(PAIRS):
        ra, rb = runs[runs["key"] == ka].set_index("seed"), runs[runs["key"] == kb].set_index("seed")
        seeds = sorted(set(ra.index) & set(rb.index))
        if len(seeds) < 2:
            continue
        qa, qb = ra.loc[seeds, "test_qwk"].to_numpy(), rb.loc[seeds, "test_qwk"].to_numpy()
        d = qa - qb
        w_p = float(stats.wilcoxon(qa, qb).pvalue) if np.any(d != 0) else 1.0
        y, pa = load_preds(root, ra.loc[seeds, "run_id"])
        _, pb = load_preds(root, rb.loc[seeds, "run_id"])
        diff, lo, hi, bp = paired_bootstrap_diff(y, pa, pb, n_boot=args.n_boot, seed=j)
        rows.append({"a": label(*ka), "b": label(*kb), "n_seeds": len(seeds), "qwk_a": qa.mean(), "qwk_b": qb.mean(),
                     "diff_mean": d.mean(), "diff_median": float(np.median(d)), "wilcoxon_p": w_p,
                     "boot_diff": diff, "boot_ci_low": lo, "boot_ci_high": hi, "boot_p": bp})
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs["wilcoxon_q"] = bh_fdr(pairs["wilcoxon_p"])
        pairs["boot_q"] = bh_fdr(pairs["boot_p"])
    pairs.to_csv(out_dir / "cnn_paired.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(cfg[["label", "stem_params", "total_params", "n", "mean", "sd"]].round(4).to_string(index=False))
        if not pairs.empty:
            print("\nPaired differences (A - B):")
            print(pairs[["a", "b", "n_seeds", "diff_mean", "wilcoxon_q", "boot_diff", "boot_ci_low",
                         "boot_ci_high", "boot_q"]].round(4).to_string(index=False))

    # --- figure: (a) test QWK per variant with seeds, (b) paired differences ---
    order = [k for k in [("dend-lrf", 12), ("dend-r", 12), ("conv", 12), ("conv", 5)] if k in set(runs["key"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [1, 1.15]})
    rng = np.random.default_rng(0)
    for i, k in enumerate(order):
        q = runs.loc[runs["key"] == k, "test_qwk"].to_numpy()
        ax1.scatter(i + rng.uniform(-0.12, 0.12, len(q)), q, s=12, alpha=0.5, color=COLORS[k[0]])
        ax1.errorbar(i + 0.25, q.mean(), yerr=q.std(ddof=1) if len(q) > 1 else 0, marker="o",
                     color=COLORS[k[0]], capsize=4)
    ax1.set_xticks(range(len(order)), [label(*k).replace(", ", ",\n").replace(" conv ", " conv\n") for k in order],
                   fontsize=7)
    ax1.set_ylabel("Test QWK (seeds; mean ± SD)")
    ax1.set_title("(a) first-layer variants, rest of the CNN identical", fontsize=9)
    ax1.grid(alpha=0.3, axis="y")
    if not pairs.empty:
        y = np.arange(len(pairs))[::-1]
        err = np.vstack([pairs["boot_diff"] - pairs["boot_ci_low"], pairs["boot_ci_high"] - pairs["boot_diff"]])
        ax2.errorbar(pairs["boot_diff"], y, xerr=err, fmt="o", color="#333333", capsize=4)
        for yi, xi, q in zip(y, pairs["boot_ci_high"], pairs["boot_q"]):
            ax2.annotate(f"q={q:.2f}", (xi, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7)
        short = lambda s: s.split(" (")[0].replace("Dendritic, ", "").replace("Dense conv ", "dense ")
        ax2.set_yticks(y, [f"{short(a)} − {short(b)}" for a, b in zip(pairs["a"], pairs["b"])], fontsize=7)
        ax2.axvline(0, color="black", lw=0.8)
        ax2.margins(x=0.25)
    ax2.set_xlabel("QWK difference (95% paired-bootstrap CI)")
    ax2.set_title("(b) paired differences", fontsize=9)
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cnn.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_dir / 'cnn_by_config.csv'}, cnn_paired.csv and fig_cnn.png")


if __name__ == "__main__":
    main()
