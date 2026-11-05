"""Step 13: do dendrites regain their advantage on raw fundus pixels?

Prediction (stated before running): if the Fashion-MNIST advantage comes from input-level
locality, then on raw pixels (step 12) dANN-LRF beats dANN-R of identical size, and dANN-LRF
matches or beats a dense ANN of similar parameter count. If neither holds, the advantage is
specific to small benchmark images rather than to pixel inputs in general.

Comparisons (all on the same test images and the same seeds):
  * dANN-LRF vs. dANN-R at each soma count (identical parameter counts)
  * each dANN config vs. the dense config closest in effective parameters (log scale)
Each pair gets a Wilcoxon signed-rank test over seeds and a paired bootstrap over test
images, with Benjamini-Hochberg correction across all pairs.

Run after training (see README, step 4d):
python scripts/13_pixel_comparison.py
python scripts/13_pixel_comparison.py --features pixels_gray_96

Outputs (results/analysis/): pixels_by_config.csv, pixels_paired.csv, fig_pixels.png
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
from dendritic_dr.plotting import config_label, display_name, format_params_axis, style
from dendritic_dr.utils import results_root

MODELS = ["dann-lrf", "dann-r", "vanilla"]


def clean(v) -> str:
    return str(v).strip("[]").replace(".0", "") if pd.notna(v) else ""


def load_preds(root, run_ids):
    frames = [pd.read_csv(root / "predictions" / f"{r}.csv").set_index("id_code") for r in run_ids]
    for f in frames[1:]:
        if not f.index.equals(frames[0].index):
            raise SystemExit("prediction files are not aligned on the same test images")
    return frames[0]["true"].to_numpy(), np.stack([f["pred"].to_numpy() for f in frames])


def compare(root, runs, a, b, n_boot, seed):
    ra, rb = runs[runs["config"] == a].set_index("seed"), runs[runs["config"] == b].set_index("seed")
    seeds = sorted(set(ra.index) & set(rb.index))
    if len(seeds) < 2:
        print(f"{a} vs {b}: fewer than 2 paired seeds, skipped")
        return None
    qa, qb = ra.loc[seeds, "test_qwk"].to_numpy(), rb.loc[seeds, "test_qwk"].to_numpy()
    d = qa - qb
    w_p = float(stats.wilcoxon(qa, qb).pvalue) if np.any(d != 0) else 1.0
    y, pa = load_preds(root, ra.loc[seeds, "run_id"])
    _, pb = load_preds(root, rb.loc[seeds, "run_id"])
    diff, lo, hi, bp = paired_bootstrap_diff(y, pa, pb, n_boot=n_boot, seed=seed)
    return {"config_a": a, "config_b": b, "n_seeds": len(seeds),
            "params_a": int(ra["effective_params"].iloc[0]), "params_b": int(rb["effective_params"].iloc[0]),
            "qwk_a_mean": qa.mean(), "qwk_b_mean": qb.mean(), "diff_mean": d.mean(),
            "diff_median": float(np.median(d)), "wilcoxon_p": w_p,
            "boot_diff": diff, "boot_ci_low": lo, "boot_ci_high": hi, "boot_p": bp}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="pixels_green_128")
    p.add_argument("--reference", default="resnet18_224_vanilla_h73_ce",
                   help="config drawn as a dashed reference line (pooled ResNet dense head)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.features, args.reference = "smoke_pixels", "smoke_vanilla_h73_ce"

    root = results_root(args.smoke)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_runs = pd.read_csv(root / "aptos_heads.csv")
    runs = all_runs[(all_runs["features"] == args.features) & all_runs["model"].isin(MODELS)].copy()
    if runs.empty:
        raise SystemExit(f"no runs on {args.features}; train step 4d first")
    runs["somas_n"] = runs["somas"].map(clean)

    # per-configuration summary
    cfg = (runs.groupby("config")
           .agg(model=("model", "first"), somas=("somas_n", "first"), hidden=("hidden", "first"),
                params=("effective_params", "first"), mean=("test_qwk", "mean"), sd=("test_qwk", "std"),
                n=("test_qwk", "size"))
           .reset_index().sort_values(["model", "params"]))
    cfg["label"] = [config_label(r) for r in cfg.to_dict("records")]
    cfg.to_csv(out_dir / "pixels_by_config.csv", index=False)

    # paired comparisons
    rows = []
    lrf, rnd, dense = (cfg[cfg["model"] == m] for m in MODELS)
    for k, r in enumerate(lrf.itertuples()):
        match = rnd[rnd["somas"] == r.somas]
        if len(match):
            res = compare(root, runs, r.config, match["config"].iloc[0], args.n_boot, seed=k)
            if res:
                rows.append({"comparison": "LRF - random", "somas": int(r.somas), **res})
    if len(dense):
        for k, r in enumerate(pd.concat([lrf, rnd]).itertuples()):
            nearest = dense.loc[(np.log(dense["params"]) - np.log(r.params)).abs().idxmin(), "config"]
            res = compare(root, runs, r.config, nearest, args.n_boot, seed=100 + k)
            if res:
                kind = "LRF - dense" if r.model == "dann-lrf" else "random - dense"
                rows.append({"comparison": kind, "somas": int(r.somas), **res})
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs["wilcoxon_q"] = bh_fdr(pairs["wilcoxon_p"])
        pairs["boot_q"] = bh_fdr(pairs["boot_p"])
    pairs.to_csv(out_dir / "pixels_paired.csv", index=False)

    ref = all_runs[all_runs["config"] == args.reference]["test_qwk"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(cfg[["label", "params", "n", "mean", "sd"]].round(4).to_string(index=False))
        if not pairs.empty:
            cols = ["comparison", "somas", "params_a", "params_b", "diff_mean", "wilcoxon_q",
                    "boot_diff", "boot_ci_low", "boot_ci_high", "boot_q"]
            print("\nPaired differences (A - B):")
            print(pairs[cols].round(4).to_string(index=False))
    if len(ref):
        print(f"\nreference {args.reference}: QWK {ref.mean():.4f} (n = {len(ref)})")

    # --- figure: (a) QWK vs. parameters, (b) paired differences per soma count ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for m in MODELS:
        sub = cfg[cfg["model"] == m].sort_values("params")
        if sub.empty:
            continue
        st = style(m)
        ax1.errorbar(sub["params"], sub["mean"], yerr=sub["sd"].fillna(0), marker=st["marker"],
                     color=st["color"], capsize=3, label=display_name(m))
    if len(ref):
        ax1.axhline(ref.mean(), ls="--", lw=1, color="#555555", label="Dense ANN, pooled ResNet-18")
    ax1.set_xscale("log")
    format_params_axis(ax1)
    ax1.set_xlabel("Effective trainable parameters (log scale)")
    ax1.set_ylabel("Test QWK (mean ± SD over seeds)")
    ax1.set_title(f"(a) heads on raw pixels ({args.features})", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    if not pairs.empty:
        somas = sorted(pairs["somas"].unique())
        pos = {s: i for i, s in enumerate(somas)}
        kinds = [("LRF - random", "#0072B2", "o"), ("LRF - dense", "#009E73", "^")]
        for j, (kind, color, marker) in enumerate(kinds):
            sub = pairs[pairs["comparison"] == kind]
            if sub.empty:
                continue
            x = np.array([pos[s] for s in sub["somas"]]) + (j - 0.5) * 0.18
            err = np.vstack([sub["boot_diff"] - sub["boot_ci_low"], sub["boot_ci_high"] - sub["boot_diff"]])
            ax2.errorbar(x, sub["boot_diff"], yerr=err, ls="none", marker=marker, color=color, capsize=4,
                         label=kind.replace("LRF", "local RF").replace("-", "−"))
            for xi, yi, q in zip(x, sub["boot_ci_high"], sub["boot_q"]):
                ax2.annotate(f"q={q:.2f}", (xi, yi), xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=6)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_xticks(range(len(somas)), [str(s) for s in somas])
        ax2.margins(y=0.18)
        ax2.legend(fontsize=7, loc="lower right")
    ax2.set_xlabel("Somas per dANN (dense partner: closest parameter count)")
    ax2.set_ylabel("QWK difference")
    ax2.set_title("(b) paired differences, 95% bootstrap CI", fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pixels.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_dir / 'pixels_by_config.csv'}, pixels_paired.csv and fig_pixels.png")


if __name__ == "__main__":
    main()
