"""Step 11: receptive-field sweep. Does the dendritic locality advantage shrink with depth?

Hypothesis (stated before running): local-receptive-field dendrites (dANN-LRF) beat
randomly sampling dendrites (dANN-R) of identical size when map cells are local
(early ResNet stages) and the gap shrinks to zero at layer4, whose cells see most of
the image.

Design: ResNet-18 feature maps from stages 1-4 at 224 px, each pooled to ~25k inputs so
only locality differs (layer1 64x20x20, layer2 128x14x14, layer3 256x10x10, layer4
512x7x7). At every stage dANN-LRF and dANN-R have identical parameter counts (default
128 somas x 16 dendrites x 16 synapses, 3x3 window) and are trained with the same 10 seeds.

For each stage this script reports mean test QWK of both variants and the paired
difference LRF - R:
  * Wilcoxon signed-rank test over seeds (training variability)
  * paired bootstrap over test images of the mean-over-seeds difference (test-set variability)
  * Benjamini-Hochberg correction across stages for both
  * a trend test: Spearman correlation between stage (1-4) and per-seed differences

Run after training (see README, step 4c):
python scripts/11_receptive_field_sweep.py
python scripts/11_receptive_field_sweep.py --somas 32      # secondary size, if trained

Outputs (results/analysis/): rf_sweep.csv, fig_rf_sweep.png
"""
from __future__ import annotations

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from dendritic_dr.analysis import bh_fdr, paired_bootstrap_diff
from dendritic_dr.plotting import style
from dendritic_dr.utils import results_root

DEFAULT_GRID = {1: 56, 2: 28, 3: 14, 4: 7}  # raw map size at 224 px if no --grid was used


def stage_of(features: str) -> int:
    m = re.search(r"_l(\d)", features)
    return int(m.group(1)) if m else 4


def grid_of(features: str) -> int:
    m = re.search(r"_g(\d+)", features)
    return int(m.group(1)) if m else DEFAULT_GRID[stage_of(features)]


def clean(v) -> str:
    return str(v).strip("[]").replace(".0", "") if pd.notna(v) else ""


def load_preds(root, run_ids):
    frames = [pd.read_csv(root / "predictions" / f"{r}.csv").set_index("id_code") for r in run_ids]
    for f in frames[1:]:
        if not f.index.equals(frames[0].index):
            raise SystemExit("prediction files are not aligned on the same test images")
    return frames[0]["true"].to_numpy(), np.stack([f["pred"].to_numpy() for f in frames])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="resnet18_224", help="feature-name prefix of the spatial runs")
    p.add_argument("--somas", type=int, default=128)
    p.add_argument("--dendrites", type=int, default=16)
    p.add_argument("--synapses", type=int, default=16)
    p.add_argument("--rf-size", type=int, default=3)
    p.add_argument("--reference", default=None, help="config of a pooled reference head (default <prefix>_vanilla_h73_ce)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.prefix = "smoke"

    root = results_root(args.smoke)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(root / "aptos_heads.csv")
    sel = runs[runs["features"].astype(str).str.startswith(args.prefix)
               & runs["features"].astype(str).str.contains("spatial")
               & runs["model"].isin(["dann-lrf", "dann-r"])
               & (runs["somas"].map(clean) == str(args.somas))
               & (runs["dendrites"].map(clean) == str(args.dendrites))
               & (runs["synapses"].map(clean) == str(args.synapses))].copy()
    if "rf_size" in sel:
        lrf = sel["model"] == "dann-lrf"
        sel = sel[~lrf | (sel["rf_size"].map(clean) == str(args.rf_size))]
    if sel.empty:
        raise SystemExit("no matching spatial runs; check --prefix/--somas or train step 4c first")
    sel["stage"] = sel["features"].map(stage_of)

    rows, per_seed = [], []
    for stage, g in sorted(sel.groupby("stage")):
        feats = g["features"].iloc[0]
        a = g[g["model"] == "dann-lrf"].set_index("seed")
        b = g[g["model"] == "dann-r"].set_index("seed")
        seeds = sorted(set(a.index) & set(b.index))
        if len(seeds) < 2:
            print(f"layer{stage}: fewer than 2 paired seeds, skipped")
            continue
        qa, qb = a.loc[seeds, "test_qwk"].to_numpy(), b.loc[seeds, "test_qwk"].to_numpy()
        d = qa - qb
        per_seed += [(stage, x) for x in d]
        w_p = float(stats.wilcoxon(qa, qb).pvalue) if np.any(d != 0) else 1.0
        y, pa = load_preds(root, a.loc[seeds, "run_id"])
        _, pb = load_preds(root, b.loc[seeds, "run_id"])
        bdiff, lo, hi, bp = paired_bootstrap_diff(y, pa, pb, n_boot=args.n_boot, seed=stage)
        grid = grid_of(feats)
        rows.append({"stage": f"layer{stage}", "stage_num": stage, "features": feats, "grid": grid,
                     "window_fraction_of_width": round(args.rf_size / grid, 3), "n_seeds": len(seeds),
                     "params": int(a["effective_params"].iloc[0]),
                     "lrf_mean": qa.mean(), "lrf_sd": qa.std(ddof=1), "r_mean": qb.mean(), "r_sd": qb.std(ddof=1),
                     "diff_mean": d.mean(), "diff_median": float(np.median(d)), "wilcoxon_p": w_p,
                     "boot_diff": bdiff, "boot_ci_low": lo, "boot_ci_high": hi, "boot_p": bp})
    df = pd.DataFrame(rows)
    df["wilcoxon_q"] = bh_fdr(df["wilcoxon_p"])
    df["boot_q"] = bh_fdr(df["boot_p"])
    ps = pd.DataFrame(per_seed, columns=["stage", "diff"])
    trend = stats.spearmanr(ps["stage"], ps["diff"]) if ps["stage"].nunique() > 1 else None
    df.to_csv(out_dir / "rf_sweep.csv", index=False)

    ref_cfg = args.reference or f"{args.prefix}_vanilla_h73_ce"
    ref = runs[runs["config"] == ref_cfg]["test_qwk"]

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        cols = ["stage", "grid", "window_fraction_of_width", "n_seeds", "params", "lrf_mean", "r_mean",
                "diff_mean", "wilcoxon_q", "boot_diff", "boot_ci_low", "boot_ci_high", "boot_q"]
        print(df[cols].round(4).to_string(index=False))
    if trend is not None:
        print(f"\ntrend (Spearman, stage vs. per-seed LRF - R): rho = {trend.statistic:.3f}, p = {trend.pvalue:.4f}"
              "  (hypothesis predicts rho < 0)")
    if len(ref):
        print(f"reference {ref_cfg}: QWK {ref.mean():.4f} (n = {len(ref)})")

    # --- figure: (a) QWK by stage, (b) LRF - R with 95% bootstrap CI ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    x = df["stage_num"].to_numpy()
    for model, mcol, scol, lab in [("dann-lrf", "lrf_mean", "lrf_sd", "dANN (local RF)"),
                                   ("dann-r", "r_mean", "r_sd", "dANN (random)")]:
        st = style(model)
        ax1.errorbar(x, df[mcol], yerr=df[scol], marker=st["marker"], color=st["color"], capsize=3, label=lab)
    if len(ref):
        ax1.axhline(ref.mean(), ls="--", lw=1, color=style("vanilla")["color"], label="Dense ANN, pooled layer4")
    ax1.set_xticks(x, [f"layer{s}\n({g}x{g})" for s, g in zip(x, df["grid"])])
    ax1.set_xlabel("ResNet-18 stage (map grid)  → larger receptive fields")
    ax1.set_ylabel("Test QWK (mean ± SD over seeds)")
    ax1.set_title(f"(a) {args.somas} somas, identical parameter counts", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    err = np.vstack([df["boot_diff"] - df["boot_ci_low"], df["boot_ci_high"] - df["boot_diff"]])
    ax2.errorbar(x, df["boot_diff"], yerr=err, marker="o", color="#333333", capsize=4)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.margins(y=0.18)  # room for the q labels
    for xi, yi, q in zip(x, df["boot_ci_high"], df["boot_q"]):
        ax2.annotate(f"q={q:.2f}", (xi, yi), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
    ax2.set_xticks(x, [f"layer{s}" for s in x])
    ax2.set_xlabel("ResNet-18 stage")
    ax2.set_ylabel("QWK difference, LRF − random")
    title = "(b) locality effect, 95% paired-bootstrap CI"
    if trend is not None:
        title += f"\ntrend: Spearman ρ = {trend.statistic:.2f}, p = {trend.pvalue:.3f}"
    ax2.set_title(title, fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_rf_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_dir / 'rf_sweep.csv'} and {out_dir / 'fig_rf_sweep.png'}")


if __name__ == "__main__":
    main()
