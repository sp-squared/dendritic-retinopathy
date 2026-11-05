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

Confound control (--rf-sizes): with one fixed window (3x3) the window covers a growing
fraction of the image at deeper stages (0.15 of the width at layer1, 0.43 at layer4), so
depth and window size change together. --rf-sizes gives one window per stage, e.g.
6 4 3 2 keeps the window at ~30% of the map width at every stage (6/20, 4/14, 3/10, 2/7).
dANN-R runs do not depend on the window and are reused.

Run after training (see README, step 4c):
python scripts/11_receptive_field_sweep.py                       # fixed 3x3 window
python scripts/11_receptive_field_sweep.py --rf-sizes 6 4 3 2    # window ~30% of width
python scripts/11_receptive_field_sweep.py --somas 32            # secondary size, if trained
python scripts/11_receptive_field_sweep.py --features resnet18_224_spatial_l1 --rf-size 3 --tag l1_unpooled_rf3

By default the sweep uses the pooled stage features of step 4c (<prefix>_spatial_l1_g20,
_l2_g14, _l3_g10 and <prefix>_spatial); --features picks other feature sets (then --tag is
required so the default outputs are not overwritten).

Outputs (results/analysis/): rf_sweep.csv, fig_rf_sweep.png
(with --rf-sizes: rf_sweep_scaled.csv, fig_rf_sweep_scaled.png, or use --tag)
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
    p.add_argument("--rf-size", type=int, default=3, help="one window size for every stage")
    p.add_argument("--rf-sizes", type=int, nargs=4, default=None, metavar=("L1", "L2", "L3", "L4"),
                   help="window size per stage (layer1..layer4); overrides --rf-size")
    p.add_argument("--tag", default=None, help="output-name suffix (default 'scaled' with --rf-sizes)")
    p.add_argument("--features", nargs="+", default=None,
                   help="feature sets to analyse, one per stage (default: the pooled step-4c stages)")
    p.add_argument("--reference", default=None, help="config of a pooled reference head (default <prefix>_vanilla_h73_ce)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.prefix = "smoke"
    rf_of = (dict(zip([1, 2, 3, 4], args.rf_sizes)) if args.rf_sizes
             else {k: args.rf_size for k in [1, 2, 3, 4]})
    if args.features and not args.tag:
        raise SystemExit("--features needs --tag, so the default sweep outputs are not overwritten")
    tag = args.tag if args.tag is not None else ("scaled" if args.rf_sizes else "")
    suffix = f"_{tag}" if tag else ""

    root = results_root(args.smoke)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(root / "aptos_heads.csv")
    feats_col = runs["features"].astype(str)
    if args.features:
        wanted = feats_col.isin(args.features)
    elif args.smoke:
        wanted = feats_col.str.startswith(args.prefix) & feats_col.str.contains("spatial")
    else:  # the pooled stages of step 4c (unpooled or other grids need --features)
        wanted = feats_col.isin([f"{args.prefix}_spatial_l1_g20", f"{args.prefix}_spatial_l2_g14",
                                 f"{args.prefix}_spatial_l3_g10", f"{args.prefix}_spatial"])
    sel = runs[wanted
               & runs["model"].isin(["dann-lrf", "dann-r"])
               & (runs["somas"].map(clean) == str(args.somas))
               & (runs["dendrites"].map(clean) == str(args.dendrites))
               & (runs["synapses"].map(clean) == str(args.synapses))].copy()
    sel["stage"] = sel["features"].map(stage_of)
    if "rf_size" in sel:
        lrf = sel["model"] == "dann-lrf"
        want = sel["stage"].map(lambda k: str(rf_of[k]))
        sel = sel[~lrf | (sel["rf_size"].map(clean) == want)]
    if sel.empty:
        raise SystemExit("no matching spatial runs; check --prefix/--somas or train step 4c first")

    rows, per_seed = [], []
    for stage, g in sorted(sel.groupby("stage")):
        if g["features"].nunique() > 1:
            raise SystemExit(f"layer{stage}: several feature sets {sorted(g['features'].unique())}; pick one with --features")
        feats = g["features"].iloc[0]
        a = g[g["model"] == "dann-lrf"].set_index("seed")
        b = g[g["model"] == "dann-r"].set_index("seed")
        seeds = sorted(set(a.index) & set(b.index))
        if len(seeds) < 2:
            print(f"layer{stage}: fewer than 2 paired seeds (dANN-LRF rf {rf_of[stage]} trained?), skipped")
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
                     "rf_size": rf_of[stage], "window_fraction_of_width": round(rf_of[stage] / grid, 3),
                     "n_seeds": len(seeds),
                     "params": int(a["effective_params"].iloc[0]),
                     "lrf_mean": qa.mean(), "lrf_sd": qa.std(ddof=1), "r_mean": qb.mean(), "r_sd": qb.std(ddof=1),
                     "diff_mean": d.mean(), "diff_median": float(np.median(d)), "wilcoxon_p": w_p,
                     "boot_diff": bdiff, "boot_ci_low": lo, "boot_ci_high": hi, "boot_p": bp})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no stage has paired dANN-LRF and dANN-R runs")
    df["wilcoxon_q"] = bh_fdr(df["wilcoxon_p"])
    df["boot_q"] = bh_fdr(df["boot_p"])
    ps = pd.DataFrame(per_seed, columns=["stage", "diff"])
    trend = stats.spearmanr(ps["stage"], ps["diff"]) if ps["stage"].nunique() > 1 else None
    csv_path, fig_path = out_dir / f"rf_sweep{suffix}.csv", out_dir / f"fig_rf_sweep{suffix}.png"
    df.to_csv(csv_path, index=False)

    ref_cfg = args.reference or f"{args.prefix}_vanilla_h73_ce"
    ref = runs[runs["config"] == ref_cfg]["test_qwk"]

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        cols = ["stage", "grid", "rf_size", "window_fraction_of_width", "n_seeds", "params", "lrf_mean", "r_mean",
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
    ax1.set_xticks(x, [f"layer{s}\n({g}x{g}, rf {r})" for s, g, r in zip(x, df["grid"], df["rf_size"])])
    ax1.set_xlabel("ResNet-18 stage (map grid)  → larger receptive fields")
    ax1.set_ylabel("Test QWK (mean ± SD over seeds)")
    frac = df["window_fraction_of_width"]
    win = (f"window {frac.min():.0%}–{frac.max():.0%} of map width" if args.rf_sizes
           else f"{args.rf_size}x{args.rf_size} window")
    ax1.set_title(f"(a) {args.somas} somas, equal parameter counts, {win}", fontsize=9)
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
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
