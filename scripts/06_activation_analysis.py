"""Step 6: analyse trained heads like a gene-expression experiment.

For each selected configuration (all seeds), test images are passed through the
trained head and the last hidden layer is recorded, giving an images x units
matrix (the analogue of samples x genes). Then:

  * differential activation: Kruskal-Wallis across DR grades + Spearman trend,
    Benjamini-Hochberg FDR across units
  * selectivity: how many grades each unit responds to (tests the dANN
    "mixed selectivity" claim of Chavlis & Poirazi 2025 on clinical data)
  * representation: PCA and the |rho(PC1, grade)| "progression score"
  * figures: clustered heatmap of significant units, PCA by grade, selectivity

By default the best configuration (highest mean validation QWK) of each model
type is analysed; pass --configs to choose explicitly (config tags are in the
`config` column of results/aptos_heads.csv).

Examples
--------
python scripts/06_activation_analysis.py --features resnet18_224
python scripts/06_activation_analysis.py --configs resnet18_224_dann-r_s128_d16_y16_ce resnet18_224_vanilla_h64_ce
python scripts/06_activation_analysis.py --smoke

Outputs (under results/analysis/, or results/smoke/analysis/):
  activation_summary.csv              one row per run
  activation_summary_by_config.csv    mean/SD over seeds
  units/<run_id>.csv                  per-unit statistics (the "DE table")
  fig_heatmap_<config>.png, fig_pca_<config>.png, fig_selectivity.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import leaves_list, linkage

from dendritic_dr.analysis import (
    differential_activation,
    hidden_activations,
    load_checkpoint,
    pca,
    progression_score,
)
from dendritic_dr.features import load_features, to_tensors
from dendritic_dr.utils import results_root

GRADE_LABELS = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 PDR"]
GRADE_CMAP = plt.get_cmap("viridis", 5)  # ordered palette for ordered grades


def select_configs(runs: pd.DataFrame, requested):
    if requested:
        missing = set(requested) - set(runs["config"])
        if missing:
            raise SystemExit(f"configs not found in results: {sorted(missing)}")
        return list(requested)
    best = runs.groupby(["model", "config"])["val_qwk"].mean().reset_index()
    return best.loc[best.groupby("model")["val_qwk"].idxmax(), "config"].tolist()


def heatmap(A, y, units_df, title, out, fdr, top_k=60):
    sig = units_df[(units_df["kruskal_q"] < fdr)].nsmallest(top_k, "kruskal_q")
    if len(sig) < 2:  # fall back to the most variable units so a figure still exists
        sig = units_df[units_df["active"]].assign(v=A.std(0)[units_df["active"]]).nlargest(top_k, "v")
        title += " (no FDR-significant units; most variable shown)"
    if len(sig) < 2:
        print(f"  skipping heatmap for {title}: fewer than 2 active units")
        return
    M = A[:, sig["unit"].to_numpy()]
    M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    order_img = np.argsort(y, kind="stable")
    order_unit = leaves_list(linkage(M.T, method="average", metric="correlation"))
    M = M[order_img][:, order_unit].T
    fig = plt.figure(figsize=(7, 5.4))
    ax_bar = fig.add_axes([0.1, 0.88, 0.75, 0.04])
    ax = fig.add_axes([0.1, 0.17, 0.75, 0.69])
    cax = fig.add_axes([0.87, 0.17, 0.02, 0.69])
    ax_bar.imshow(y[order_img][None, :], aspect="auto", cmap=GRADE_CMAP, vmin=-0.5, vmax=4.5)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([0], ["Grade"])
    ax_bar.set_title(title, fontsize=9)
    im = ax.imshow(np.clip(M, -3, 3), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
    ax.set_xlabel("Test images (sorted by DR grade)")
    ax.set_ylabel(f"Hidden units (n={M.shape[0]}, clustered)")
    ax.set_yticks([])
    fig.colorbar(im, cax=cax, label="Activation (z-score)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GRADE_CMAP(g)) for g in range(5)]
    fig.legend(handles, GRADE_LABELS, loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.475, 0.0))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def pca_plot(A, y, title, out):
    scores, var = pca(A, 2)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for g in range(5):
        m = y == g
        ax.scatter(scores[m, 0], scores[m, 1], s=10, alpha=0.7, color=GRADE_CMAP(g), label=GRADE_LABELS[g])
    ax.set_xlabel(f"PC1 ({var[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.0%} variance)" if len(var) > 1 else "PC2")
    ax.set_title(f"{title}\nprogression score |rho(PC1, grade)| = {progression_score(A, y):.2f}", fontsize=9)
    ax.legend(fontsize=7, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def selectivity_plot(unit_tables: dict, out):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bins = np.arange(0.5, 6.5, 1)
    width = 0.8 / max(1, len(unit_tables))
    for i, (config, df) in enumerate(unit_tables.items()):
        vals = df["n_grades_responsive"].dropna()
        counts, _ = np.histogram(vals, bins=bins)
        ax.bar(np.arange(1, 6) - 0.4 + width * (i + 0.5), counts / max(1, counts.sum()), width=width, label=config)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Number of DR grades a unit responds to\n(1 = grade-specific, 5 = mixed selectivity)")
    ax.set_ylabel("Fraction of active units")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=None, help="restrict to runs on these features (e.g. resnet18_224)")
    p.add_argument("--configs", nargs="+", default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    root = results_root(args.smoke)
    out_dir = root / "analysis"
    (out_dir / "units").mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(root / "aptos_heads.csv")
    if args.features:
        runs = runs[runs["features"] == args.features]
    configs = select_configs(runs, args.configs)
    print("analysing configs:", *configs, sep="\n  ")

    feats_cache, summary, pooled_units = {}, [], {}
    for config in configs:
        cruns = runs[runs["config"] == config].sort_values("seed")
        for i, row in enumerate(cruns.itertuples()):
            model, ckpt = load_checkpoint(root / "models" / f"{row.run_id}.pt")
            fname = ckpt["model_args"]["features"]
            if fname not in feats_cache:
                feats_cache[fname] = load_features(fname, smoke=(fname == "smoke"))
            raw = feats_cache[fname]
            x, yt = to_tensors(raw, "cpu", ckpt["feature_mean"], ckpt["feature_std"])[args.split]
            y = yt.numpy()
            A = hidden_activations(model, x)
            units = differential_activation(A, y)
            units.to_csv(out_dir / "units" / f"{row.run_id}.csv", index=False)
            pooled_units.setdefault(config, []).append(units)
            act = units[units["active"]]
            summary.append(
                {
                    "run_id": row.run_id,
                    "config": config,
                    "model": row.model,
                    "seed": row.seed,
                    "test_qwk": row.test_qwk,
                    "n_units": len(units),
                    "n_active": int(units["active"].sum()),
                    "frac_dead": 1 - units["active"].mean(),
                    "n_sig_kruskal": int((units["kruskal_q"] < args.fdr).sum()),
                    "frac_sig_of_active": float((act["kruskal_q"] < args.fdr).mean()) if len(act) else np.nan,
                    "n_sig_increasing": int(((units["spearman_q"] < args.fdr) & (units["spearman_rho"] > 0)).sum()),
                    "n_sig_decreasing": int(((units["spearman_q"] < args.fdr) & (units["spearman_rho"] < 0)).sum()),
                    "median_selectivity_index": float(act["selectivity_index"].median()) if len(act) else np.nan,
                    "mean_grades_responsive": float(act["n_grades_responsive"].mean()) if len(act) else np.nan,
                    "progression_score": progression_score(A, y),
                }
            )
            if i == 0:  # figures from the lowest seed of each config
                heatmap(A, y, units, f"{config} (seed {row.seed})", out_dir / f"fig_heatmap_{config}.png", args.fdr)
                pca_plot(A, y, f"{config} (seed {row.seed})", out_dir / f"fig_pca_{config}.png")

    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / "activation_summary.csv", index=False)
    num = sdf.drop(columns=["run_id", "model", "seed"]).groupby("config")
    by_cfg = num.agg(["mean", "std"])
    by_cfg.columns = [f"{a}_{b}" for a, b in by_cfg.columns]
    by_cfg.insert(0, "n_seeds", num.size())
    by_cfg.to_csv(out_dir / "activation_summary_by_config.csv")
    selectivity_plot({c: pd.concat(v) for c, v in pooled_units.items()}, out_dir / "fig_selectivity.png")

    cols = ["n_seeds", "test_qwk_mean", "frac_sig_of_active_mean", "mean_grades_responsive_mean", "progression_score_mean"]
    with pd.option_context("display.width", 160, "display.max_columns", 10):
        print("\n", by_cfg[cols].round(3).to_string())
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
