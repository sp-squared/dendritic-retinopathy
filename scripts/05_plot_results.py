"""Step 5: figures for the report.

  results/figures/aptos_qwk_vs_params.png   test QWK (mean +/- SD over seeds) vs. effective params
  results/figures/aptos_confusion_<model>.png  summed test confusion matrix of each model's
                                               best configuration (row-normalised)
  results/figures/fmnist_acc_vs_params.png  same idea for the Fashion-MNIST replication

Example
-------
python scripts/05_plot_results.py
python scripts/05_plot_results.py --smoke     # plots from results/smoke/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (works over SSH / headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dendritic_dr.plotting import config_label, display_name, format_params_axis, style
from dendritic_dr.utils import results_root

CONFIG_COLS = ["somas", "dendrites", "synapses", "rf_size", "hidden", "dropout", "loss", "features"]
GRADES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 PDR"]


def config_key(df):
    if "config" in df.columns:
        return df["config"].astype(str)
    cols = [c for c in CONFIG_COLS if c in df.columns]
    return df[["model"] + cols].astype(object).fillna("").astype(str).agg("|".join, axis=1)


def _clean(v) -> str:
    """CSV cell -> tidy string: 16.0 -> '16', '[128]' -> '128', NaN -> ''."""
    if v is None or (isinstance(v, float) and v != v) or str(v) == "":
        return ""
    t = str(v).strip("[]")
    try:
        f = float(t)
        return str(int(f)) if f.is_integer() else t
    except ValueError:
        return t


def _lines(g):
    """Split configs into plotted lines.

    Main line per model: its size sweep at the most common synapses-per-dendrite.
    If a model also has runs at other synapse counts (an ablation), those form a
    separate dashed line together with the main-sweep point at the same soma
    count, so the ablation is never joined onto the size sweep.
    Returns a list of (model, label, is_ablation, sub-DataFrame).
    """
    out = []
    order = ["dann-lrf", "dann-r", "vanilla"]
    for model in sorted(g["model"].unique(), key=lambda m: order.index(m) if m in order else 99):
        sub = g[g["model"] == model]
        syns = sub["_syn"][sub["_syn"] != ""]
        if syns.nunique() <= 1:
            out.append((model, display_name(model), False, sub))
            continue
        base = syns.mode().iloc[0]
        main = sub[sub["_syn"] == base]
        out.append((model, f"{display_name(model)}, {base} synapses/dendrite", False, main))
        abl = sub[sub["_syn"] != base]
        for somas in abl["_somas"].unique():
            line = pd.concat([main[main["_somas"] == somas], abl[abl["_somas"] == somas]])
            lo, hi = line["_syn"].astype(float).min(), line["_syn"].astype(float).max()
            out.append((model, f"{display_name(model)}, {somas} somas, {lo:g}→{hi:g} synapses", True, line))
    return out


def metric_vs_params(df, metric, ylabel, out):
    df = df.assign(config=config_key(df))
    df = df.assign(
        _syn=df["synapses"].map(_clean) if "synapses" in df else "",
        _somas=df["somas"].map(_clean) if "somas" in df else "",
    )
    g = df.groupby(["model", "config", "_syn", "_somas"]).agg(
        params=("effective_params", "first"), mean=(metric, "mean"), sd=(metric, "std"), n=(metric, "size")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for model, label, is_abl, sub in _lines(g):
        sub = sub.sort_values("params")
        st = style(model)
        ax.errorbar(sub["params"], sub["mean"], yerr=sub["sd"].fillna(0), capsize=3, markersize=6,
                    linewidth=1.5, color=st["color"], marker=st["marker"], label=label,
                    linestyle="--" if is_abl else "-",
                    markerfacecolor="white" if is_abl else st["color"])
    ax.set_xscale("log")
    format_params_axis(ax)
    ax.set_xlabel("Effective trainable parameters (log scale)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(title=f"mean ± SD over {int(g['n'].max())} seeds", title_fontsize=8, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    return g.drop(columns=["_syn", "_somas"])


def confusion_plots(df, out_dir: Path):
    df = df.assign(config=config_key(df))
    for model, sub in df.groupby("model"):
        best_cfg = sub.groupby("config")["val_qwk"].mean().idxmax()  # choose by validation, not test
        cms = [np.array(json.loads(c)) for c in sub[sub["config"] == best_cfg]["test_confusion"]]
        cm = np.sum(cms, axis=0).astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks(range(5), GRADES, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(5), GRADES, fontsize=8)
        ax.set_xlabel("Predicted grade")
        ax.set_ylabel("True grade")
        label = config_label(sub[sub["config"] == best_cfg].iloc[0])
        ax.set_title(f"{label}\n(test set, summed over {len(cms)} seeds)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        out = out_dir / f"aptos_confusion_{model}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="read/write under results/smoke/")
    p.add_argument("--aptos-csv", default=None, help="defaults to <results>/aptos_heads.csv")
    p.add_argument("--fmnist-csv", default=None, help="defaults to <results>/fmnist.csv")
    args = p.parse_args()
    root = results_root(args.smoke)
    args.aptos_csv = args.aptos_csv or str(root / "aptos_heads.csv")
    args.fmnist_csv = args.fmnist_csv or str(root / "fmnist.csv")
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    aptos, fmnist = Path(args.aptos_csv), Path(args.fmnist_csv)
    if aptos.exists():
        df = pd.read_csv(aptos)
        summary = metric_vs_params(df, "test_qwk", "Test quadratic weighted kappa", out_dir / "aptos_qwk_vs_params.png")
        summary.to_csv(out_dir / "aptos_summary.csv", index=False)
        confusion_plots(df, out_dir)
    else:
        print(f"skipping APTOS plots: {aptos} not found")
    if fmnist.exists():
        metric_vs_params(pd.read_csv(fmnist), "test_acc", "Test accuracy", out_dir / "fmnist_acc_vs_params.png")
    else:
        print(f"skipping Fashion-MNIST plot: {fmnist} not found")


if __name__ == "__main__":
    main()
