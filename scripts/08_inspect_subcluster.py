"""Step 8 (exploratory): inspect a subcluster seen in a PCA plot from step 6.

Re-computes the PCA of one trained head's hidden activations (same code as
06_activation_analysis.py, so the coordinates match its fig_pca_*.png), selects
images of one grade inside a PC1/PC2 box (the subcluster) and the rest of that
grade (the comparison group), then compares the two groups on properties that
track acquisition rather than disease:

  * original resolution (width x height of the raw Kaggle image; APTOS images
    come from several cameras/clinics, and resolution is a strong camera proxy)
  * file size, mean brightness, contrast (SD of grey levels), sharpness
    (variance of the Laplacian) of the preprocessed image

Group differences are tested with Mann-Whitney U and Benjamini-Hochberg FDR.

Example (defaults match the size-matched dANN, seed 0, grade-0 subcluster at
upper left: PC1 < 0, PC2 > 2.5):
    python scripts/08_inspect_subcluster.py
    python scripts/08_inspect_subcluster.py --config resnet18_224_vanilla_h73_ce --pc2-min 2

Outputs (results/analysis/):
  subcluster_images.csv          per-image group, PCs and properties
  subcluster_stats.csv           group medians, Mann-Whitney p and q per property
  fig_subcluster_pca.png         PCA with the selected box highlighted (check the selection)
  fig_subcluster_contact.png     example images: subcluster (top) vs. comparison (bottom)
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats

from dendritic_dr.analysis import bh_fdr, hidden_activations, load_checkpoint, pca
from dendritic_dr.features import load_features, to_tensors
from dendritic_dr.utils import DATA_DIR, results_root

PROPS = ["orig_width", "orig_height", "file_kb", "brightness", "contrast", "sharpness"]


def image_props(raw_path, proc_path):
    with Image.open(raw_path) as im:  # header only: cheap
        w, h = im.size
    g = np.asarray(Image.open(proc_path).convert("L"), dtype=np.float32)
    fundus = g[g > 7]  # ignore black border
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return {
        "orig_width": w,
        "orig_height": h,
        "file_kb": raw_path.stat().st_size / 1024,
        "brightness": float(fundus.mean()) if fundus.size else 0.0,
        "contrast": float(fundus.std()) if fundus.size else 0.0,
        "sharpness": float(lap.var()),
    }


def contact_sheet(df, proc_dir, out, n=8, seed=0):
    groups = [("subcluster", "Subcluster"), ("comparison", "Comparison")]
    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.8))
    for r, (g, title) in enumerate(groups):
        sub = df[df["group"] == g]
        picks = sub.sample(min(n, len(sub)), random_state=seed) if len(sub) else sub
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                row = picks.iloc[c]
                ax.imshow(Image.open(proc_dir / f"{row['id_code']}.png"))
                ax.set_title(f"{int(row['orig_width'])}x{int(row['orig_height'])}", fontsize=6)
        axes[r, 0].text(-0.15, 0.5, title, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=9)
    fig.suptitle("Random examples (title = original resolution)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="resnet18_224_dann-r_s128_d16_y16_ce")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--grade", type=int, default=0)
    p.add_argument("--pc1-max", type=float, default=0.0)
    p.add_argument("--pc1-min", type=float, default=-np.inf)
    p.add_argument("--pc2-min", type=float, default=2.5)
    p.add_argument("--pc2-max", type=float, default=np.inf)
    p.add_argument("--raw-dir", default=str(DATA_DIR / "raw" / "aptos" / "train_images"))
    p.add_argument("--proc-dir", default=str(DATA_DIR / "processed" / "aptos_512"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    from pathlib import Path

    root = results_root(args.smoke)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.config}_seed{args.seed}"
    model, ckpt = load_checkpoint(root / "models" / f"{run_id}.pt")
    fname = ckpt["model_args"]["features"]
    raw = load_features(fname, smoke=(fname == "smoke"))
    x, yt = to_tensors(raw, "cpu", ckpt["feature_mean"], ckpt["feature_std"])[args.split]
    y = yt.numpy()
    ids = np.asarray(raw[f"ids_{args.split}"]).astype(str)
    scores, var = pca(hidden_activations(model, x), 2)

    df = pd.DataFrame({"id_code": ids, "grade": y, "pc1": scores[:, 0], "pc2": scores[:, 1]})
    in_box = (df["pc1"].between(args.pc1_min, args.pc1_max)) & (df["pc2"].between(args.pc2_min, args.pc2_max))
    df["group"] = np.where(df["grade"] != args.grade, "other_grade",
                           np.where(in_box, "subcluster", "comparison"))
    n_sub, n_cmp = (df["group"] == "subcluster").sum(), (df["group"] == "comparison").sum()
    print(f"{run_id}: grade {args.grade} -> subcluster n={n_sub}, comparison n={n_cmp}")
    # PCA with the box highlighted, to confirm the selection matches the cluster you saw
    fig, ax = plt.subplots(figsize=(5, 4.2))
    colors = {"other_grade": "#cccccc", "comparison": "#3b528b", "subcluster": "#e6550d"}
    labels = {"other_grade": "other grades", "comparison": f"grade {args.grade}, main group",
              "subcluster": f"grade {args.grade}, subcluster"}
    for g in ["other_grade", "comparison", "subcluster"]:
        m = df["group"] == g
        ax.scatter(df.loc[m, "pc1"], df.loc[m, "pc2"], s=10, alpha=0.8, color=colors[g], label=f"{labels[g]} (n={m.sum()})")
    ax.set_xlabel(f"PC1 ({var[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.0%} variance)")
    lo1 = max(args.pc1_min, df["pc1"].min()); hi1 = min(args.pc1_max, df["pc1"].max())
    lo2 = max(args.pc2_min, df["pc2"].min()); hi2 = min(args.pc2_max, df["pc2"].max())
    ax.add_patch(plt.Rectangle((lo1, lo2), hi1 - lo1, hi2 - lo2, fill=False, ls="--", lw=1, color="#e6550d"))
    ax.set_title(f"{run_id}: subcluster selection (dashed box)", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_subcluster_pca.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out_dir / 'fig_subcluster_pca.png'}")
    if n_sub < 3:
        raise SystemExit("fewer than 3 images in the box: look at fig_subcluster_pca.png and adjust "
                         "--pc1-min/--pc1-max/--pc2-min/--pc2-max")

    raw_dir, proc_dir = Path(args.raw_dir), Path(args.proc_dir)
    sel = df[df["group"] != "other_grade"].copy()
    props = [image_props(raw_dir / f"{i}.png", proc_dir / f"{i}.png") for i in sel["id_code"]]
    sel = pd.concat([sel.reset_index(drop=True), pd.DataFrame(props)], axis=1)
    sel["resolution"] = sel["orig_width"].astype(int).astype(str) + "x" + sel["orig_height"].astype(int).astype(str)
    sel.to_csv(out_dir / "subcluster_images.csv", index=False)

    a, b = sel[sel["group"] == "subcluster"], sel[sel["group"] == "comparison"]
    rows = []
    for prop in PROPS:
        pval = stats.mannwhitneyu(a[prop], b[prop]).pvalue if a[prop].nunique() + b[prop].nunique() > 2 else np.nan
        rows.append({"property": prop, "subcluster_median": a[prop].median(),
                     "comparison_median": b[prop].median(), "mannwhitney_p": pval})
    st = pd.DataFrame(rows)
    st["q"] = bh_fdr(st["mannwhitney_p"])
    st.to_csv(out_dir / "subcluster_stats.csv", index=False)

    with pd.option_context("display.width", 140):
        print("\nProperty comparison (Mann-Whitney U, BH-corrected):")
        print(st.round(4).to_string(index=False))
        for g, d in [("subcluster", a), ("comparison", b)]:
            top = d["resolution"].value_counts().head(5)
            share = (top / len(d) * 100).round(0).astype(int).astype(str) + "%"
            print(f"\nMost common original resolutions, {g} (n={len(d)}):")
            print(pd.DataFrame({"n": top, "share": share}).to_string())
    contact_sheet(sel, proc_dir, out_dir / "fig_subcluster_contact.png", seed=args.seed)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
