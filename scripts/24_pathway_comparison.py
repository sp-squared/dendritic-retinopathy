"""Step 24 (extension, study 2): compare the pathway-wired dANN with its controls (step 23).

1. AUC per model: per repeat (mean, SD over the 10 repeats of 5-fold CV) and for the
   repeat-averaged out-of-fold probability, with a 95% bootstrap interval over donors.
2. Paired comparisons of pathway-dann with each other model: difference in AUC of the
   repeat-averaged probabilities, 95% interval and two-sided p from a paired bootstrap over
   donors (a donor's samples are resampled together), and Wilcoxon over the per-repeat AUCs
   (repeats share data, so this p is optimistic); Benjamini-Hochberg across the comparisons.
3. Which pathways drove the predictions: dendrites ranked by mean |gradient x activation|.
   Prediction check: are pathways matching the course project's enrichment themes (immune,
   extracellular matrix / collagen / integrin, TGF-beta / BMP) over-represented among the top
   --top dendrites? (Fisher's exact test against all pathway dendrites.) The top<k>_share column
   (from step 23) gives how often each pathway made the top 25 across CV repeats: one model's
   attributions are unstable, so read only pathways that recur.

python scripts/24_pathway_comparison.py
python scripts/24_pathway_comparison.py --name dr_vs_diabetic
python scripts/24_pathway_comparison.py --smoke

Outputs (results/pathway/<name>/): auc_by_model.csv, auc_tests.csv, top_pathways.csv, fig_pathway.png
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
from sklearn.metrics import roc_auc_score

from dendritic_dr.analysis import bh_fdr
from dendritic_dr.utils import results_root

ORDER = ["pathway-dann", "random-dann", "dense-mlp", "logreg", "pathway-score"]
LABEL = {"pathway-dann": "pathway dANN", "random-dann": "random-gene dANN", "dense-mlp": "dense MLP (matched)",
         "logreg": "logistic regression", "pathway-score": "pathway scores + logistic"}
COLOR = {"pathway-dann": "#0072B2", "random-dann": "#E69F00", "dense-mlp": "#6E6E6E", "logreg": "#009E73",
         "pathway-score": "#CC79A7"}
THEMES = re.compile(r"immun|interleukin|interferon|cytokine|chemokine|complement|toll|neutrophil|antigen|"
                    r"extracellular matrix|collagen|integrin|elastic fib|laminin|tgf|bmp|smad", re.I)


def donor_bootstrap(df: pd.DataFrame, cols: list[str], n_boot: int, seed: int = 0):
    """AUC of each column on donor-level bootstrap resamples (n_boot x len(cols))."""
    rng = np.random.default_rng(seed)
    donors = df["donor"].unique()
    by = {d: g for d, g in df.groupby("donor")}
    out = []
    for _ in range(n_boot):
        s = pd.concat([by[d] for d in rng.choice(donors, len(donors), replace=True)])
        if s["y"].nunique() < 2:
            continue
        out.append([roc_auc_score(s["y"], s[c]) for c in cols])
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="dr_vs_control")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.name = "smoke"
    out = results_root(args.smoke) / "pathway" / args.name
    oof = pd.read_csv(out / "oof.csv")
    models = [m for m in ORDER if m in set(oof["model"])]

    # 1. AUC per repeat and of the repeat-averaged probabilities
    per_rep = oof.groupby(["model", "repeat"]).apply(lambda g: roc_auc_score(g["y"], g["prob"]),
                                                    include_groups=False).unstack(0)[models]
    avg = oof.groupby(["model", "title", "donor", "y"])["prob"].mean().unstack(0).reset_index()
    boot = donor_bootstrap(avg, models, args.n_boot)
    rows = []
    for j, m in enumerate(models):
        rows.append({"model": m, "label": LABEL[m], "auc_mean_over_repeats": per_rep[m].mean(),
                     "auc_sd_over_repeats": per_rep[m].std(), "auc_averaged": roc_auc_score(avg["y"], avg[m]),
                     "ci_low": np.percentile(boot[:, j], 2.5), "ci_high": np.percentile(boot[:, j], 97.5)})
    by_model = pd.DataFrame(rows)
    by_model.to_csv(out / "auc_by_model.csv", index=False)

    # 2. pathway-dann vs. each other model
    tests = []
    if "pathway-dann" in models:
        i0 = models.index("pathway-dann")
        for j, m in enumerate(models):
            if m == "pathway-dann":
                continue
            d = boot[:, i0] - boot[:, j]
            obs = roc_auc_score(avg["y"], avg["pathway-dann"]) - roc_auc_score(avg["y"], avg[m])
            p_boot = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            x, z = per_rep["pathway-dann"], per_rep[m]
            p_w = float(stats.wilcoxon(x, z).pvalue) if len(x) > 1 and not np.allclose(x, z) else np.nan
            tests.append({"a": "pathway-dann", "b": m, "auc_a": by_model.loc[i0, "auc_averaged"],
                          "auc_b": by_model.loc[j, "auc_averaged"], "diff": obs,
                          "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                          "p_bootstrap": p_boot, "p_wilcoxon_repeats": p_w})
        if tests:
            for r, q in zip(tests, bh_fdr([r["p_bootstrap"] for r in tests])):
                r["q_bootstrap"] = q
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "auc_tests.csv", index=False)

    # 3. top pathways and the enrichment-theme check
    contrib = pd.read_csv(out / "contributions.csv")
    contrib["theme"] = contrib["name"].astype(str).str.contains(THEMES)
    top = contrib.head(args.top).copy()
    a, b = int(top["theme"].sum()), args.top - int(top["theme"].sum())
    c, d = int(contrib["theme"].sum()) - a, len(contrib) - args.top - (int(contrib["theme"].sum()) - a)
    odds, p_theme = stats.fisher_exact([[a, b], [c, d]], alternative="greater") if len(contrib) > args.top else (np.nan, np.nan)
    top.to_csv(out / "top_pathways.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(by_model.round(3).to_string(index=False))
        print()
        print(tests.round(4).to_string(index=False) if len(tests) else "no comparisons")
        print(f"\ntop {args.top} pathway dendrites: {a} match the enrichment themes vs. "
              f"{contrib['theme'].mean():.0%} of all {len(contrib)} (Fisher one-sided p = {p_theme:.3g})")
        stab = [c for c in top.columns if c.endswith("_share")]
        print(top[["name", "size", "categories", "mean_abs_contribution", "case_minus_control"] + stab].head(15)
              .round(4).to_string(index=False))

    # 4. figure: AUC per model (left), top pathways (right)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        r = by_model.iloc[i]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(per_rep)), per_rep[m], s=10, color=COLOR[m], alpha=0.5, lw=0)
        ax.errorbar(i, r["auc_averaged"], yerr=[[r["auc_averaged"] - r["ci_low"]], [r["ci_high"] - r["auc_averaged"]]],
                    fmt="D", color="black", ms=5, capsize=3)
    ax.axhline(0.5, color="#888888", ls=":", lw=1)
    ax.set_xticks(range(len(models)), [LABEL[m] for m in models], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC (dots: CV repeats; diamond: averaged,\n95% donor-bootstrap interval)")
    ax.set_title("(a) Classification", fontsize=9, loc="left")
    k = min(15, len(top))
    t = top.head(k).iloc[::-1]
    colors = ["#D55E00" if v > 0 else "#0072B2" for v in t["case_minus_control"]]
    bx.barh(range(k), t["mean_abs_contribution"], color=colors)
    bx.set_yticks(range(k), [n[:55] + ("..." if len(n) > 55 else "") for n in t["name"].astype(str)], fontsize=7)
    bx.set_xlabel("mean |gradient x activation| of the dendrite")
    bx.set_title("(b) Most influential pathway dendrites (orange: pushes toward case)", fontsize=9, loc="left")
    for a_ in (ax, bx):
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(axis="x" if a_ is bx else "y", color="#dddddd", lw=0.6)
        a_.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "fig_pathway.png", dpi=200)
    print(f"wrote {out / 'auc_by_model.csv'}, {out / 'auc_tests.csv'}, {out / 'top_pathways.csv'}, {out / 'fig_pathway.png'}")


if __name__ == "__main__":
    main()
