"""Step 22 (extension, study 1): where does the dendritic advantage flip? Learning curve at 2048 px.

At 512 px with 44 training images, clustered dendrites found microaneurysms better than a dense
12 x 12 layer with the same parameters; at 2048 px with 427 images the dense layer did best. Two
things changed at once (resolution and data). Here resolution is fixed at 2048 px and only the
number of training images varies (script 20 --n-train: nested random subsets of the 427 IDRiD +
DDR training images, identical for every layer and seed; the same 40 validation images and the same
IDRiD (27) and DDR (225) test images throughout).

For each training-set size and test set:
  * microaneurysm FROC, microaneurysms found at 1 false positive per image, and mean FROC over
    the four lesion types, per first layer (mean, SD over seeds)
  * paired by seed (Wilcoxon, BH within each test set): every dendritic layer vs. dense 12 x 12
    (same 2,320 parameters), and clustered vs. scattered dendrites
  * crossover: the training-set size where (dendritic - dense 12 x 12) microaneurysm FROC changes
    sign from positive to negative, interpolated on log(n); 95% interval from a paired bootstrap
    over seeds (shares of bootstrap draws in which the dendritic layer stays ahead at every
    size, or behind at every size, are reported)

Prediction (stated before running): dendritic layers beat dense 12 x 12 at the smallest sizes and
lose at 427 images; the crossover lies between ~25 and ~200 images.

python scripts/22_learning_curve.py
python scripts/22_learning_curve.py --smoke

Full-data runs (427 images) are the step 4i/4j runs; runs without --n-train count as --n-full.
Outputs (results/analysis/): learning_curve_by_n.csv, learning_curve_tests.csv,
learning_curve_crossover.csv, fig_learning_curve.png
"""
from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from dendritic_dr.analysis import bh_fdr
from dendritic_dr.utils import results_root

D12 = ("conv", 12, 0)
LAYERS = [("dend-lrf", 12, 5), ("dend-lrf", 12, 8), ("dend-r", 12, 0), ("conv", 5, 0), D12]
STYLE = {("dend-lrf", 12, 5): ("#0072B2", "o"), ("dend-lrf", 12, 8): ("#56B4E9", "o"),
         ("dend-r", 12, 0): ("#E69F00", "s"), ("conv", 5, 0): ("#009E73", "D"), D12: ("#6E6E6E", "D")}
METRICS = ["froc_MA", "sens1_MA", "mfroc"]


def script21():
    spec = spec_from_file_location("fullres_comparison", Path(__file__).with_name("21_fullres_comparison.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(root: Path, n_full: int) -> pd.DataFrame:
    s21 = script21()
    runs = s21.add_keys(pd.read_csv(root / "fullres_runs.csv").drop_duplicates("run_id", keep="last"))
    n = runs["config"].str.extract(r"_n(\d+)_")[0].astype(float)
    if "n_train" in runs:
        n = n.where(n.notna(), runs["n_train"])
    runs["n"] = n.fillna(n_full).astype(int)
    keep = (runs["train_on"] == "idrid+ddr") & (runs["depth"] == 3) & (runs["loss"] == "tversky")
    return runs[keep & runs["key"].isin(LAYERS)].copy()


def crossover(ns, diffs):
    """First n where the mean difference goes from > 0 to <= 0, interpolated on log n.
    nan if it never does (always positive, always negative, or negative first)."""
    ns, d = np.asarray(ns, float), np.asarray(diffs, float)
    if len(d) < 2 or d[0] <= 0:
        return np.nan
    for i in range(1, len(d)):
        if d[i] <= 0:
            a, b = np.log(ns[i - 1]), np.log(ns[i])
            t = d[i - 1] / (d[i - 1] - d[i])
            return float(np.exp(a + t * (b - a)))
    return np.nan


def paired(a: pd.DataFrame, b: pd.DataFrame, m: str):
    seeds = a.index.intersection(b.index)
    x, y = a.loc[seeds, m].to_numpy(float), b.loc[seeds, m].to_numpy(float)
    d = x - y
    p = float(stats.wilcoxon(x, y).pvalue) if len(d) >= 2 and not np.allclose(d, 0) else np.nan
    return seeds, x, y, p


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-full", type=int, default=427, help="training images in runs made without --n-train")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    root = results_root(args.smoke)
    runs = load(root, 24 if args.smoke else args.n_full)
    test_sets = [ds for ds in ["idrid", "ddr"] if f"{ds}_froc_MA" in runs]
    label = script21().label
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    sizes = sorted(runs["n"].unique())
    print(f"training-set sizes: {sizes}; layers: {[label(k) for k in LAYERS if k in set(runs['key'])]}")

    # 1. mean / SD per layer and size
    cols = [f"{ds}_{m}" for ds in test_sets for m in METRICS] + ["best_epoch"]
    agg = runs.groupby(["n", "stem", "kernel", "cluster"])[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.insert(0, "n_seeds", runs.groupby(["n", "stem", "kernel", "cluster"]).size())
    agg.reset_index().to_csv(out / "learning_curve_by_n.csv", index=False)

    # 2. paired tests per size
    pairs = [(k, D12) for k in LAYERS if k[0].startswith("dend")] + [(("dend-lrf", 12, 5), ("dend-r", 12, 0))]
    rows = []
    for ds in test_sets:
        block = []
        for n in sizes:
            by = {k: s.set_index("seed") for k, s in runs[runs["n"] == n].groupby("key")}
            for a, b in pairs:
                if a not in by or b not in by:
                    continue
                for m in METRICS:
                    seeds, x, y, pv = paired(by[a], by[b], f"{ds}_{m}")
                    block.append({"test_set": ds, "n_train": n, "a": label(a), "b": label(b), "metric": m,
                                  "n_seeds": len(seeds), "mean_a": x.mean(), "mean_b": y.mean(),
                                  "diff": (x - y).mean(), "p": pv})
        for r, q in zip(block, bh_fdr([r["p"] for r in block]) if block else []):
            r["q"] = q
        rows += block
    tests = pd.DataFrame(rows)
    tests.to_csv(out / "learning_curve_tests.csv", index=False)

    # 3. crossover of (dendritic - dense 12 x 12) microaneurysm FROC, bootstrap over seeds
    rng = np.random.default_rng(0)
    cross, curves = [], {}
    for ds in test_sets:
        for a in [k for k in LAYERS if k[0].startswith("dend")]:
            mats, ns = [], []
            for n in sizes:
                by = {k: s.set_index("seed") for k, s in runs[runs["n"] == n].groupby("key")}
                if a not in by or D12 not in by:
                    continue
                seeds = by[a].index.intersection(by[D12].index)
                mats.append((by[a].loc[seeds, f"{ds}_froc_MA"] - by[D12].loc[seeds, f"{ds}_froc_MA"]).to_numpy(float))
                ns.append(n)
            if len(ns) < 2:
                continue
            k = min(len(m) for m in mats)
            D = np.stack([m[:k] for m in mats])  # sizes x seeds (seeds aligned by sorted seed id)
            boots = np.array([D[:, rng.integers(0, k, k)].mean(1) for _ in range(args.n_boot)])
            xs = np.array([crossover(ns, b) for b in boots])
            curves[(ds, a)] = (np.array(ns), D.mean(1), np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0))
            cross.append({"test_set": ds, "layer": label(a), "vs": label(D12), "sizes": " ".join(map(str, ns)),
                          "crossover_n": crossover(ns, D.mean(1)),
                          "ci_low": np.nanpercentile(xs, 2.5) if np.isfinite(xs).any() else np.nan,
                          "ci_high": np.nanpercentile(xs, 97.5) if np.isfinite(xs).any() else np.nan,
                          "share_no_crossover": float(np.isnan(xs).mean()),
                          "share_always_ahead": float((boots > 0).all(1).mean()),
                          "share_always_behind": float((boots <= 0).all(1).mean())})
    pd.DataFrame(cross).to_csv(out / "learning_curve_crossover.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        show = tests[tests["metric"] == "froc_MA"] if len(tests) else tests
        print(show.round(4).to_string(index=False) if len(show) else "no paired comparisons yet")
        print("\ncrossover (dendritic - dense 12x12 microaneurysm FROC goes from > 0 to <= 0):")
        print(pd.DataFrame(cross).round(3).to_string(index=False) if cross else "  not enough sizes yet")

    # 4. figure: MA FROC vs n (top), difference vs dense 12 x 12 with bootstrap band (bottom)
    fig, axes = plt.subplots(2, len(test_sets), figsize=(4.4 * len(test_sets), 5.6), squeeze=False, sharex=True)
    for j, ds in enumerate(test_sets):
        ax = axes[0, j]
        for k in LAYERS:
            s = runs[runs["key"] == k]
            if s.empty:
                continue
            g = s.groupby("n")[f"{ds}_froc_MA"].agg(["mean", "std"])
            c, mk = STYLE[k]
            ax.errorbar(g.index, g["mean"], yerr=g["std"], color=c, marker=mk, ms=4, lw=1.4, capsize=2, label=label(k))
        ax.set_title(f"{'IDRiD' if ds == 'idrid' else 'DDR'} test", fontsize=10)
        ax.set_ylabel("microaneurysm FROC")
        ax2 = axes[1, j]
        for (dsk, a), (ns, mean, lo, hi) in curves.items():
            if dsk != ds:
                continue
            c, mk = STYLE[a]
            ax2.fill_between(ns, lo, hi, color=c, alpha=0.15, lw=0)
            ax2.plot(ns, mean, color=c, marker=mk, ms=4, lw=1.4)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_ylabel("dendritic - dense 12x12\n(MA FROC, 95% bootstrap band)")
        ax2.set_xlabel("training images (log scale)")
        for a_ in (ax, ax2):
            a_.set_xscale("log")
            a_.set_xticks(sizes, [str(s) for s in sizes])
            a_.minorticks_off()
            a_.grid(color="#dddddd", lw=0.6)
            a_.set_axisbelow(True)
            a_.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Microaneurysm detection at 2048 px vs. training-set size (mean +/- SD over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_learning_curve.png", dpi=200)
    print(f"wrote {out / 'learning_curve_by_n.csv'}, {out / 'learning_curve_tests.csv'}, "
          f"{out / 'learning_curve_crossover.csv'}, {out / 'fig_learning_curve.png'}")


if __name__ == "__main__":
    main()
