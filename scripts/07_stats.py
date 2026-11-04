"""Step 7: statistics for the model comparison.

1. Per run: 95% percentile-bootstrap CI of test QWK (resampling test images).
2. Per configuration: mean, SD and 95% t-interval of test QWK over seeds.
3. Paired comparisons, dendritic vs. dense:
   * Wilcoxon signed-rank test on test QWK, paired by seed (variability from
     training randomness). With n seeds the smallest attainable two-sided
     p-value is 2/2^n, so use >= 6 seeds (10 recommended).
   * Paired bootstrap of the mean-over-seeds QWK difference on the same test
     images (variability from the finite test set).
   * Benjamini-Hochberg correction across all comparisons.
   By default every dendritic config is paired with the dense config closest in
   effective parameter count; use --pair A B (repeatable) to choose explicitly.

Examples
--------
python scripts/07_stats.py
python scripts/07_stats.py --pair resnet18_224_dann-r_s128_d16_y16_ce resnet18_224_vanilla_h64_ce
python scripts/07_stats.py --smoke

Outputs (results/analysis/): stats_runs.csv, stats_by_config.csv, stats_paired.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from dendritic_dr.analysis import bh_fdr, bootstrap_qwk_ci, paired_bootstrap_diff
from dendritic_dr.utils import results_root


def load_preds(root, run_ids):
    """Stack per-image predictions of several runs, aligned on id_code."""
    frames = [pd.read_csv(root / "predictions" / f"{r}.csv").set_index("id_code") for r in run_ids]
    ids = frames[0].index
    for f in frames[1:]:
        if not f.index.equals(ids):
            raise SystemExit("prediction files are not aligned on the same test images")
    return frames[0]["true"].to_numpy(), np.stack([f["pred"].to_numpy() for f in frames])


def auto_pairs(runs):
    params = runs.groupby("config").agg(model=("model", "first"), params=("effective_params", "first"))
    dense = params[params["model"] == "vanilla"]
    if dense.empty:
        return []
    pairs = []
    for cfg, r in params[params["model"] != "vanilla"].iterrows():
        nearest = (np.log(dense["params"]) - np.log(r["params"])).abs().idxmin()
        pairs.append((cfg, nearest))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=None)
    p.add_argument("--pair", nargs=2, action="append", metavar=("CONFIG_A", "CONFIG_B"))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    root = results_root(args.smoke)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(root / "aptos_heads.csv")
    if args.features:
        runs = runs[runs["features"] == args.features]

    # 1. per-run bootstrap CIs
    rows = []
    for r in runs.itertuples():
        y, pr = load_preds(root, [r.run_id])
        lo, hi = bootstrap_qwk_ci(y, pr[0], n_boot=args.n_boot, seed=int(r.seed))
        rows.append({"run_id": r.run_id, "config": r.config, "model": r.model, "seed": r.seed,
                     "effective_params": r.effective_params, "test_qwk": r.test_qwk, "ci_low": lo, "ci_high": hi})
    run_df = pd.DataFrame(rows)
    run_df.to_csv(out_dir / "stats_runs.csv", index=False)

    # 2. per-config summary over seeds
    def t_ci(v):
        v = np.asarray(v, dtype=float)
        if len(v) < 2:
            return (np.nan, np.nan)
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return (v.mean() - h, v.mean() + h)

    cfg_rows = []
    for cfg, g in run_df.groupby("config"):
        lo, hi = t_ci(g["test_qwk"])
        cfg_rows.append({"config": cfg, "model": g["model"].iloc[0], "effective_params": g["effective_params"].iloc[0],
                         "n_seeds": len(g), "qwk_mean": g["test_qwk"].mean(), "qwk_sd": g["test_qwk"].std(ddof=1),
                         "qwk_seed_ci_low": lo, "qwk_seed_ci_high": hi,
                         "mean_bootstrap_ci_width": (g["ci_high"] - g["ci_low"]).mean()})
    cfg_df = pd.DataFrame(cfg_rows).sort_values(["model", "effective_params"])
    cfg_df.to_csv(out_dir / "stats_by_config.csv", index=False)

    # 3. paired comparisons
    pairs = [tuple(x) for x in args.pair] if args.pair else auto_pairs(runs)
    pair_rows = []
    for a, b in pairs:
        ra, rb = runs[runs["config"] == a], runs[runs["config"] == b]
        seeds = sorted(set(ra["seed"]) & set(rb["seed"]))
        if not seeds:
            print(f"no common seeds for {a} vs {b}; skipped")
            continue
        qa = ra.set_index("seed").loc[seeds, "test_qwk"].to_numpy()
        qb = rb.set_index("seed").loc[seeds, "test_qwk"].to_numpy()
        w_p = np.nan
        if len(seeds) >= 2 and np.any(qa != qb):
            w_p = float(stats.wilcoxon(qa, qb).pvalue)
        if len(seeds) < 6:
            print(f"note: {a} vs {b} has only {len(seeds)} paired seeds; "
                  f"minimum attainable Wilcoxon p = {2 / 2 ** len(seeds):.3f}")
        y, pa = load_preds(root, ra.set_index("seed").loc[seeds, "run_id"])
        _, pb = load_preds(root, rb.set_index("seed").loc[seeds, "run_id"])
        diff, lo, hi, boot_p = paired_bootstrap_diff(y, pa, pb, n_boot=args.n_boot)
        pair_rows.append({"config_a": a, "config_b": b, "n_seeds": len(seeds),
                          "params_a": int(ra["effective_params"].iloc[0]), "params_b": int(rb["effective_params"].iloc[0]),
                          "qwk_a_mean": qa.mean(), "qwk_b_mean": qb.mean(), "median_seed_diff": float(np.median(qa - qb)),
                          "wilcoxon_p": w_p, "bootstrap_diff": diff, "bootstrap_ci_low": lo,
                          "bootstrap_ci_high": hi, "bootstrap_p": boot_p})
    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df["wilcoxon_q"] = bh_fdr(pair_df["wilcoxon_p"])
        pair_df["bootstrap_q"] = bh_fdr(pair_df["bootstrap_p"])
    pair_df.to_csv(out_dir / "stats_paired.csv", index=False)

    with pd.option_context("display.width", 180, "display.max_columns", 20):
        print("\nPer configuration (test QWK over seeds):")
        print(cfg_df.round(4).to_string(index=False))
        if not pair_df.empty:
            print("\nPaired comparisons (A - B):")
            cols = ["config_a", "config_b", "n_seeds", "params_a", "params_b", "median_seed_diff",
                    "wilcoxon_q", "bootstrap_diff", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_q"]
            print(pair_df[cols].round(4).to_string(index=False))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
