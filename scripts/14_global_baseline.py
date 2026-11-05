"""Step 14: is the raw-pixel result driven by global image properties?

On raw 128 x 128 green-channel pixels (step 4d) the best model was a dense ANN with ONE
hidden unit, i.e. a single weighted sum over the whole image. This step asks whether that
unit mostly reads global, lesion-independent properties (brightness, contrast, sharpness;
the same properties that separated cameras in step 8) rather than disease.

1. Baselines: multinomial logistic regression on a handful of global statistics per image
   (no pixels, no CNN). Regularisation strength is chosen by validation QWK.
     brightness_contrast  grey-level mean and SD inside the fundus
     global               + sharpness and green-channel mean/SD
     global_acquisition   + original resolution and file size (camera proxies; needs raw files)
   Each is compared with the one-unit dense ANN (mean over its seeds) by a paired bootstrap
   over the same test images.
2. Correlation of the one-unit model's hidden activation with each property on the test
   set (Spearman; |rho| because the unit's sign is arbitrary), per seed.

If a few global statistics come close to the one-unit model, the raw-pixel result mostly
reflects image-wide (largely acquisition) differences, not local lesions.

Run after step 4d:
python scripts/14_global_baseline.py

Outputs (results/analysis/): image_properties.csv, global_baseline.csv, global_unit_corr.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dendritic_dr.analysis import bootstrap_qwk_ci, hidden_activations, load_checkpoint, paired_bootstrap_diff
from dendritic_dr.data import image_properties
from dendritic_dr.features import load_features, to_tensors
from dendritic_dr.metrics import quadratic_kappa
from dendritic_dr.utils import DATA_DIR, results_root

SETS = {
    "brightness_contrast": ["brightness", "contrast"],
    "global": ["brightness", "contrast", "sharpness", "green_mean", "green_std"],
    "global_acquisition": ["brightness", "contrast", "sharpness", "green_mean", "green_std",
                           "orig_width", "orig_height", "file_kb"],
}
LOG_COLS = {"orig_width", "orig_height", "file_kb", "sharpness"}


def properties_table(splits, proc_dir, raw_dir, cache):
    if cache.exists():
        df = pd.read_csv(cache)
        if set(splits["id_code"]) <= set(df["id_code"]):
            return df
    have_raw = raw_dir.exists()
    rows = []
    for i in tqdm(splits["id_code"], desc="image properties", leave=False):
        raw = raw_dir / f"{i}.png" if have_raw else None
        rows.append({"id_code": i, **image_properties(proc_dir / f"{i}.png", raw)})
    df = pd.DataFrame(rows).merge(splits[["id_code", "diagnosis", "split"]], on="id_code")
    df.to_csv(cache, index=False)
    return df


def design(df, cols):
    X = df[cols].to_numpy(dtype=float, copy=True)
    for j, c in enumerate(cols):
        if c in LOG_COLS:
            X[:, j] = np.log1p(X[:, j])
    return X


def fit_baseline(props, cols):
    tr, va, te = (props[props["split"] == s] for s in ["train", "val", "test"])
    scaler = StandardScaler().fit(design(tr, cols))
    Xtr, Xva, Xte = (scaler.transform(design(d, cols)) for d in (tr, va, te))
    best = None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, tr["diagnosis"])
        q = quadratic_kappa(va["diagnosis"], clf.predict(Xva))
        if best is None or q > best[0]:
            best = (q, C, clf)
    val_q, C, clf = best
    return val_q, C, pd.Series(clf.predict(Xte), index=te["id_code"].to_numpy())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pixels_green_128_vanilla_h1_ce", help="the one-unit raw-pixel model")
    p.add_argument("--proc-dir", default=str(DATA_DIR / "processed" / "aptos_512"))
    p.add_argument("--raw-dir", default=str(DATA_DIR / "raw" / "aptos" / "train_images"))
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    root = results_root(False)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(DATA_DIR / "processed" / "splits.csv")
    props = properties_table(splits, Path(args.proc_dir), Path(args.raw_dir), out_dir / "image_properties.csv")

    runs = pd.read_csv(root / "aptos_heads.csv")
    runs = runs[runs["config"] == args.config].sort_values("seed")
    if runs.empty:
        raise SystemExit(f"no runs of {args.config}; train step 4d first")
    preds = [pd.read_csv(root / "predictions" / f"{r}.csv").set_index("id_code") for r in runs["run_id"]]
    ids = preds[0].index
    y = preds[0]["true"].to_numpy()
    pa = np.stack([f.loc[ids, "pred"].to_numpy() for f in preds])
    ref_q = float(np.mean([quadratic_kappa(y, row) for row in pa]))

    # 1. global-statistics baselines vs. the one-unit model
    rows = []
    for k, (name, cols) in enumerate(SETS.items()):
        if any(c not in props or props[c].isna().all() for c in cols):
            print(f"{name}: skipped (raw images not found at {args.raw_dir})")
            continue
        val_q, C, pred = fit_baseline(props, cols)
        pb = pred.loc[ids].to_numpy()
        test_q = quadratic_kappa(y, pb)
        lo, hi = bootstrap_qwk_ci(y, pb, n_boot=args.n_boot)
        d, dlo, dhi, dp = paired_bootstrap_diff(y, pa, pb[None], n_boot=args.n_boot, seed=k)
        rows.append({"baseline": name, "features": " ".join(cols), "C": C, "val_qwk": val_q, "test_qwk": test_q,
                     "test_ci_low": lo, "test_ci_high": hi, "one_unit_qwk": ref_q,
                     "one_unit_minus_baseline": d, "diff_ci_low": dlo, "diff_ci_high": dhi, "diff_p": dp})
    base = pd.DataFrame(rows)
    base.to_csv(out_dir / "global_baseline.csv", index=False)

    # 2. what the one hidden unit tracks (test split)
    feats = load_features(runs["features"].iloc[0])
    test_ids = np.asarray(feats["ids_test"]).astype(str)
    tp = props.set_index("id_code").loc[test_ids]
    cols = [c for c in SETS["global_acquisition"]
            if c in tp and tp[c].notna().all() and tp[c].nunique() > 1] + ["diagnosis"]
    corr = []
    for r in runs.itertuples():
        model, ckpt = load_checkpoint(root / "models" / f"{r.run_id}.pt")
        x, _ = to_tensors(feats, "cpu", ckpt["feature_mean"], ckpt["feature_std"])["test"]
        a = hidden_activations(model, x)[:, 0]
        for c in cols:
            rho = stats.spearmanr(a, tp[c]).statistic if a.std() > 0 else np.nan
            corr.append({"seed": r.seed, "property": "grade" if c == "diagnosis" else c, "rho": rho})
    corr = pd.DataFrame(corr)
    corr.to_csv(out_dir / "global_unit_corr.csv", index=False)
    summ = (corr.assign(abs_rho=corr["rho"].abs()).groupby("property", sort=False)["abs_rho"]
            .agg(["mean", "min", "max"]).round(3))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"one-unit model {args.config}: mean test QWK {ref_q:.4f} over {len(runs)} seeds\n")
        cols_out = ["baseline", "val_qwk", "test_qwk", "test_ci_low", "test_ci_high",
                    "one_unit_minus_baseline", "diff_ci_low", "diff_ci_high", "diff_p"]
        print(base[cols_out].round(4).to_string(index=False))
        print("\n|Spearman rho| of the hidden unit with each property (test set, over seeds):")
        print(summ.to_string())
    print(f"\nwrote {out_dir / 'global_baseline.csv'}, global_unit_corr.csv, image_properties.csv")


if __name__ == "__main__":
    main()
