"""Step 2: crop/resize APTOS 2019 fundus images and create stratified splits.

Expects the Kaggle files unpacked at:
    data/raw/aptos/train.csv
    data/raw/aptos/train_images/<id_code>.png

Writes:
    data/processed/aptos_<size>/<id_code>.png
    data/processed/splits.csv   (id_code, diagnosis, split)

Example
-------
python scripts/02_prepare_aptos.py --size 512
"""
from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from dendritic_dr.data import make_splits, preprocess_image
from dendritic_dr.utils import DATA_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default=str(DATA_DIR / "raw" / "aptos"))
    p.add_argument("--size", type=int, default=512, help="output side length in pixels")
    p.add_argument("--seed", type=int, default=42, help="split seed (keep fixed for the whole project)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from pathlib import Path

    raw = Path(args.raw_dir)
    labels = pd.read_csv(raw / "train.csv")
    out_dir = DATA_DIR / "processed" / f"aptos_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(labels)} labelled images; grade counts:")
    print(labels["diagnosis"].value_counts().sort_index().to_string())

    for id_code in tqdm(labels["id_code"], desc="preprocessing"):
        dst = out_dir / f"{id_code}.png"
        if dst.exists() and not args.overwrite:
            continue
        preprocess_image(raw / "train_images" / f"{id_code}.png", args.size).save(dst)

    splits = make_splits(labels, args.seed, args.val_frac, args.test_frac)
    splits_path = DATA_DIR / "processed" / "splits.csv"
    splits.to_csv(splits_path, index=False)
    print(f"\nsplits written to {splits_path}")
    print(pd.crosstab(splits["split"], splits["diagnosis"]).to_string())


if __name__ == "__main__":
    main()
