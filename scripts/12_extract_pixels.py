"""Step 12: raw-pixel inputs for the dendritic heads (no CNN backbone).

The ResNet experiments (steps 4-4c) found no locality advantage for dendrites on CNN
features at any depth. The report's interpretation is that the Fashion-MNIST advantage
needs input-level locality, as in raw pixels. This step makes that a direct test: the
same heads are trained on downsampled fundus pixels instead of features.

Each preprocessed fundus image (step 2) is downsampled to --size x --size and one channel
is kept (default green, where vessels, microaneurysms and hemorrhages have the most
contrast). 128 x 128 gives 16,384 inputs, comparable to the ~25k of the sweep. Pixels are
stored as uint8; training standardises each pixel with training-split statistics, as for
features.

Example
-------
python scripts/12_extract_pixels.py                     # data/features/pixels_green_128.npz
python scripts/12_extract_pixels.py --size 96 --channel gray

Writes data/features/pixels_<channel>_<size>.npz with, per split, X_<split> (N x C x size x size,
uint8), y_<split> and ids_<split>, so scripts 04 and 13 use it like any feature file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from dendritic_dr.data import CHANNELS, fundus_pixels
from dendritic_dr.utils import DATA_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--channel", choices=["gray", *CHANNELS], default="green")
    p.add_argument("--image-dir", default=None, help="defaults to data/processed/aptos_512")
    args = p.parse_args()

    image_dir = Path(args.image_dir or DATA_DIR / "processed" / "aptos_512")
    splits = pd.read_csv(DATA_DIR / "processed" / "splits.csv")
    out = {}
    for split in ["train", "val", "test"]:
        df = splits[splits["split"] == split].reset_index(drop=True)
        X = np.stack([fundus_pixels(Image.open(image_dir / f"{i}.png"), args.size, args.channel)
                      for i in tqdm(df["id_code"], desc=split, leave=False)])
        out[f"X_{split}"], out[f"y_{split}"] = X, df["diagnosis"].to_numpy()
        out[f"ids_{split}"] = df["id_code"].to_numpy().astype(str)
        print(f"{split}: {X.shape} {X.dtype}")

    dst = DATA_DIR / "features" / f"pixels_{args.channel}_{args.size}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
