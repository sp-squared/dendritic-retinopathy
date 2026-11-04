"""Step 3: extract frozen ImageNet-pretrained backbone features for each split.

Training heads on cached features makes the dANN-vs-dense comparison fast
enough to run many seeds and configurations on modest local hardware.

Example
-------
python scripts/03_extract_features.py --backbone resnet18 --input-size 224
python scripts/03_extract_features.py --backbone resnet50 --input-size 384

Writes data/features/<backbone>_<input-size>.npz containing, per split,
X_<split> (N x D float32), y_<split> (N,), ids_<split> (N,).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from dendritic_dr.data import FundusDataset
from dendritic_dr.utils import DATA_DIR, default_num_workers, get_device

BACKBONES = {
    # name: (constructor, weights enum)
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def build_backbone(name: str) -> nn.Module:
    ctor, weights = BACKBONES[name]
    net = ctor(weights=weights)
    # Replace the classifier with identity so the forward pass returns pooled features.
    if hasattr(net, "fc"):
        net.fc = nn.Identity()
    else:
        net.classifier = nn.Identity()
    return net.eval()


@torch.no_grad()
def extract(net, loader, device):
    feats, ys, ids = [], [], []
    for x, y, id_code in tqdm(loader, leave=False):
        feats.append(net(x.to(device)).float().cpu().numpy())
        ys.append(y.numpy())
        ids.extend(id_code)
    return np.concatenate(feats), np.concatenate(ys), np.array(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=list(BACKBONES), default="resnet18")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--image-dir", default=None, help="defaults to data/processed/aptos_512")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=default_num_workers())
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    image_dir = args.image_dir or DATA_DIR / "processed" / "aptos_512"
    splits = pd.read_csv(DATA_DIR / "processed" / "splits.csv")
    device = get_device(args.device)
    net = build_backbone(args.backbone).to(device)

    tf = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    out = {}
    for split in ["train", "val", "test"]:
        ds = FundusDataset(splits[splits["split"] == split], image_dir, tf)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        X, y, ids = extract(net, loader, device)
        out[f"X_{split}"], out[f"y_{split}"], out[f"ids_{split}"] = X, y, ids
        print(f"{split}: {X.shape}")

    dst = DATA_DIR / "features" / f"{args.backbone}_{args.input_size}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
