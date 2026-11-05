"""Step 3: extract frozen ImageNet-pretrained backbone features for each split.

Training heads on cached features makes the dANN-vs-dense comparison fast
enough to run many seeds and configurations on modest local hardware.

Example
-------
python scripts/03_extract_features.py --backbone resnet18 --input-size 224
python scripts/03_extract_features.py --backbone resnet18 --input-size 224 --spatial   # 512 x 7 x 7 maps
python scripts/03_extract_features.py --backbone resnet18 --input-size 512 --spatial --grid 8
python scripts/03_extract_features.py --backbone resnet18 --input-size 224 --spatial --layer 2 --grid 14

Writes data/features/<backbone>_<input-size>[_spatial[_l<layer>][_g<grid>]].npz containing, per split,
X_<split> (N x D float32, or N x C x H x W float16 with --spatial), y_<split>, ids_<split>.

--spatial keeps the last convolutional feature map instead of global-average-pooling it,
so each grid cell describes one region of the fundus (7 x 7 at 224 px, 16 x 16 at 512 px).
--grid N average-pools that map to N x N to limit file size (16 x 16 x 512 x 3,662 images
in float16 is about 1 GB; 8 x 8 is about 240 MB).
--layer K (ResNet only) stops after residual stage K instead of the last one. Earlier
stages have smaller receptive fields, i.e. more local map cells. At 224 px the raw maps are
layer1 64x56x56, layer2 128x28x28, layer3 256x14x14, layer4 512x7x7. For the receptive-field
sweep, grids are chosen so every stage has ~25k inputs (layer1 --grid 20, layer2 --grid 14,
layer3 --grid 10, layer4 no grid), so only locality differs between stages.
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


def build_backbone(name: str, spatial: bool = False, grid: int | None = None, layer: int = 4) -> nn.Module:
    ctor, weights = BACKBONES[name]
    net = ctor(weights=weights)
    if spatial:
        # everything up to (not including) global pooling: output (N, C, H, W)
        if hasattr(net, "fc"):  # ResNet: stem + residual stages 1..layer
            stages = [net.layer1, net.layer2, net.layer3, net.layer4][:layer]
            body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, *stages)
        else:
            if layer != 4:
                raise SystemExit("--layer is only supported for ResNet backbones")
            body = net.features
        return (nn.Sequential(body, nn.AdaptiveAvgPool2d(grid)) if grid else body).eval()
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
    p.add_argument("--spatial", action="store_true", help="keep the spatial feature map (no global pooling)")
    p.add_argument("--grid", type=int, default=None, help="with --spatial: pool the map to grid x grid")
    p.add_argument("--layer", type=int, choices=[1, 2, 3, 4], default=4,
                   help="with --spatial (ResNet): last residual stage to keep (earlier = more local cells)")
    args = p.parse_args()

    image_dir = args.image_dir or DATA_DIR / "processed" / "aptos_512"
    splits = pd.read_csv(DATA_DIR / "processed" / "splits.csv")
    device = get_device(args.device)
    if args.layer != 4 and not args.spatial:
        raise SystemExit("--layer needs --spatial")
    net = build_backbone(args.backbone, args.spatial, args.grid, args.layer).to(device)

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
        if args.spatial:
            X = X.astype(np.float16)  # halves file size; heads convert back to float32
        out[f"X_{split}"], out[f"y_{split}"], out[f"ids_{split}"] = X, y, ids
        print(f"{split}: {X.shape}")

    suffix = ""
    if args.spatial:
        suffix = "_spatial" + (f"_l{args.layer}" if args.layer != 4 else "") + (f"_g{args.grid}" if args.grid else "")
    dst = DATA_DIR / "features" / f"{args.backbone}_{args.input_size}{suffix}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
