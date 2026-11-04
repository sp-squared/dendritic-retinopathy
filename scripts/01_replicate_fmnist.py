"""Step 1: replicate the dANN vs. vanilla ANN comparison on Fashion-MNIST.

Examples
--------
python scripts/01_replicate_fmnist.py --model dann-lrf --seeds 0 1 2
python scripts/01_replicate_fmnist.py --model dann-r   --seeds 0 1 2
python scripts/01_replicate_fmnist.py --model vanilla --hidden 256 --seeds 0 1 2
python scripts/01_replicate_fmnist.py --smoke     # offline check with random data

Each run appends one row to results/fmnist.csv (results/smoke/ with --smoke).
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from dendritic_dr.models import add_model_args, build_model, effective_params
from dendritic_dr.utils import DATA_DIR, append_result, get_device, results_root, set_seed

IMAGE_SHAPE = (1, 28, 28)


def load_fmnist(smoke: bool):
    if smoke:
        g = torch.Generator().manual_seed(0)
        xtr, ytr = torch.rand(2000, 1, 28, 28, generator=g), torch.randint(0, 10, (2000,), generator=g)
        xte, yte = torch.rand(500, 1, 28, 28, generator=g), torch.randint(0, 10, (500,), generator=g)
        return xtr, ytr, xte, yte
    from torchvision.datasets import FashionMNIST

    root = DATA_DIR / "fmnist"
    tr = FashionMNIST(root, train=True, download=True)
    te = FashionMNIST(root, train=False, download=True)
    # Whole dataset fits in memory: skip DataLoader workers entirely.
    to_x = lambda d: (d.data.float() / 255.0).unsqueeze(1)
    return to_x(tr), tr.targets, to_x(te), te.targets


@torch.no_grad()
def evaluate(model, x, y, device, bs=2048):
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(x), bs):
        xb, yb = x[i : i + bs].to(device), y[i : i + bs].to(device)
        out = model(xb)
        loss += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return correct / len(x), loss / len(x)


def run(args, seed, data, device):
    set_seed(seed)
    args.seed = seed
    xtr, ytr, xte, yte = data
    model = build_model(args, in_features=784, n_out=10, image_shape=IMAGE_SHAPE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = effective_params(model)
    print(f"[seed {seed}] {args.model}: {n_params:,} effective params")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr_acc, tr_loss = evaluate(model, xtr, ytr, device)
        te_acc, te_loss = evaluate(model, xte, yte, device)
        print(f"  epoch {epoch:3d}  train acc {tr_acc:.4f} loss {tr_loss:.4f} | test acc {te_acc:.4f} loss {te_loss:.4f}")

    append_result(
        results_root(args.smoke) / "fmnist.csv",
        {
            "model": args.model,
            "seed": seed,
            "somas": args.somas if args.model != "vanilla" else "",
            "dendrites": args.dendrites if args.model != "vanilla" else "",
            "synapses": args.synapses if args.model == "dann-r" else "",
            "rf_size": args.rf_size if args.model == "dann-lrf" else "",
            "hidden": args.hidden if args.model == "vanilla" else "",
            "effective_params": n_params,
            "epochs": args.epochs,
            "train_acc": tr_acc,
            "test_acc": te_acc,
            "train_loss": tr_loss,
            "test_loss": te_loss,
            "seconds": round(time.time() - t0, 1),
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true", help="tiny random data, 1 epoch, no download")
    args = p.parse_args()
    if args.smoke:
        args.epochs = 1

    device = get_device(args.device)
    print(f"device: {device}")
    data = load_fmnist(args.smoke)
    for seed in args.seeds:
        run(args, seed, data, device)


if __name__ == "__main__":  # required on Windows for multiprocessing safety
    main()
