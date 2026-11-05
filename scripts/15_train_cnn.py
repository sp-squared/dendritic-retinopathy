"""Step 15: small CNN trained end to end on 512 px fundus images, dendritic vs. dense first layer.

All earlier tests put the dendritic layer where a microaneurysm is at most one input wide
(frozen CNN features, or pixels downsampled to 128 px). Here the first trainable layer
sees 512 px green-channel images, where small lesions span several pixels, and it is
applied as a convolution so lesions are found wherever they are (see dendritic_dr/cnn.py).

Prediction, stated before running: if dendrites help when lesions are local at the input
resolution, dend-lrf beats dend-r (same parameters, same 12 x 12 field, only clustering
differs) and beats the parameter-matched dense 12 x 12 convolution.

Input: data/features/pixels_green_512.npz from
    python scripts/12_extract_pixels.py --size 512
(uint8, about 1 GB in memory; kept on the GPU when there is one).

Examples
--------
python scripts/15_train_cnn.py --stem dend-lrf --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/15_train_cnn.py --stem dend-r   --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/15_train_cnn.py --stem conv --kernel 12 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/15_train_cnn.py --stem conv --kernel 5  --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/15_train_cnn.py --smoke --stem dend-lrf --seeds 0 1      # synthetic, a few minutes on CPU

Each run appends a row to results/cnn_runs.csv and writes per-image test predictions to
results/predictions/<run_id>.csv (checkpoints go to results/models/, git-ignored).
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dendritic_dr.cnn import STEMS, build_cnn
from dendritic_dr.metrics import summarize
from dendritic_dr.utils import DATA_DIR, append_result, get_device, results_root, set_seed

SPLITS = ("train", "val", "test")


def smoke_images(seed: int = 0, size: int = 64) -> dict:
    """Synthetic fundus-like images: noise plus small bright 'lesions' at random positions,
    more of them at higher grades (local signal that can appear anywhere)."""
    rng = np.random.default_rng(seed)
    out = {}
    for split, n in zip(SPLITS, (400, 100, 100)):
        y = rng.choice(5, size=n, p=[0.45, 0.12, 0.25, 0.08, 0.10])
        X = rng.normal(100, 20, size=(n, 1, size, size))
        for i, g in enumerate(y):
            for _ in range(3 * g):
                cy, cx = rng.integers(2, size - 3, size=2)
                X[i, 0, cy - 1:cy + 2, cx - 1:cx + 2] += 60
        out[f"X_{split}"] = np.clip(X, 0, 255).astype(np.uint8)
        out[f"y_{split}"], out[f"ids_{split}"] = y, np.array([f"smoke_{split}_{i:04d}" for i in range(n)])
    return out


def pixel_stats(X: np.ndarray, chunk: int = 64) -> tuple[float, float]:
    """Mean and SD of all pixels, in chunks (a float64 copy of 512 px images would need ~8 GB)."""
    total, sq, n = 0.0, 0.0, 0
    for i in range(0, len(X), chunk):
        c = X[i:i + chunk].astype(np.float64)
        total, sq, n = total + c.sum(), sq + (c * c).sum(), n + c.size
    mean = total / n
    return float(mean), float(np.sqrt(max(sq / n - mean * mean, 0.0))) + 1e-6


def config_tag(args) -> str:
    tag = f"cnn{args.size}_{args.channel}_{args.stem}_k{args.kernel}"
    if args.stem.startswith("dend"):
        tag += f"_s{args.somas}_d{args.dendrites}_y{args.synapses}"
        if args.stem == "dend-lrf":
            tag += f"_c{args.cluster}"
    return tag


def batches(n, batch_size, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, batch_size):
        yield torch.from_numpy(order[i:i + batch_size])


def prepare(xb, mean, sd, device, augment=False):
    x = (xb.to(device, non_blocking=True).float() - mean) / sd
    if augment:  # random horizontal / vertical flips per image (fundus orientation is arbitrary)
        n = x.shape[0]
        fh = torch.rand(n, device=x.device) < 0.5
        fv = torch.rand(n, device=x.device) < 0.5
        x = torch.where(fh[:, None, None, None], x.flip(-1), x)
        x = torch.where(fv[:, None, None, None], x.flip(-2), x)
    return x


@torch.no_grad()
def predict(model, X, mean, sd, device, batch_size, amp):
    model.eval()
    out = []
    for idx in batches(len(X), batch_size):
        with torch.autocast(device.type, enabled=amp):
            out.append(model(prepare(X[idx], mean, sd, device)).argmax(1).cpu())
    return torch.cat(out).numpy()


def run(args, seed, data, device, out_root):
    set_seed(seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (data[s] for s in SPLITS)
    mean, sd = data["mean"], data["sd"]
    model = build_cnn(args.stem, in_ch=Xtr.shape[1], somas=args.somas, dendrites=args.dendrites,
                      synapses=args.synapses, kernel=args.kernel, cluster=args.cluster, stride=args.stride,
                      seed=seed, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
    rng = np.random.default_rng(seed)
    ytr_d = ytr.to(device)
    best_q, best_state, best_epoch, bad = -np.inf, None, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for idx in batches(len(Xtr), args.batch_size, rng):
            x = prepare(Xtr[idx], mean, sd, device, augment=True)
            with torch.autocast(device.type, enabled=args.amp):
                loss = F.cross_entropy(model(x), ytr_d[idx.to(device)])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        val_q = summarize(yva.numpy(), predict(model, Xva, mean, sd, device, args.batch_size, args.amp))["qwk"]
        if val_q > best_q:
            best_q, best_state, best_epoch, bad = val_q, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    pred = predict(model, Xte, mean, sd, device, args.batch_size, args.amp)
    test = summarize(yte.numpy(), pred)
    tag = config_tag(args)
    run_id = f"{tag}_seed{seed}"
    print(f"[seed {seed}] {args.stem:8s} k{args.kernel:<2d} stem params {model.stem_params():>6,}  "
          f"total {model.effective_params():>8,}  best epoch {best_epoch:3d}  val QWK {best_q:.4f}  "
          f"test QWK {test['qwk']:.4f}  ({time.time() - t0:.0f} s)")

    (out_root / "models").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args) | {"seed": seed}, "run_id": run_id},
               out_root / "models" / f"{run_id}.pt")
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id_code": data["ids_test"], "true": yte.numpy(), "pred": pred}).to_csv(
        out_root / "predictions" / f"{run_id}.csv", index=False)
    append_result(out_root / "cnn_runs.csv", {
        "run_id": run_id, "config": tag, "stem": args.stem, "kernel": args.kernel,
        "cluster": args.cluster if args.stem == "dend-lrf" else "",
        "somas": args.somas, "dendrites": args.dendrites if args.stem.startswith("dend") else "",
        "synapses": args.synapses if args.stem.startswith("dend") else "", "stride": args.stride,
        "size": args.size, "channel": args.channel, "seed": seed,
        "stem_params": model.stem_params(), "effective_params": model.effective_params(),
        "best_epoch": best_epoch, "val_qwk": best_q, "test_qwk": test["qwk"], "test_acc": test["acc"],
        "test_confusion": test["confusion"], "seconds": round(time.time() - t0, 1)})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", choices=STEMS, required=True)
    p.add_argument("--kernel", type=int, default=12, help="first-layer receptive field (k x k)")
    p.add_argument("--cluster", type=int, default=5, help="dend-lrf: c x c sub-window per dendrite")
    p.add_argument("--somas", type=int, default=16, help="first-layer output channels")
    p.add_argument("--dendrites", type=int, default=8, help="dendrites per soma")
    p.add_argument("--synapses", type=int, default=16, help="inputs per dendrite")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--channel", default="green")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    device = get_device(args.device)
    args.amp = device.type == "cuda" and not args.no_amp
    if args.smoke:
        args.size, args.channel, args.epochs = 64, "smoke", min(args.epochs, 8)
        raw = smoke_images()
    else:
        path = DATA_DIR / "features" / f"pixels_{args.channel}_{args.size}.npz"
        if not path.exists():
            raise SystemExit(f"{path} not found; run: python scripts/12_extract_pixels.py --size {args.size} "
                             f"--channel {args.channel}")
        raw = dict(np.load(path, allow_pickle=True))
    store = device if device.type == "cuda" else torch.device("cpu")  # uint8 images: ~1 GB at 512 px
    data = {s: (torch.from_numpy(raw[f"X_{s}"]).to(store), torch.from_numpy(raw[f"y_{s}"]).long()) for s in SPLITS}
    xtr = raw["X_train"]
    data["mean"], data["sd"] = pixel_stats(xtr)  # training-split statistics only
    data["ids_test"] = np.asarray(raw["ids_test"]).astype(str)
    print(f"images {tuple(xtr.shape[1:])}, train/val/test {[len(raw[f'y_{s}']) for s in SPLITS]}, device {device}"
          f"{' (mixed precision)' if args.amp else ''}")
    out_root = results_root(args.smoke)
    for seed in args.seeds:
        run(args, seed, data, device, out_root)


if __name__ == "__main__":
    main()
