"""Step 4: train a dendritic or dense head on cached backbone features.

Model selection uses validation QWK (early stopping); the test split is only
evaluated once, at the best validation epoch.

Examples
--------
python scripts/04_train_heads.py --features resnet18_224 --model dann-r  --somas 128 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/04_train_heads.py --features resnet18_224 --model vanilla --hidden 64 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/04_train_heads.py --features resnet18_224 --model dann-r --loss ordinal
python scripts/04_train_heads.py --smoke   # offline check with synthetic features

Spatial feature maps (from 03 --spatial): dendrites can sample local patches of the grid
python scripts/04_train_heads.py --features resnet18_224_spatial --model dann-lrf --somas 128 --rf-size 3 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/04_train_heads.py --features resnet18_224_spatial --model dann-r   --somas 128 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/04_train_heads.py --smoke --features smoke_spatial --model dann-lrf --rf-size 3

Each run writes
  results/aptos_heads.csv                 one summary row (confusion matrix as JSON)
  results/models/<run_id>.pt              best checkpoint + config + standardisation stats
  results/predictions/<run_id>.csv        per-image test predictions (id_code, true, pred)
(under results/smoke/ when --smoke is used).
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dendritic_dr.features import load_features, standardization, to_tensors
from dendritic_dr.metrics import head_outputs, ordinal_loss, predict, summarize
from dendritic_dr.models import add_model_args, build_model, compact_state_dict, effective_params
from dendritic_dr.utils import append_result, get_device, results_root, set_seed

# Args stored in each checkpoint so the model can be rebuilt by later scripts.
MODEL_ARG_KEYS = ["model", "somas", "dendrites", "synapses", "rf_size", "hidden", "dropout",
                  "mask_seed", "seed", "loss", "features", "class_weights", "image_shape"]


def config_tag(args) -> str:
    """Short, filename-safe description of a configuration (without the seed)."""
    if args.model == "vanilla":
        arch = "h" + "-".join(map(str, args.hidden))
    else:
        arch = f"s{'-'.join(map(str, args.somas))}_d{args.dendrites}_y{args.synapses}"
        if args.model == "dann-lrf":
            arch += f"_rf{args.rf_size}"
    tag = f"{args.features}_{args.model}_{arch}_{args.loss}"
    if args.dropout > 0:
        tag += f"_do{args.dropout:g}"
    if args.class_weights:
        tag += "_cw"
    return tag


def make_loss(kind: str, class_weights):
    if kind == "ordinal":
        return ordinal_loss
    return lambda logits, y: F.cross_entropy(logits, y, weight=class_weights)


@torch.no_grad()
def eval_split(model, x, y, loss_kind):
    model.eval()
    preds = predict(model(x), loss_kind)
    return summarize(y.cpu().numpy(), preds), preds


def run(args, seed, raw, data, mu, sd, device, out_root):
    set_seed(seed)
    args.seed = seed
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]
    n_out = head_outputs(args.loss)
    model = build_model(args, in_features=xtr.shape[1], n_out=n_out, image_shape=args.image_shape).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cw = None
    if args.class_weights and args.loss == "ce":
        counts = torch.bincount(ytr, minlength=5).float()
        cw = (counts.sum() / (5 * counts.clamp(min=1))).to(device)
    loss_fn = make_loss(args.loss, cw)

    n_params = effective_params(model)
    best_qwk, best_state, best_epoch, bad = -np.inf, None, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(xtr), args.batch_size):
            idx = perm[i : i + args.batch_size]
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_qwk = eval_split(model, xva, yva, args.loss)[0]["qwk"]
        if val_qwk > best_qwk:
            best_qwk, best_state, best_epoch, bad = val_qwk, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    test, test_preds = eval_split(model, xte, yte, args.loss)
    tag = config_tag(args)
    run_id = f"{tag}_seed{seed}"
    print(
        f"[seed {seed}] {args.model:8s} params {n_params:>9,}  best epoch {best_epoch:3d}  "
        f"val QWK {best_qwk:.4f}  test QWK {test['qwk']:.4f}  test acc {test['acc']:.4f}"
    )

    # Checkpoint: everything needed to rebuild the model and its inputs exactly.
    (out_root / "models").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": compact_state_dict(model),  # masks regenerated from the seed on load
            "compact": True,
            "model_args": {k: getattr(args, k) for k in MODEL_ARG_KEYS},
            "in_features": int(xtr.shape[1]),
            "n_out": n_out,
            "feature_mean": mu,
            "feature_std": sd,
            "run_id": run_id,
            "config": tag,
        },
        out_root / "models" / f"{run_id}.pt",
    )
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"id_code": raw["ids_test"], "true": yte.cpu().numpy(), "pred": test_preds}
    ).to_csv(out_root / "predictions" / f"{run_id}.csv", index=False)

    append_result(
        out_root / "aptos_heads.csv",
        {
            "run_id": run_id,
            "config": tag,
            "features": args.features,
            "model": args.model,
            "loss": args.loss,
            "seed": seed,
            "somas": args.somas if args.model != "vanilla" else "",
            "dendrites": args.dendrites if args.model != "vanilla" else "",
            "synapses": args.synapses if args.model != "vanilla" else "",
            "rf_size": args.rf_size if args.model == "dann-lrf" else "",
            "hidden": args.hidden if args.model == "vanilla" else "",
            "dropout": args.dropout,
            "class_weights": args.class_weights,
            "effective_params": n_params,
            "best_epoch": best_epoch,
            "val_qwk": best_qwk,
            "test_qwk": test["qwk"],
            "test_acc": test["acc"],
            "test_confusion": test["confusion"],
            "seconds": round(time.time() - t0, 1),
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--features", default="resnet18_224", help="name of data/features/<name>.npz")
    p.add_argument("--loss", choices=["ce", "ordinal"], default="ce")
    p.add_argument("--class-weights", action="store_true", help="inverse-frequency weights (ce only)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 30)
        args.features = "smoke_spatial" if "spatial" in args.features else "smoke"

    device = get_device(args.device)
    raw = load_features(args.features, args.smoke)
    args.image_shape = raw["image_shape"]
    if args.model == "dann-lrf" and args.image_shape is None:
        raise SystemExit("dann-lrf needs spatial features (C x H x W); extract them with 03 --spatial")
    if args.image_shape is not None:
        print(f"spatial features: {args.image_shape[0]} channels x {args.image_shape[1]} x {args.image_shape[2]} grid")
    mu, sd = standardization(raw)
    data = to_tensors(raw, device, mu, sd)
    out_root = results_root(args.smoke)
    for seed in args.seeds:
        run(args, seed, raw, data, mu, sd, device, out_root)


if __name__ == "__main__":
    main()
