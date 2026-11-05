"""Step 18: train on lesion masks: do dendrites become lesion detectors when the task requires it?

Step 17 found that CNNs trained on image-level grades learn no lesion detectors in their
first layer. Here the same four first-layer variants are trained from scratch on IDRiD's
pixel-level lesion masks (microaneurysms MA, haemorrhages HE, hard exudates EX, soft
exudates SE), predicting for every cell of the first layer's stride-4 grid which lesions
it contains (multi-label, binary cross-entropy weighted for rare positives).

  --depth 0   linear readout: a 1 x 1 layer directly on the 16 first-layer units, so the
              somas / channels themselves must become lesion detectors (the direct test)
  --depth 3   three dilated 3 x 3 conv blocks (32 channels) between the first layer and the
              readout: realistic, but later layers can compensate for a weak first layer

Data: IDRiD's official split (54 training images, 27 test images); 10 training images are
held out (fixed) for early stopping on validation mean average precision (AP). Training
uses random flips and 90-degree rotations. Evaluation on the 27 test images, on cells inside
the fundus and outside the optic disc:
  * AP and AUC per lesion type (AP's chance level = the lesion's share of cells)
  * per first-layer unit and per dendrite: AUC for lesion vs. lesion-free cells, as in
    step 17: the best unit and the share of units with AUC >= 0.75

Prediction (stated before running): with lesion supervision, clustered dendrites (each
confined to a 5 x 5 px patch) become better small-lesion detectors than scattered ones:
higher test AP for MA and HE at depth 0, and more dendrites with AUC >= 0.75.

python scripts/18_train_lesions.py --stem dend-lrf --depth 0 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/18_train_lesions.py --smoke --stem dend-lrf --depth 0 --seeds 0 1

The first run caches the preprocessed IDRiD set in data/features/idrid_512.npz.
Each run appends a row to results/lesion_runs.csv (checkpoints: results/models/, git-ignored).
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from dendritic_dr.cnn import STEMS, build_lesion_net
from dendritic_dr.idrid import GRID_STRIDE, LESIONS, cell_grids, load_idrid, smoke_idrid, stem_maps, unit_aucs
from dendritic_dr.utils import DATA_DIR, append_result, get_device, results_root, set_seed


def smoke_data():
    items = smoke_idrid(n=16)
    ids = np.array(list(items))
    green = np.stack([items[k][0] for k in ids])
    masks = np.stack([np.stack([items[k][1][t] for t in LESIONS + ["OD"]]) for k in ids])
    split = np.array(["train"] * 12 + ["test"] * 4)
    return {"ids": ids, "split": split, "green": green, "masks": masks}


def torch_targets(green_u8, masks, shape):
    """Grid targets for a batch on the GPU: lesions (B, 4, H, W) and valid cells (B, 1, H, W)."""
    pool = lambda m: F.max_pool2d(m.float(), GRID_STRIDE, GRID_STRIDE, padding=GRID_STRIDE // 2)[..., : shape[0], : shape[1]]
    fundus = F.avg_pool2d((green_u8 > 7).float(), GRID_STRIDE, GRID_STRIDE, padding=GRID_STRIDE // 2,
                          count_include_pad=False)[..., : shape[0], : shape[1]] > 0.9
    od = F.max_pool2d(pool(masks[:, 4:5]), 5, 1, padding=2) > 0
    valid = fundus & ~od
    return pool(masks[:, :4]) * valid, valid.float()


def augment(x, m, gen):
    """Same random flips / 90-degree rotation for the image batch and its pixel masks."""
    if torch.rand(1, generator=gen) < 0.5:
        x, m = x.flip(-1), m.flip(-1)
    if torch.rand(1, generator=gen) < 0.5:
        x, m = x.flip(-2), m.flip(-2)
    k = int(torch.randint(0, 4, (1,), generator=gen))
    return torch.rot90(x, k, (-2, -1)), torch.rot90(m, k, (-2, -1))


@torch.no_grad()
def predict(model, green, mean, sd, device, amp, bs=4):
    model.eval()
    out = []
    for i in range(0, len(green), bs):
        x = (green[i:i + bs].to(device).float() - mean) / sd
        with torch.autocast(device.type, enabled=amp):
            out.append(torch.sigmoid(model(x).float()).cpu())
    return torch.cat(out).numpy()


def lesion_metrics(prob, grids):
    """AP and AUC per lesion type over valid cells of all images."""
    out = {}
    for j, t in enumerate(LESIONS):
        y = np.concatenate([g[0][j][g[1]] for g in grids])
        p = np.concatenate([pr[j][: g[1].shape[0], : g[1].shape[1]][g[1]] for pr, g in zip(prob, grids)])
        if y.sum() == 0 or y.all():
            out[t] = (np.nan, np.nan, float(y.mean()))
        else:
            out[t] = (average_precision_score(y, p), roc_auc_score(y, p), float(y.mean()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", choices=STEMS, required=True)
    p.add_argument("--depth", type=int, default=0, help="conv blocks between first layer and readout (0 = linear)")
    p.add_argument("--kernel", type=int, default=12)
    p.add_argument("--cluster", type=int, default=5)
    p.add_argument("--somas", type=int, default=16)
    p.add_argument("--dendrites", type=int, default=8)
    p.add_argument("--synapses", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-pos-weight", type=float, default=100.0, help="cap on the rare-positive weight per lesion")
    p.add_argument("--n-val", type=int, default=10, help="training images held out for early stopping")
    p.add_argument("--idrid-dir", default=str(DATA_DIR / "raw" / "idrid"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    device = get_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    if args.smoke:
        data, args.size, args.epochs, args.n_val = smoke_data(), 64, min(args.epochs, 15), 3
    else:
        data = load_idrid(args.idrid_dir, DATA_DIR / "features" / f"idrid_{args.size}.npz", args.size)
    rng = np.random.default_rng(42)  # fixed validation images for every run
    tr_all = np.flatnonzero(data["split"] == "train")
    val_idx = np.sort(rng.choice(tr_all, args.n_val, replace=False))
    tr_idx = np.setdiff1d(tr_all, val_idx)
    te_idx = np.flatnonzero(data["split"] == "test")
    print(f"IDRiD: train {len(tr_idx)}, val {len(val_idx)}, test {len(te_idx)} images; device {device}"
          f"{' (mixed precision)' if amp else ''}")

    green = torch.from_numpy(data["green"])[:, None]  # (N, 1, S, S) uint8
    masks = torch.from_numpy(data["masks"])           # (N, 5, S, S) bool
    g_tr = data["green"][tr_idx].astype(np.float64)
    mean, sd = float(g_tr.mean()), float(g_tr.std()) + 1e-6  # training images only
    store = device if device.type == "cuda" else torch.device("cpu")
    green_d, masks_d = green.to(store), masks.to(store)
    out_root = results_root(args.smoke)

    for seed in args.seeds:
        set_seed(seed)
        model = build_lesion_net(args.stem, depth=args.depth, somas=args.somas, dendrites=args.dendrites,
                                 synapses=args.synapses, kernel=args.kernel, cluster=args.cluster, seed=seed).to(device)
        with torch.no_grad():
            shape = tuple(model(torch.zeros(1, 1, args.size, args.size, device=device)).shape[-2:])
        grids = {i: cell_grids(data["green"][i], data["masks"][i], shape) for i in np.r_[val_idx, te_idx]}

        # rare-positive weights from the training images
        les, val = torch_targets(green_d[tr_idx], masks_d[tr_idx], shape)
        pos = les.sum((0, 2, 3))
        neg = val.sum() - pos
        pos_weight = (neg / pos.clamp(min=1)).clamp(max=args.max_pos_weight).to(device)[None, :, None, None]

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        gen = torch.Generator().manual_seed(seed)
        best, best_state, best_epoch, bad = -np.inf, None, 0, 0
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = tr_idx[torch.randperm(len(tr_idx), generator=gen).numpy()]
            for i in range(0, len(order), args.batch_size):
                b = order[i:i + args.batch_size]
                x, m = augment(green_d[b].to(device), masks_d[b].to(device), gen)
                y, v = torch_targets(x, m, shape)
                with torch.autocast(device.type, enabled=amp):
                    logits = model((x.float() - mean) / sd)
                loss = (F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pos_weight,
                                                           reduction="none") * v).sum() / (v.sum() * 4).clamp(min=1)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            vm = lesion_metrics(predict(model, green_d[val_idx], mean, sd, device, amp), [grids[i] for i in val_idx])
            val_map = float(np.nanmean([vm[t][0] for t in LESIONS]))
            if val_map > best:
                best, best_state, best_epoch, bad = val_map, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
        model.load_state_dict(best_state)

        # test: lesion AP / AUC, and lesion selectivity of first-layer units and dendrites
        test_grids = [grids[i] for i in te_idx]
        tm = lesion_metrics(predict(model, green_d[te_idx], mean, sd, device, amp), test_grids)
        model.eval()
        units, dends = [], []
        for i in te_idx:
            x = (green_d[i:i + 1].to(device).float() - mean) / sd
            u, d = stem_maps(model, x)
            units.append(u)
            dends.append(d)
        urng = np.random.default_rng(0)
        row = {"stem": args.stem, "kernel": args.kernel, "depth": args.depth, "seed": seed}
        tag = f"lesion{args.size}_{args.stem}_k{args.kernel}_depth{args.depth}"
        if args.stem.startswith("dend"):
            tag += f"_s{args.somas}_d{args.dendrites}_y{args.synapses}" + (f"_c{args.cluster}" if args.stem == "dend-lrf" else "")
        row = {"run_id": f"{tag}_seed{seed}", "config": tag, **row,
               "stem_params": model.stem_params(), "total_params": model.effective_params(),
               "best_epoch": best_epoch, "val_map": best,
               "test_map": float(np.nanmean([tm[t][0] for t in LESIONS]))}
        for j, t in enumerate(LESIONS):
            ap, auc, prev = tm[t]
            row |= {f"ap_{t}": ap, f"auc_{t}": auc, f"prev_{t}": prev}
            pos_g = [g[0][j] for g in test_grids]
            neg_g = [g[2] for g in test_grids]
            for level, acts in [("unit", units), ("dend", dends)]:
                if acts[0] is None:
                    continue
                a = unit_aucs(acts, pos_g, neg_g, urng)
                row |= {f"{level}_best_{t}": float(np.nanmax(a)), f"{level}_frac75_{t}": float(np.nanmean(a >= 0.75))}
        print(f"[seed {seed}] {args.stem:8s} k{args.kernel:<2d} depth {args.depth}  best epoch {best_epoch:3d}  "
              f"val mAP {best:.3f}  test mAP {row['test_map']:.3f}  AP " +
              " ".join(f"{t} {row[f'ap_{t}']:.3f}" for t in LESIONS) +
              f"  best unit AUC MA {row['unit_best_MA']:.2f}  ({time.time() - t0:.0f} s)")
        (out_root / "models").mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "args": vars(args) | {"seed": seed}, "mean": mean, "sd": sd},
                   out_root / "models" / f"{row['run_id']}.pt")
        append_result(out_root / "lesion_runs.csv", row)


if __name__ == "__main__":
    main()
