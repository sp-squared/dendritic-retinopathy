"""Step 20: lesion detection at near-native resolution on IDRiD + DDR, Tversky loss, FROC.

Step 18 found that, trained on lesion masks at 512 px, clustered dendrites find more
microaneurysms (MA) than scattered dendrites or a dense 12 x 12 layer, and a dense 5 x 5
layer does as well. Microaneurysm AP stayed low (~0.2), and at 512 px a microaneurysm is
only one to three pixels wide, with 44 training images. This step removes those limits:

* resolution: the fundus is resized to 2048 px (``--size``), so a microaneurysm spans
  roughly 4-12 px; training uses random 384 px patches, half of them centred near a lesion
* data: IDRiD's 44 training images plus DDR's 383 (``--train-on idrid`` for the ablation)
* loss: Tversky with alpha = 0.3 (false positives) < beta = 0.7 (misses), favouring recall;
  ``--loss bce`` gives step 18's weighted cross-entropy for comparison
* scoring: FROC per lesion type (mean sensitivity at 1/8 ... 8 false positives per image,
  and sensitivity at 1 per image), plus cell-level AP / AUC, on IDRiD's 27 test images
  (the same images as step 18) and on DDR's 225 test images

Model selection: early stopping on the mean FROC score over the four lesion types on held-out
images (IDRiD: step 18's 10 images; DDR: 30 fixed images of its validation split), with a
cosine learning-rate schedule so runs settle before the epoch limit.

Prediction (stated before running): at this resolution the clustered dendrites' microaneurysm
advantage over scattered dendrites from step 18 persists (higher MA FROC on both test sets),
and clustered dendrites match the dense 5 x 5 layer.

python scripts/20_train_fullres.py --stem dend-lrf --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/20_train_fullres.py --stem conv --kernel 5 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/20_train_fullres.py --smoke --stem dend-lrf --seeds 0 1

The first run preprocesses all images into data/features/lesions_2048/ (once, ~15 min).
Each run appends a row to results/fullres_runs.csv (checkpoints: results/models/, git-ignored).
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from dendritic_dr.cnn import STEMS, build_lesion_net
from dendritic_dr.fullres import (Curve, Froc, PatchSampler, Store, augment, build_index, candidates,
                                  grid_targets, lesion_labels, smoke_store, tversky_loss, unpack)
from dendritic_dr.idrid import LESIONS, cell_grids
from dendritic_dr.utils import DATA_DIR, append_result, get_device, results_root, set_seed


def nested_subset(keys: list[str], n: int, seed: int = 2026) -> list[str]:
    """The first n keys of one fixed random order of all training keys: subsets are nested
    (the 25-image set contains the 10-image set) and do not depend on the model or seed."""
    if not 1 <= n <= len(keys):
        raise SystemExit(f"--n-train must be between 1 and {len(keys)}")
    order = np.random.default_rng(seed).permutation(len(keys))
    return [keys[i] for i in sorted(order[:n])]


@torch.no_grad()
def evaluate(model, store, keys, mean, sd, device, amp, shape, grids=None):
    """Cell-level AP / AUC and FROC per lesion type over full images."""
    model.eval()
    curves = {t: Curve() for t in LESIONS}
    frocs = {t: Froc() for t in LESIONS}
    for key in keys:
        green, bits = store.get(key)
        les, valid, _ = grids[key] if grids is not None and key in grids else cell_grids(green, unpack(bits), shape)
        x = (torch.from_numpy(green).to(device).float()[None, None] - mean) / sd
        with torch.autocast(device.type, enabled=amp):
            logit = model(x).float()[0, :, : shape[0], : shape[1]]
        lg = logit.cpu().numpy()
        valid_t = torch.from_numpy(valid).to(device)
        for j, t in enumerate(LESIONS):
            curves[t].add(les[j][valid], lg[j][valid])
            ys, xs, sc = candidates(logit[j], valid_t)
            lab, n = lesion_labels(les[j])
            frocs[t].add(sc, lab[ys, xs], n)
    out = {}
    for t in LESIONS:
        out |= {f"ap_{t}": curves[t].ap(), f"auc_{t}": curves[t].auc(), f"prev_{t}": curves[t].prevalence(),
                f"froc_{t}": frocs[t].score(), f"sens1_{t}": frocs[t].sensitivity_at(1.0),
                f"n_les_{t}": frocs[t].n_lesions}
    out["map"] = float(np.nanmean([out[f"ap_{t}"] for t in LESIONS]))
    out["mfroc"] = float(np.nanmean([out[f"froc_{t}"] for t in LESIONS]))
    out["n_images"] = len(keys)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", choices=STEMS, required=True)
    p.add_argument("--depth", type=int, default=3, help="conv blocks between first layer and readout (0 = linear)")
    p.add_argument("--kernel", type=int, default=12)
    p.add_argument("--cluster", type=int, default=5)
    p.add_argument("--somas", type=int, default=16)
    p.add_argument("--dendrites", type=int, default=8)
    p.add_argument("--synapses", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--train-on", nargs="+", choices=["idrid", "ddr"], default=["idrid", "ddr"])
    p.add_argument("--n-train", type=int, default=None,
                   help="use only this many training images (extension study 1: learning curve); the subsets "
                        "are nested and identical for every first layer and seed")
    p.add_argument("--size", type=int, default=2048, help="fundus diameter in px after resizing")
    p.add_argument("--patch", type=int, default=384, help="training patch size in px (multiple of 4)")
    p.add_argument("--lesion-frac", type=float, default=0.5, help="share of patches centred near a lesion")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=150, help="batches per epoch")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=["tversky", "bce"], default="tversky")
    p.add_argument("--alpha", type=float, default=0.3, help="Tversky weight on false positives")
    p.add_argument("--beta", type=float, default=0.7, help="Tversky weight on misses")
    p.add_argument("--gamma", type=float, default=1.0, help="focal Tversky exponent (1 = plain Tversky)")
    p.add_argument("--max-pos-weight", type=float, default=100.0, help="--loss bce: cap on the rare-positive weight")
    p.add_argument("--n-val-ddr", type=int, default=30)
    p.add_argument("--idrid-dir", default=str(DATA_DIR / "raw" / "idrid"))
    p.add_argument("--ddr-dir", default=str(DATA_DIR / "raw" / "ddr"))
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.patch % 4:
        raise SystemExit("--patch must be a multiple of 4")

    device = get_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    if args.smoke:
        store = smoke_store()
        args.size, args.patch, args.batch_size = 128, 64, 4
        args.steps, args.epochs, args.patience = min(args.steps, 10), min(args.epochs, 4), 4
    else:
        index = build_index(args.idrid_dir, args.ddr_dir, args.size, DATA_DIR / "features" / f"lesions_{args.size}",
                            n_val_ddr=args.n_val_ddr)
        store = Store(index, DATA_DIR / "features" / f"lesions_{args.size}")
    have = set(store.index["dataset"])
    if missing := set(args.train_on) - have:
        raise SystemExit(f"no {', '.join(sorted(missing))} images found; see data/README.md")
    train_keys = store.keys("train", args.train_on)
    n_full = len(train_keys)
    if args.n_train is not None:
        train_keys = nested_subset(train_keys, args.n_train)
    val_keys = store.keys("val", args.train_on)
    test_sets = {ds: store.keys("test", [ds]) for ds in ["idrid", "ddr"] if ds in have}
    print(f"train {len(train_keys)}" + (f" of {n_full}" if len(train_keys) < n_full else "") + f" images ({' + '.join(args.train_on)}), val {len(val_keys)}, test " +
          ", ".join(f"{ds} {len(k)}" for ds, k in test_sets.items()) +
          f"; {args.size} px, {args.patch} px patches; device {device}{' (mixed precision)' if amp else ''}")

    greens, bits = zip(*(store.get(k) for k in train_keys))
    sub = np.concatenate([g[::8, ::8].ravel() for g in greens]).astype(np.float64)
    mean, sd = float(sub.mean()), float(sub.std()) + 1e-6  # training images only
    out_root = results_root(args.smoke)
    train_name = "+".join(args.train_on)

    for seed in args.seeds:
        set_seed(seed)
        model = build_lesion_net(args.stem, depth=args.depth, somas=args.somas, dendrites=args.dendrites,
                                 synapses=args.synapses, kernel=args.kernel, cluster=args.cluster, seed=seed).to(device)
        with torch.no_grad():
            pshape = tuple(model(torch.zeros(1, 1, args.patch, args.patch, device=device)).shape[-2:])
            fshape = tuple(model(torch.zeros(1, 1, args.size, args.size, device=device)).shape[-2:])
        val_grids = {}
        for k in val_keys:
            g, b = store.get(k)
            val_grids[k] = cell_grids(g, unpack(b), fshape)
        sampler = PatchSampler(list(greens), list(bits), args.patch, args.lesion_frac, seed=seed)

        if args.loss == "bce":  # rare-positive weights from a sample of training patches
            g, m = sampler.sample(256)
            y, v = grid_targets(torch.from_numpy(g), torch.from_numpy(m), pshape)
            pos = y.sum((0, 2, 3))
            pos_weight = ((v.sum() - pos) / pos.clamp(min=1)).clamp(max=args.max_pos_weight).to(device)[None, :, None, None]

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        gen = torch.Generator().manual_seed(seed)
        best, best_state, best_epoch, bad = -np.inf, None, 0, 0
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for _ in range(args.steps):
                g, m = sampler.sample(args.batch_size)
                x, mk = augment(torch.from_numpy(g).to(device), torch.from_numpy(m).to(device), gen)
                y, v = grid_targets(x, mk, pshape)
                with torch.autocast(device.type, enabled=amp):
                    logits = model((x.float() - mean) / sd)
                if args.loss == "tversky":
                    loss = tversky_loss(logits, y, v, args.alpha, args.beta, args.gamma)
                else:
                    loss = (F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pos_weight,
                                                               reduction="none") * v).sum() / (v.sum() * 4).clamp(min=1)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            sched.step()
            vm = evaluate(model, store, val_keys, mean, sd, device, amp, fshape, val_grids)
            if vm["mfroc"] > best:
                best, best_state, best_epoch, bad = vm["mfroc"], copy.deepcopy(model.state_dict()), epoch, 0
            else:
                bad += 1
            print(f"  epoch {epoch:3d}  loss {np.mean(losses):.4f}  val mean FROC {vm['mfroc']:.3f}  "
                  f"MA FROC {vm['froc_MA']:.3f}  mAP {vm['map']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
            if bad >= args.patience:
                break
        model.load_state_dict(best_state)

        tag = f"fullres{args.size}_p{args.patch}_{train_name}_{args.loss}_{args.stem}_k{args.kernel}_depth{args.depth}"
        if args.n_train is not None:
            tag = tag.replace(f"_{train_name}_", f"_{train_name}_n{len(train_keys)}_")
        if args.stem.startswith("dend"):
            tag += f"_s{args.somas}_d{args.dendrites}_y{args.synapses}" + (f"_c{args.cluster}" if args.stem == "dend-lrf" else "")
        row = {"run_id": f"{tag}_seed{seed}", "config": tag, "stem": args.stem, "kernel": args.kernel,
               "cluster": args.cluster if args.stem == "dend-lrf" else None,
               "depth": args.depth, "train_on": train_name, "loss": args.loss, "size": args.size, "seed": seed,
               "n_train": len(train_keys), "n_train_idrid": sum(k.startswith("idrid/") for k in train_keys),
               "stem_params": model.stem_params(), "total_params": model.effective_params(),
               "best_epoch": best_epoch, "epochs_run": epoch, "val_mfroc": best}
        for ds, keys in test_sets.items():
            tm = evaluate(model, store, keys, mean, sd, device, amp, fshape)
            row |= {f"{ds}_{k}": v for k, v in tm.items()}
            print(f"[seed {seed}] {args.stem:8s} k{args.kernel:<2d} {ds:5s} test: mean FROC {tm['mfroc']:.3f}  "
                  "FROC " + " ".join(f"{t} {tm[f'froc_{t}']:.3f}" for t in LESIONS) +
                  f"  MA sens@1FP {tm['sens1_MA']:.3f}  mAP {tm['map']:.3f}", flush=True)
        print(f"[seed {seed}] best epoch {best_epoch} of {epoch}  ({time.time() - t0:.0f} s)")
        (out_root / "models").mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "args": vars(args) | {"seed": seed}, "mean": mean, "sd": sd},
                   out_root / "models" / f"{row['run_id']}.pt")
        append_result(out_root / "fullres_runs.csv", row)


if __name__ == "__main__":
    main()
