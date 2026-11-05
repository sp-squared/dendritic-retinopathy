"""Shrink saved head checkpoints by dropping dense masked matrices.

Older checkpoints stored each dendritic layer as a dense weight matrix plus a dense
mask of the same size, although almost every entry is zero (a spatial dANN layer is
2,048 x 25,088, i.e. hundreds of MB). The compact format keeps only the weights at
unmasked positions; masks are regenerated from the mask seed when loading.

For every checkpoint this script
  1. rebuilds the model from its saved arguments and checks that the regenerated
     masks are IDENTICAL to the stored ones,
  2. writes the compact version and checks that it reproduces the original model's
     outputs exactly on random inputs,
  3. only then replaces the file. Any checkpoint failing a check is left untouched.

Example
-------
python scripts/compact_checkpoints.py              # results/models/
python scripts/compact_checkpoints.py --dry-run    # report only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dendritic_dr.analysis import load_checkpoint
from dendritic_dr.models import MaskedLinear, build_model, compact_state_dict
from dendritic_dr.utils import results_root


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@torch.no_grad()
def convert(path: Path, dry_run: bool) -> tuple[str, int, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    before = path.stat().st_size
    if ckpt.get("compact"):
        return "already compact", before, before

    import argparse as ap

    model = build_model(ap.Namespace(**ckpt["model_args"]), ckpt["in_features"], ckpt["n_out"])
    stored = ckpt["state_dict"]
    for name, m in model.named_modules():  # 1. regenerated masks must match exactly
        if isinstance(m, MaskedLinear) and not torch.equal(m.mask.cpu(), stored[f"{name}.mask"].cpu()):
            return f"SKIPPED: regenerated mask for {name} differs", before, before
    model.load_state_dict(stored)
    model.eval()

    new = dict(ckpt, state_dict=compact_state_dict(model), compact=True)
    tmp = path.with_suffix(".compact.tmp")
    torch.save(new, tmp)
    restored, _ = load_checkpoint(tmp)  # 2. outputs must be identical
    x = torch.randn(64, ckpt["in_features"])
    if not torch.equal(model(x), restored(x)):
        tmp.unlink()
        return "SKIPPED: compact model outputs differ", before, before
    after = tmp.stat().st_size
    if dry_run:
        tmp.unlink()
        return "ok (dry run)", before, after
    tmp.replace(path)  # 3. replace only after both checks pass
    return "compacted", before, after


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=None, help="defaults to results/models (results/smoke/models with --smoke)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    folder = Path(args.dir) if args.dir else results_root(args.smoke) / "models"
    files = sorted(folder.glob("*.pt"))
    total_before = total_after = 0
    skipped = []
    for f in files:
        status, b, a = convert(f, args.dry_run)
        total_before += b
        total_after += a
        if status.startswith("SKIPPED"):
            skipped.append((f.name, status))
        print(f"{status:>18}  {human(b):>9} -> {human(a):>9}  {f.name}")
    print(f"\n{len(files)} checkpoints: {human(total_before)} -> {human(total_after)}")
    if skipped:
        print(f"{len(skipped)} left unchanged (see SKIPPED lines above)")


if __name__ == "__main__":
    main()
