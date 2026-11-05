"""Dendritic ANN (dANN) and vanilla MLP baseline.

A simplified re-implementation of the dendritic ANN idea in
Chavlis & Poirazi (2025), Nature Communications, doi:10.1038/s41467-025-56297-9:
restricted input sampling per dendrite + each dendrite feeding a single soma.
It is NOT the authors' official code; compare against their release before
claiming an exact replication.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import local_rf_dendrite_mask, random_dendrite_mask, soma_mask


class MaskedLinear(nn.Module):
    """Linear layer whose weight is multiplied by a fixed boolean mask.

    Masked-out weights receive zero gradient, so they never become active.
    Initialisation uses each unit's *effective* fan-in (number of unmasked
    inputs) so sparse units are not under-scaled.
    """

    def __init__(self, mask: torch.Tensor, bias: bool = True):
        super().__init__()
        out_f, in_f = mask.shape
        self.register_buffer("mask", mask.float())
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.mask.sum(dim=1, keepdim=True).clamp(min=1)
        bound = torch.sqrt(6.0 / fan_in)  # He-uniform for ReLU
        with torch.no_grad():
            self.weight.uniform_(-1.0, 1.0).mul_(bound).mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    def effective_params(self) -> int:
        n = int(self.mask.sum().item())
        return n + (self.bias.numel() if self.bias is not None else 0)


class IndexedMaskedLinear(nn.Module):
    """Memory-light equivalent of MaskedLinear for very large, very sparse layers.

    Instead of a dense (out x in) weight and mask, each output unit stores the column
    indices of its k inputs (buffer `idx`, sorted per row) and k weights. The forward
    pass gathers those inputs: y = sum_j x[:, idx[:, j]] * weight[:, j] + bias. This is
    the same function as MaskedLinear with the same mask, but memory grows with the
    number of synapses, not with in_features (a 2,048-dendrite layer on 200k inputs:
    ~130 KB of weights instead of several GB). Used automatically by `masked_linear`.
    """

    def __init__(self, mask: torch.Tensor, bias: bool = True):
        super().__init__()
        mask = mask.bool()
        out_f, in_f = mask.shape
        counts = mask.sum(dim=1)
        if not bool((counts == counts[0]).all()):
            raise ValueError("IndexedMaskedLinear needs the same number of inputs per unit")
        k = int(counts[0])
        idx = torch.nonzero(mask)[:, 1].reshape(out_f, k)  # row-major: columns ascending per row
        self.in_features = in_f
        self.register_buffer("idx", idx)
        self.weight = nn.Parameter(torch.empty(out_f, k))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = (6.0 / max(self.weight.shape[1], 1)) ** 0.5  # He-uniform on the effective fan-in
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x[:, self.idx] * self.weight).sum(-1)
        return y + self.bias if self.bias is not None else y

    def dense_mask(self) -> torch.Tensor:
        m = torch.zeros(self.idx.shape[0], self.in_features, dtype=torch.bool, device=self.idx.device)
        m.scatter_(1, self.idx, True)
        return m

    def effective_params(self) -> int:
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)


# Dense masked layers above this many entries (out x in) switch to IndexedMaskedLinear.
# All earlier experiments are below it (largest: 4,096 x 25,088 = 1.0e8), so their
# checkpoints and results are unchanged.
INDEXED_THRESHOLD = 200_000_000


def masked_linear(mask: torch.Tensor, bias: bool = True) -> nn.Module:
    if mask.numel() > INDEXED_THRESHOLD:
        return IndexedMaskedLinear(mask, bias)
    return MaskedLinear(mask, bias)


def _is_masked(m: nn.Module) -> bool:
    return isinstance(m, (MaskedLinear, IndexedMaskedLinear))


class DendriticLayer(nn.Module):
    """inputs -> dendrites (ReLU) -> somas (ReLU)."""

    def __init__(self, dendrite_mask: torch.Tensor, n_somas: int, dendrites_per_soma: int):
        super().__init__()
        self.dendrites = masked_linear(dendrite_mask)
        self.somas = MaskedLinear(soma_mask(n_somas, dendrites_per_soma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.somas(F.relu(self.dendrites(x))))


class DANN(nn.Module):
    """Stack of dendritic layers followed by a dense output layer.

    sampling="random": each dendrite samples `synapses` random inputs.
    sampling="lrf":    first layer uses local receptive fields on an image of
                       shape `image_shape` (C, H, W) with patch size `rf_size`;
                       deeper layers use random sampling (their inputs are soma
                       activations with no spatial layout).
    """

    def __init__(
        self,
        in_features: int,
        n_out: int,
        somas: list[int],
        dendrites_per_soma: int = 16,
        synapses: int = 16,
        sampling: str = "random",
        image_shape: tuple[int, int, int] | None = None,
        rf_size: int = 4,
        dropout: float = 0.0,
        seed: int = 0,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)
        layers: list[nn.Module] = []
        prev = in_features
        for i, n_somas in enumerate(somas):
            n_dend = n_somas * dendrites_per_soma
            if i == 0 and sampling == "lrf":
                if image_shape is None:
                    raise ValueError("sampling='lrf' needs image_shape")
                c = image_shape[0]
                # single-channel images (Fashion-MNIST): full patch, as in the original setup;
                # many-channel feature maps: `synapses` sampled inside each local patch
                lrf_syn = None if c * rf_size * rf_size <= synapses else synapses
                dmask = local_rf_dendrite_mask(image_shape, n_somas, dendrites_per_soma, rf_size, rng,
                                               synapses=lrf_syn)
            elif sampling in ("random", "lrf"):
                dmask = random_dendrite_mask(prev, n_dend, min(synapses, prev), rng)
            else:
                raise ValueError(f"unknown sampling {sampling!r}")
            layers.append(DendriticLayer(dmask, n_somas, dendrites_per_soma))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = n_somas
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(prev, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x.flatten(1)))


class VanillaMLP(nn.Module):
    """Fully connected baseline: Linear -> ReLU per hidden layer, dense output."""

    def __init__(self, in_features: int, n_out: int, hidden: list[int], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(prev, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x.flatten(1)))


def _mask_signature(mask: torch.Tensor) -> list[int]:
    """Cheap fingerprint of a mask's nonzero positions (count + index checksum)."""
    idx = torch.nonzero(mask.bool().flatten()).flatten().to(torch.int64)
    return [int(idx.numel()), int((idx % 1_000_003).sum().item()), int(mask.shape[0]), int(mask.shape[1])]


def _layer_signature(m: nn.Module) -> list[int]:
    """Mask fingerprint of either masked layer type (identical for the same mask)."""
    if isinstance(m, IndexedMaskedLinear):
        rows = torch.arange(m.idx.shape[0], device=m.idx.device).unsqueeze(1)
        flat = (rows * m.in_features + m.idx).flatten().cpu().to(torch.int64)
        return [int(flat.numel()), int((flat % 1_000_003).sum().item()), int(m.idx.shape[0]), int(m.in_features)]
    return _mask_signature(m.mask.cpu())


def _unmasked_values(m: nn.Module) -> torch.Tensor:
    """Weights at unmasked positions, in row-major order (same order for both types)."""
    if isinstance(m, IndexedMaskedLinear):
        return m.weight.detach().cpu().flatten().clone()
    return m.weight.detach().cpu()[m.mask.cpu().bool()].clone()


def compact_state_dict(model: nn.Module) -> dict:
    """State dict without the dense masked matrices: for every MaskedLinear only the
    weights at unmasked positions are kept (plus a mask fingerprint). Masks are
    regenerated from the mask seed when loading. A spatial dANN layer shrinks from
    hundreds of MB (dense weight + dense mask) to ~100 KB."""
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for name, m in model.named_modules():
        if _is_masked(m):
            sd.pop(f"{name}.weight")
            sd.pop(f"{name}.mask" if isinstance(m, MaskedLinear) else f"{name}.idx")
            sd[f"{name}.weight_values"] = _unmasked_values(m)
            sd[f"{name}.mask_signature"] = torch.tensor(_layer_signature(m))
    return sd


def load_compact_state_dict(model: nn.Module, sd: dict) -> None:
    """Inverse of compact_state_dict. `model` must be built with the same args (same
    mask seed), so its regenerated masks match; this is checked via the fingerprint."""
    sd = dict(sd)
    for name, m in model.named_modules():
        if _is_masked(m):
            vals = sd.pop(f"{name}.weight_values")
            sig = sd.pop(f"{name}.mask_signature").tolist()
            if sig != _layer_signature(m):
                raise ValueError(f"regenerated mask for {name!r} does not match the checkpoint")
            with torch.no_grad():
                if isinstance(m, IndexedMaskedLinear):
                    m.weight.copy_(vals.reshape(m.weight.shape).to(m.weight.device, m.weight.dtype))
                else:
                    m.weight.zero_()
                    m.weight[m.mask.bool()] = vals.to(m.weight.device, m.weight.dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    masked = {f"{n}.{t}" for n, mm in model.named_modules() if _is_masked(mm)
              for t in ("weight", "mask", "idx")}
    if set(missing) - masked or unexpected:
        raise ValueError(f"checkpoint mismatch: missing {set(missing) - masked}, unexpected {unexpected}")


def effective_params(model: nn.Module) -> int:
    """Trainable parameters that can actually be non-zero (masked weights excluded)."""
    total = 0
    masked_ids = set()
    for m in model.modules():
        if _is_masked(m):
            total += m.effective_params()
            masked_ids.update(id(p) for p in m.parameters())
    total += sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in masked_ids)
    return total


def build_model(args, in_features: int, n_out: int, image_shape=None) -> nn.Module:
    """Construct a model from parsed CLI args (shared by the training scripts)."""
    if args.model == "vanilla":
        return VanillaMLP(in_features, n_out, args.hidden, dropout=args.dropout)
    sampling = "lrf" if args.model == "dann-lrf" else "random"
    if image_shape is None:
        image_shape = getattr(args, "image_shape", None)
    if image_shape is not None:
        image_shape = tuple(int(v) for v in image_shape)
    return DANN(
        in_features,
        n_out,
        somas=args.somas,
        dendrites_per_soma=args.dendrites,
        synapses=args.synapses,
        sampling=sampling,
        image_shape=image_shape,
        rf_size=args.rf_size,
        dropout=args.dropout,
        seed=args.mask_seed if args.mask_seed is not None else args.seed,
    )


def add_model_args(p) -> None:
    """CLI options shared by the training scripts."""
    p.add_argument("--model", choices=["dann-r", "dann-lrf", "vanilla"], default="dann-r")
    p.add_argument("--somas", type=int, nargs="+", default=[256], help="somas per dendritic layer")
    p.add_argument("--dendrites", type=int, default=16, help="dendrites per soma")
    p.add_argument("--synapses", type=int, default=16, help="inputs per dendrite (random sampling)")
    p.add_argument("--rf-size", type=int, default=4, help="patch size for dann-lrf")
    p.add_argument("--hidden", type=int, nargs="+", default=[256], help="hidden sizes for vanilla MLP")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mask-seed", type=int, default=None, help="defaults to --seed")


__all__ = [
    "MaskedLinear",
    "IndexedMaskedLinear",
    "masked_linear",
    "DendriticLayer",
    "DANN",
    "VanillaMLP",
    "effective_params",
    "compact_state_dict",
    "load_compact_state_dict",
    "build_model",
    "add_model_args",
]
