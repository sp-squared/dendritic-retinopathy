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


class DendriticLayer(nn.Module):
    """inputs -> dendrites (ReLU) -> somas (ReLU)."""

    def __init__(self, dendrite_mask: torch.Tensor, n_somas: int, dendrites_per_soma: int):
        super().__init__()
        self.dendrites = MaskedLinear(dendrite_mask)
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
                dmask = local_rf_dendrite_mask(image_shape, n_somas, dendrites_per_soma, rf_size, rng)
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


def effective_params(model: nn.Module) -> int:
    """Trainable parameters that can actually be non-zero (masked weights excluded)."""
    total = 0
    masked_ids = set()
    for m in model.modules():
        if isinstance(m, MaskedLinear):
            total += m.effective_params()
            masked_ids.update(id(p) for p in m.parameters())
    total += sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in masked_ids)
    return total


def build_model(args, in_features: int, n_out: int, image_shape=None) -> nn.Module:
    """Construct a model from parsed CLI args (shared by the training scripts)."""
    if args.model == "vanilla":
        return VanillaMLP(in_features, n_out, args.hidden, dropout=args.dropout)
    sampling = "lrf" if args.model == "dann-lrf" else "random"
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
    "DendriticLayer",
    "DANN",
    "VanillaMLP",
    "effective_params",
    "build_model",
    "add_model_args",
]
