"""Connectivity masks for dendritic layers.

A dendritic layer has two sparse stages:
  inputs  --(dendrite mask)-->  dendrites  --(soma mask)-->  somas
The dendrite mask decides which inputs each dendrite samples (restricted
sampling); the soma mask wires each dendrite to exactly one soma
(structured connectivity). Masks are boolean tensors of shape (out, in).
"""
from __future__ import annotations

import numpy as np
import torch


def random_dendrite_mask(
    in_features: int,
    n_dendrites: int,
    synapses_per_dendrite: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Each dendrite samples `synapses_per_dendrite` inputs uniformly at random
    (the "random sampling" variant, usable for any input, e.g. feature vectors)."""
    if synapses_per_dendrite > in_features:
        raise ValueError(
            f"synapses_per_dendrite ({synapses_per_dendrite}) > in_features ({in_features})"
        )
    mask = np.zeros((n_dendrites, in_features), dtype=bool)
    for d in range(n_dendrites):
        idx = rng.choice(in_features, size=synapses_per_dendrite, replace=False)
        mask[d, idx] = True
    return torch.from_numpy(mask)


def local_rf_dendrite_mask(
    image_shape: tuple[int, int, int],
    n_somas: int,
    dendrites_per_soma: int,
    rf_size: int,
    rng: np.random.Generator,
    jitter: int | None = None,
) -> torch.Tensor:
    """Local receptive fields for image inputs flattened as (C, H, W).

    Each soma gets a random centre on the image; each of its dendrites samples
    an `rf_size` x `rf_size` patch (all channels) centred near that soma centre
    (within +/- `jitter` pixels). Dendrites of one soma therefore cover a local
    neighbourhood, loosely mimicking spatially clustered synaptic input.
    """
    c, h, w = image_shape
    if jitter is None:
        jitter = max(1, rf_size // 2)
    half = rf_size // 2
    mask = np.zeros((n_somas * dendrites_per_soma, c, h, w), dtype=bool)
    for s in range(n_somas):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        for j in range(dendrites_per_soma):
            dy, dx = rng.integers(-jitter, jitter + 1, size=2)
            y = int(np.clip(cy + dy, half, h - 1 - half))
            x = int(np.clip(cx + dx, half, w - 1 - half))
            d = s * dendrites_per_soma + j
            mask[d, :, y - half : y - half + rf_size, x - half : x - half + rf_size] = True
    return torch.from_numpy(mask.reshape(n_somas * dendrites_per_soma, c * h * w))


def soma_mask(n_somas: int, dendrites_per_soma: int) -> torch.Tensor:
    """Block-diagonal mask: soma i receives only from its own dendrites."""
    mask = torch.zeros(n_somas, n_somas * dendrites_per_soma, dtype=torch.bool)
    for s in range(n_somas):
        mask[s, s * dendrites_per_soma : (s + 1) * dendrites_per_soma] = True
    return mask
