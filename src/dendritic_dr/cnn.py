"""Small CNN trained end to end on fundus images, with a dendritic or dense first layer.

The heads in steps 4-4e sat on top of a frozen backbone or on downsampled pixels, where a
microaneurysm is smaller than one input. Here the first trainable layer sees 512 px images,
where small lesions span several pixels, and it is applied convolutionally, so a lesion is
detected wherever it lies in the fundus (a single non-convolutional layer would only look
at fixed positions).

First-layer variants (same output channels, same receptive field, identical parameter
counts for the three 12 x 12 variants):

  dend-lrf  dendritic convolution: each output channel is a soma with D dendrites; each
            dendrite samples S inputs CLUSTERED inside a random c x c sub-window of the
            k x k receptive field (local receptive field), ReLU per dendrite, the soma sums
            its dendrites (one weight each), ReLU.
  dend-r    the same, but each dendrite's S inputs are SCATTERED anywhere in the k x k field.
  conv      an ordinary dense k x k convolution + ReLU (k = 12: parameter-matched to the
            dendritic layers; k = 5: a conventional small kernel).

The rest of the network (four 3 x 3 conv-BN-ReLU-maxpool blocks, global average pooling,
dropout, linear output) is identical for all variants.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_dendrite_mask(n_dendrites: int, in_ch: int, kernel: int, synapses: int,
                       rng: np.random.Generator, cluster: int | None = None) -> torch.Tensor:
    """Boolean kernel mask (n_dendrites, in_ch, kernel, kernel) with `synapses` ones per dendrite.

    cluster=None: positions drawn anywhere in the in_ch x kernel x kernel field (scattered).
    cluster=c:    each dendrite picks a random c x c sub-window of the field and draws its
                  positions inside it (clustered, i.e. a local receptive field)."""
    if cluster is not None and not 1 <= cluster <= kernel:
        raise ValueError("cluster must be between 1 and kernel")
    space = in_ch * (cluster or kernel) ** 2
    if synapses > space:
        raise ValueError(f"{synapses} synapses do not fit in {space} positions")
    mask = np.zeros((n_dendrites, in_ch, kernel, kernel), dtype=bool)
    for d in range(n_dendrites):
        if cluster is None:
            flat = rng.choice(in_ch * kernel * kernel, size=synapses, replace=False)
            c, y, x = np.unravel_index(flat, (in_ch, kernel, kernel))
        else:
            y0, x0 = rng.integers(0, kernel - cluster + 1, size=2)
            flat = rng.choice(space, size=synapses, replace=False)
            c, y, x = np.unravel_index(flat, (in_ch, cluster, cluster))
            y, x = y + y0, x + x0
        mask[d, c, y, x] = True
    return torch.from_numpy(mask)


class DendriticConv2d(nn.Module):
    """Convolutional dendritic layer: masked dendrite kernels -> ReLU -> per-soma sum -> ReLU.

    Equivalent to applying one dANN layer (restricted sampling, each dendrite feeding one
    soma) at every image position with shared weights."""

    def __init__(self, in_ch: int, somas: int, dendrites: int, synapses: int, kernel: int,
                 stride: int, cluster: int | None, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        n_dend = somas * dendrites
        self.stride, self.padding = stride, kernel // 2
        self.register_buffer("mask", conv_dendrite_mask(n_dend, in_ch, kernel, synapses, rng, cluster).float())
        self.weight = nn.Parameter(torch.empty(n_dend, in_ch, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(n_dend))
        bound = (6.0 / synapses) ** 0.5  # He-uniform on the effective fan-in
        with torch.no_grad():
            self.weight.uniform_(-bound, bound).mul_(self.mask)
        # soma stage: each soma sums its own dendrites (grouped 1 x 1 conv, one weight per dendrite)
        self.soma = nn.Conv2d(n_dend, somas, kernel_size=1, groups=somas)
        nn.init.kaiming_uniform_(self.soma.weight, nonlinearity="relu")
        nn.init.zeros_(self.soma.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = F.relu(F.conv2d(x, self.weight * self.mask, self.bias, stride=self.stride, padding=self.padding))
        return F.relu(self.soma(d))

    def effective_params(self) -> int:
        return int(self.mask.sum().item()) + self.bias.numel() + sum(p.numel() for p in self.soma.parameters())


class DenseConvStem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x))

    def effective_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SmallCNN(nn.Module):
    def __init__(self, stem: nn.Module, stem_out: int, widths=(32, 64, 128, 128), n_out: int = 5,
                 dropout: float = 0.2):
        super().__init__()
        self.stem = stem
        blocks, prev = [], stem_out
        for w in widths:
            blocks += [nn.Conv2d(prev, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2)]
            prev = w
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(prev, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))

    def stem_params(self) -> int:
        return self.stem.effective_params()

    def effective_params(self) -> int:
        rest = sum(p.numel() for n, p in self.named_parameters() if not n.startswith("stem."))
        return self.stem_params() + rest


STEMS = ["dend-lrf", "dend-r", "conv"]


def build_cnn(stem: str, in_ch: int = 1, somas: int = 16, dendrites: int = 8, synapses: int = 16,
              kernel: int = 12, cluster: int = 5, stride: int = 4, seed: int = 0, dropout: float = 0.2) -> SmallCNN:
    if stem == "dend-lrf":
        s = DendriticConv2d(in_ch, somas, dendrites, synapses, kernel, stride, cluster, seed)
    elif stem == "dend-r":
        s = DendriticConv2d(in_ch, somas, dendrites, synapses, kernel, stride, None, seed)
    elif stem == "conv":
        s = DenseConvStem(in_ch, somas, kernel, stride)
    else:
        raise ValueError(f"unknown stem {stem!r}; choose from {STEMS}")
    return SmallCNN(s, somas, dropout=dropout)


class LesionNet(nn.Module):
    """Dense lesion prediction on the first layer's stride-4 grid (one logit per lesion type
    per cell). depth = 0: a 1 x 1 linear readout directly on the first-layer units, so those
    units themselves must carry the lesion signal. depth > 0: that many 3 x 3 conv-BN-ReLU
    blocks (dilations 1, 2, 4, ...: more context, same resolution) before the readout."""

    def __init__(self, stem: nn.Module, stem_out: int, depth: int = 0, width: int = 32, n_out: int = 4):
        super().__init__()
        self.stem, self.depth = stem, depth
        blocks, prev = [], stem_out
        for i in range(depth):
            d = 2 ** i
            blocks += [nn.Conv2d(prev, width, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(width), nn.ReLU()]
            prev = width
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv2d(prev, n_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))

    def stem_params(self) -> int:
        return self.stem.effective_params()

    def effective_params(self) -> int:
        return self.stem_params() + sum(p.numel() for n, p in self.named_parameters() if not n.startswith("stem."))


def build_lesion_net(stem: str, depth: int = 0, in_ch: int = 1, somas: int = 16, dendrites: int = 8,
                     synapses: int = 16, kernel: int = 12, cluster: int = 5, stride: int = 4, seed: int = 0,
                     width: int = 32) -> LesionNet:
    """Same first-layer variants as build_cnn, with a lesion readout instead of a grade head."""
    s = build_cnn(stem, in_ch, somas, dendrites, synapses, kernel, cluster, stride, seed).stem
    return LesionNet(s, somas, depth=depth, width=width)
