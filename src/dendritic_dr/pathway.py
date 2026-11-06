"""Pathway-wired dendritic networks for gene expression (extension, study 2).

Each dendrite is one Reactome pathway: it receives only that pathway's genes. Each soma is one
top-level Reactome category (e.g. Immune System, Extracellular matrix organization) and sums the
dendrites of the pathways under it. The control keeps every dendrite's size but draws its genes
at random, so the two networks have identical parameter counts and differ only in whether the
wiring follows biology (as clustered vs. scattered dendrites differed only in locality).

Masks come from two Reactome files (https://reactome.org/download-data):
  ReactomePathways.gmt            pathway name, stable id, member gene symbols
  ReactomePathwaysRelation.txt    parent -> child stable ids (all species; human = R-HSA-)
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import MaskedLinear

REACTOME_GMT_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
REACTOME_REL_URL = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"


# ----------------------------------------------------------------------------- Reactome

def read_gmt(path: Path) -> dict[str, tuple[str, set[str]]]:
    """{stable id: (name, genes)} from a GMT file (plain or zipped)."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            text = z.read([n for n in z.namelist() if n.endswith(".gmt")][0]).decode()
    else:
        text = path.read_text()
    out = {}
    for line in io.StringIO(text):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            out[parts[1]] = (parts[0], {g for g in parts[2:] if g})
    return out


def read_relations(path: Path, prefix: str = "R-HSA-") -> dict[str, set[str]]:
    """{child: parents} for one species."""
    parents: dict[str, set[str]] = {}
    for line in Path(path).read_text().splitlines():
        a, _, b = line.partition("\t")
        if a.startswith(prefix) and b.startswith(prefix):
            parents.setdefault(b.strip(), set()).add(a.strip())
    return parents


def top_levels(pid: str, parents: dict[str, set[str]]) -> set[str]:
    """Ancestors of pid with no parent (pid itself if it has none)."""
    roots, stack, seen = set(), [pid], set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        ps = parents.get(p, set())
        if not ps:
            roots.add(p)
        stack.extend(ps)
    return roots


def build_masks(genes: list[str], gmt: dict, parents: dict, min_size: int = 10, max_size: int = 300):
    """Pathway (dendrite) and category (soma) masks over the given genes.

    Pathways keep only genes present in `genes` and are kept if min_size <= size <= max_size.
    A pathway feeds every top-level category it sits under. Returns a dict with
    pathway ids/names, category ids/names, M1 (pathways x genes) and M2 (categories x pathways)."""
    gidx = {g: i for i, g in enumerate(genes)}
    pids, rows = [], []
    for pid, (name, members) in sorted(gmt.items()):
        cols = sorted(gidx[g] for g in members if g in gidx)
        if min_size <= len(cols) <= max_size:
            pids.append(pid)
            rows.append(cols)
    if not pids:
        raise ValueError("no pathway passes the size filter; check gene symbols")
    M1 = np.zeros((len(pids), len(genes)), bool)
    for r, cols in enumerate(rows):
        M1[r, cols] = True
    tops = [top_levels(p, parents) for p in pids]
    cats = sorted(set().union(*tops))
    cidx = {c: i for i, c in enumerate(cats)}
    M2 = np.zeros((len(cats), len(pids)), bool)
    for j, ts in enumerate(tops):
        for t in ts:
            M2[cidx[t], j] = True
    name = {pid: gmt[pid][0] if pid in gmt else pid for pid in set(pids) | set(cats)}
    return {"pathways": pids, "pathway_names": [name[p] for p in pids], "categories": cats,
            "category_names": [name[c] for c in cats], "M1": M1, "M2": M2}


def random_like(M1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same number of genes per dendrite, genes drawn uniformly at random."""
    out = np.zeros_like(M1)
    for r, k in enumerate(M1.sum(1)):
        out[r, rng.choice(M1.shape[1], int(k), replace=False)] = True
    return out


# ----------------------------------------------------------------------------- expression

def log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """genes x samples counts -> log2(CPM + 1), computed per sample (no information shared
    between samples, so it can be done before splitting)."""
    lib = counts.sum(axis=0).replace(0, 1)
    return np.log2(counts / lib * 1e6 + 1)


def expressed(counts: pd.DataFrame, min_cpm: float = 1.0, min_frac: float = 0.2) -> pd.Index:
    """Genes with CPM >= min_cpm in >= min_frac of samples (uses no labels)."""
    cpm = counts / counts.sum(axis=0).replace(0, 1) * 1e6
    return counts.index[(cpm >= min_cpm).mean(axis=1) >= min_frac]


# ----------------------------------------------------------------------------- models

class PathwayDANN(nn.Module):
    """genes -> pathway dendrites (ReLU) -> category somas (ReLU) -> logit."""

    def __init__(self, M1: np.ndarray, M2: np.ndarray, dropout: float = 0.0):
        super().__init__()
        self.dend = MaskedLinear(torch.as_tensor(M1))
        self.soma = MaskedLinear(torch.as_tensor(M2))
        self.out = nn.Linear(M2.shape[0], 1)
        self.drop = nn.Dropout(dropout)

    def dendrites(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.dend(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(F.relu(self.soma(self.drop(self.dendrites(x)))))).squeeze(-1)

    def from_dendrites(self, d: torch.Tensor) -> torch.Tensor:
        return self.out(F.relu(self.soma(d))).squeeze(-1)

    def effective_params(self) -> int:
        return self.dend.effective_params() + self.soma.effective_params() + sum(p.numel() for p in self.out.parameters())


class DenseMLP(nn.Module):
    def __init__(self, n_in: int, h1: int, h2: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h1, h2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(h2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def effective_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def matched_dense(n_in: int, target: int, h2: int) -> tuple[int, int]:
    """Hidden sizes (h1, h2) for a dense MLP with about `target` parameters; h2 is the number of
    somas in the pathway network, h1 >= 1 absorbs the rest."""
    # params = h1 (n_in + 1) + h2 (h1 + 1) + h2 + 1
    h1 = max(1, round((target - 2 * h2 - 1) / (n_in + 1 + h2)))
    return h1, h2


def train_torch(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 3e-3,
                weight_decay: float = 1e-2, seed: int = 0) -> nn.Module:
    """Full-batch AdamW on class-balanced binary cross-entropy; a fixed number of epochs (with
    ~50 training samples there is no room for an inner validation split)."""
    torch.manual_seed(seed)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    pos = yt.sum().clamp(min=1)
    pos_weight = (len(yt) - pos) / pos
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(model(Xt), yt, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()


def dendrite_contributions(model: PathwayDANN, X: np.ndarray) -> np.ndarray:
    """Gradient x activation of the logit with respect to each dendrite, per sample
    (samples x pathways): how much each pathway dendrite pushed each prediction."""
    d = model.dendrites(torch.as_tensor(X, dtype=torch.float32)).detach().requires_grad_(True)
    model.from_dendrites(d).sum().backward()
    return (d.grad * d).detach().numpy()


# ----------------------------------------------------------------------------- smoke data

def smoke_expression(n_donors: int = 20, n_genes: int = 400, n_path: int = 40, seed: int = 0,
                     effect: float = 1.25):
    """Synthetic counts (two samples per donor), labels, a GMT and relations; the label is
    carried by the genes of the first two pathways."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    gmt = {f"R-HSA-{i}": (f"pathway {i}", set(rng.choice(genes, int(rng.integers(12, 40)), replace=False)))
           for i in range(1, n_path + 1)}
    parents = {f"R-HSA-{i}": {f"R-HSA-{1000 + i % 3}"} for i in range(1, n_path + 1)}
    for t in range(3):
        gmt[f"R-HSA-{1000 + t}"] = (f"category {t}", set())
    y_donor = np.r_[np.ones(n_donors // 2), np.zeros(n_donors - n_donors // 2)].astype(int)
    rows, cols, labels, donors = [], [], [], []
    signal = sorted(gmt["R-HSA-1"][1] | gmt["R-HSA-2"][1])
    base = rng.gamma(2.0, 50.0, n_genes)
    for d in range(n_donors):
        for site in ("Macula", "Periphery"):
            mu = base * rng.lognormal(0, 0.3, n_genes)
            if y_donor[d]:
                mu[[genes.index(g) for g in signal]] *= effect
            rows.append(rng.poisson(mu))
            cols.append(f"s{d}_{site}")
            labels.append(y_donor[d])
            donors.append(f"d{d}")
    counts = pd.DataFrame(np.array(rows).T, index=genes, columns=cols)
    meta = pd.DataFrame({"title": cols, "donor": donors, "y": labels})
    return counts, meta, gmt, parents
