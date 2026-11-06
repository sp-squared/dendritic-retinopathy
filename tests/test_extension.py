"""Tests for the extension: learning-curve helpers (study 1) and pathway-wired dANNs (study 2)."""
import importlib.util

import numpy as np
import torch

from dendritic_dr.pathway import (DenseMLP, PathwayDANN, build_masks, dendrite_contributions, expressed, log_cpm,
                                  matched_dense, random_like, smoke_expression, top_levels, train_torch)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_nested_subsets_are_nested_and_fixed():
    s20 = load("scripts/20_train_fullres.py", "s20")
    keys = [f"ddr/{i}" for i in range(100)]
    a, b = s20.nested_subset(keys, 10), s20.nested_subset(keys, 25)
    assert len(a) == 10 and len(b) == 25 and set(a) <= set(b)
    assert a == s20.nested_subset(keys, 10)  # same subset every time (every layer, every seed)


def test_crossover_interpolates_on_log_scale():
    s22 = load("scripts/22_learning_curve.py", "s22")
    assert np.isclose(s22.crossover([10, 100], [1.0, -1.0]), np.sqrt(10 * 100))
    assert np.isnan(s22.crossover([10, 100, 400], [0.1, 0.2, 0.3]))      # always ahead
    assert np.isnan(s22.crossover([10, 100, 400], [-0.1, -0.2, -0.3]))   # always behind
    assert np.isclose(s22.crossover([10, 25, 50], [0.2, 0.1, 0.0]), 50)


def test_masks_follow_pathways_and_hierarchy():
    gmt = {"R-HSA-1": ("A", {f"g{i}" for i in range(12)}), "R-HSA-2": ("B", {f"g{i}" for i in range(5, 20)}),
           "R-HSA-3": ("tiny", {"g1", "g2"}), "R-HSA-9": ("Top", set())}
    parents = {"R-HSA-1": {"R-HSA-9"}, "R-HSA-2": {"R-HSA-1"}}
    assert top_levels("R-HSA-2", parents) == {"R-HSA-9"}
    genes = [f"g{i}" for i in range(20)]
    m = build_masks(genes, gmt, parents, min_size=10, max_size=300)
    assert m["pathways"] == ["R-HSA-1", "R-HSA-2"]  # the 2-gene pathway is dropped
    assert m["M1"].sum(1).tolist() == [12, 15]
    assert m["category_names"] == ["Top"] and m["M2"].all()


def test_random_control_keeps_sizes_and_parameters():
    counts, meta, gmt, parents = smoke_expression()
    genes = sorted(set(expressed(counts)) & set().union(*(g for _, g in gmt.values())))
    m = build_masks(genes, gmt, parents)
    R = random_like(m["M1"], np.random.default_rng(0))
    assert (R.sum(1) == m["M1"].sum(1)).all() and not (R == m["M1"]).all()
    a, b = PathwayDANN(m["M1"], m["M2"]), PathwayDANN(R, m["M2"])
    assert a.effective_params() == b.effective_params()
    h1, h2 = matched_dense(len(genes), a.effective_params(), m["M2"].shape[0])
    dense = DenseMLP(len(genes), h1, h2).effective_params()
    assert abs(dense - a.effective_params()) <= len(genes) + h2 + 1  # within one hidden unit


def test_pathway_dann_learns_planted_signal_and_attributes_it():
    counts, meta, gmt, parents = smoke_expression(n_donors=30, effect=2.0)
    genes = sorted(set(expressed(counts)) & set().union(*(g for _, g in gmt.values())))
    m = build_masks(genes, gmt, parents)
    X = log_cpm(counts).loc[genes, meta["title"]].T.to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = meta["y"].to_numpy()
    # one model's attributions are unstable (the planted pathway can rank anywhere), so script 23
    # averages them over all fitted models; averaging 10 here recovers the planted signal
    total = 0
    for seed in range(10):
        torch.manual_seed(seed)
        model = train_torch(PathwayDANN(m["M1"], m["M2"]), X, y, epochs=200, seed=seed)
        c = dendrite_contributions(model, X)
        assert c.shape == (len(y), len(m["pathways"]))
        total = total + np.abs(c).mean(0)
    rank = np.argsort(np.argsort(-total))
    assert min(rank[m["pathways"].index("R-HSA-1")], rank[m["pathways"].index("R-HSA-2")]) <= 2
    with torch.no_grad():
        acc = ((model(torch.as_tensor(X, dtype=torch.float32)) > 0).numpy() == y).mean()
    assert acc > 0.9
