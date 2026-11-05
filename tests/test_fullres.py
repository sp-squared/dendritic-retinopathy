import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from dendritic_dr.fullres import (Curve, Froc, PatchSampler, candidates, find_ddr, grid_targets, lesion_labels,
                                  pack, smoke_store, split_from_path, tversky_loss, unpack)


def test_pack_unpack_roundtrip():
    m = np.random.default_rng(0).random((5, 8, 8)) > 0.7
    assert np.array_equal(unpack(pack(m)), m)
    assert unpack(np.stack([pack(m)] * 3)).shape == (3, 5, 8, 8)


def test_binned_curve_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(20000) < 0.02
    s = rng.normal(0, 1.5, len(y)) + 2.0 * y
    c = Curve()
    c.add(y[:7000], s[:7000])
    c.add(y[7000:], s[7000:])
    assert abs(c.ap() - average_precision_score(y, s)) < 1e-3
    assert abs(c.auc() - roc_auc_score(y, s)) < 1e-3
    assert abs(c.prevalence() - y.mean()) < 1e-12


def test_froc_perfect_and_counting():
    grid = np.zeros((20, 20), bool)
    grid[2:4, 2:4] = grid[10, 10] = grid[15:17, 5] = True
    lab, n = lesion_labels(grid)
    assert n == 3 and lab[1, 1] == lab[2, 2]  # one-cell tolerance
    logit = torch.full((20, 20), -10.0)
    logit[3, 3], logit[10, 10], logit[16, 5] = 5.0, 4.0, 3.0
    ys, xs, sc = candidates(logit, torch.ones(20, 20, dtype=torch.bool))
    f = Froc()
    f.add(sc, lab[ys, xs], n)
    assert f.score() == 1.0
    # one lesion found, then a false positive above the other two: sensitivity at 1/8 FP is 1/3
    f2 = Froc()
    f2.add(np.array([5.0, 4.5, 4.0, 3.0, 2.9]), np.array([1, 0, 2, 3, 3]), 3)
    assert f2.sensitivity_at(0.125) == 1 / 3 and f2.sensitivity_at(1.0) == 1.0
    f3 = Froc()
    f3.add(np.array([1.0]), np.array([0]), 2)
    assert f3.score() == 0.0


def test_tversky_rewards_recall():
    y = torch.zeros(1, 4, 8, 8)
    y[:, :, 2:4, 2:4] = 1
    v = torch.ones(1, 1, 8, 8)
    perfect = tversky_loss(torch.where(y > 0, 20.0, -20.0), y, v)
    assert perfect < 0.05
    miss = torch.where(y > 0, 20.0, -20.0)
    miss[:, :, 3, 2:4] = -20.0                        # two lesion cells missed
    over = torch.where(y > 0, 20.0, -20.0)
    over[:, :, 5, 5:7] = 20.0                         # two false-alarm cells
    assert tversky_loss(over, y, v) < tversky_loss(miss, y, v)  # beta > alpha: misses cost more
    assert tversky_loss(over, y, v, alpha=0.7, beta=0.3) > tversky_loss(miss, y, v, alpha=0.7, beta=0.3)


def test_patch_sampler_shapes_and_lesion_patches():
    store = smoke_store()
    keys = store.keys("train")
    greens, bits = zip(*(store.get(k) for k in keys))
    s = PatchSampler(list(greens), list(bits), 32, lesion_frac=1.0, seed=0)
    g, m = s.sample(16)
    assert g.shape == (16, 1, 32, 32) and g.dtype == np.uint8 and m.shape == (16, 5, 32, 32)
    assert m[:, :4].any((1, 2, 3)).all()  # every patch centred near a lesion contains one
    y, v = grid_targets(torch.from_numpy(g), torch.from_numpy(m), (9, 9))
    assert y.shape == (16, 4, 9, 9) and v.shape == (16, 1, 9, 9)


def test_find_ddr_both_layouts(tmp_path):
    img = Image.new("RGB", (8, 8))
    mask = Image.new("L", (8, 8))
    # Hugging Face mirror layout
    (tmp_path / "hf/images/train").mkdir(parents=True)
    img.save(tmp_path / "hf/images/train/a1.jpg")
    for t in ["EX", "MA"]:
        (tmp_path / f"hf/annotations/train/{t}").mkdir(parents=True)
        mask.save(tmp_path / f"hf/annotations/train/{t}/a1.tif")
    # original release layout
    (tmp_path / "orig/lesion_segmentation/valid/image").mkdir(parents=True)
    img.save(tmp_path / "orig/lesion_segmentation/valid/image/b2.jpg")
    (tmp_path / "orig/lesion_segmentation/valid/label/HE").mkdir(parents=True)
    mask.save(tmp_path / "orig/lesion_segmentation/valid/label/HE/b2.tif")
    hf, orig = find_ddr(tmp_path / "hf"), find_ddr(tmp_path / "orig")
    assert set(hf) == {"a1"} and hf["a1"][2] == "train" and set(hf["a1"][1]) == {"EX", "MA"}
    assert set(orig) == {"b2"} and orig["b2"][2] == "val" and set(orig["b2"][1]) == {"HE"}
    assert split_from_path(tmp_path / "x/b. Testing Set/y.jpg", tmp_path) == "test"
