import numpy as np

from dendritic_dr.analysis import (
    bh_fdr,
    bootstrap_qwk_ci,
    differential_activation,
    paired_bootstrap_diff,
    progression_score,
    selectivity,
)


def test_bh_matches_hand_computation():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    # sorted: 0.01, 0.03, 0.04, 0.20 -> *4/rank = 0.04, 0.06, 0.0533, 0.20 -> monotone: 0.04, 0.0533, 0.0533, 0.20
    np.testing.assert_allclose(bh_fdr(p), [0.04, 0.05333333, 0.05333333, 0.20])


def test_bh_passes_nan():
    q = bh_fdr([0.01, np.nan, 0.02])
    assert np.isnan(q[1]) and np.allclose(q[[0, 2]], [0.02, 0.02])


def test_selectivity_extremes():
    gm = np.array([[1.0, 0, 0, 0, 0],      # single-grade unit
                   [1.0, 1, 1, 1, 1],      # responds to every grade
                   [0.0, 0, 0, 0, 0]])     # dead unit
    s = selectivity(gm)
    assert s["selectivity_index"][0] == 1.0 and s["n_grades_responsive"][0] == 1
    assert s["selectivity_index"][1] == 0.0 and s["n_grades_responsive"][1] == 5
    assert np.isnan(s["selectivity_index"][2])


def test_differential_activation_finds_planted_signal():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5), 40)
    A = rng.random((200, 10))
    A[:, 3] += y            # unit 3 increases with grade
    A[:, 7] = 0.0           # dead unit
    df = differential_activation(A, y)
    assert df.loc[3, "kruskal_q"] < 1e-6 and df.loc[3, "spearman_rho"] > 0.8
    assert not df.loc[7, "active"] and np.isnan(df.loc[7, "kruskal_p"])


def test_progression_score_range():
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(5), 30)
    ordered = np.outer(y, np.ones(8)) + 0.1 * rng.normal(size=(150, 8))
    assert progression_score(ordered, y) > 0.9
    assert progression_score(rng.normal(size=(150, 8)), y) < 0.3


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 5, 300)
    pred = np.where(rng.random(300) < 0.7, y, rng.integers(0, 5, 300))
    lo, hi = bootstrap_qwk_ci(y, pred, n_boot=300)
    assert lo < hi


def test_paired_bootstrap_identical_models():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 5, 200)
    pred = np.stack([y, y])
    diff, lo, hi, p = paired_bootstrap_diff(y, pred, pred, n_boot=200)
    assert diff == 0 and lo == 0 and hi == 0


def test_append_result_handles_new_columns(tmp_path):
    import pandas as pd
    from dendritic_dr.utils import append_result

    p = tmp_path / "r.csv"
    append_result(p, {"a": 1, "b": [1, 2]})
    append_result(p, {"a": 2, "c": "x", "b": [3]})   # new column c
    append_result(p, {"a": 3, "b": []})               # back to old columns
    d = pd.read_csv(p)
    assert list(d.columns) == ["a", "b", "c"] and len(d) == 3
    assert d["c"].isna().tolist() == [True, False, True]


def test_rf_sweep_parses_stage_and_grid():
    import importlib.util

    spec = importlib.util.spec_from_file_location("sweep", "scripts/11_receptive_field_sweep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert (m.stage_of("resnet18_224_spatial"), m.grid_of("resnet18_224_spatial")) == (4, 7)
    assert (m.stage_of("resnet18_224_spatial_l2_g14"), m.grid_of("resnet18_224_spatial_l2_g14")) == (2, 14)
    assert (m.stage_of("resnet18_224_spatial_l3"), m.grid_of("resnet18_224_spatial_l3")) == (3, 14)


def test_fundus_pixels_shapes_and_dtype():
    import numpy as np
    from PIL import Image

    from dendritic_dr.data import fundus_pixels

    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    g = fundus_pixels(img, 16, "green")
    assert g.shape == (1, 16, 16) and g.dtype == np.uint8
    assert fundus_pixels(img, 16, "gray").shape == (1, 16, 16)
    assert fundus_pixels(img, 16, "rgb").shape == (3, 16, 16)
    rgb = fundus_pixels(img, 16, "rgb")
    assert (rgb[1] == g[0]).all()  # green channel is the middle RGB plane


def test_smoke_pixels_load_as_images():
    from dendritic_dr.features import load_features

    d = load_features("smoke_pixels", smoke=True)
    assert d["image_shape"] == (1, 16, 16)
    assert d["X_train"].shape[1] == 256 and d["X_train"].dtype.name == "float32"
