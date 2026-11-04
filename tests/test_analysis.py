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
