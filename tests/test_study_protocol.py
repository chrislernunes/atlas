import numpy as np
import pandas as pd

from study.allocators.weights import cap_for, hrp, min_variance, project_to_constraints
from study.backtest.walk_forward import walk_forward
from study.data.generate import make_synthetic_panel
from study.estimators.covariance import factor_cov, ledoit_wolf_cov, sample_cov
from study.metrics.evaluation import decide_h1, decide_h2, h3_reversal, theta_from_seed_deltas
from study.protocol import NT_MIN, STRATUM_CELLS, WEIGHT_CAP
from study.regime.detector import detect_stress


def test_stratum_matches_the_locked_ratio():
    expected = tuple(
        (n, t, c)
        for n in (8, 20, 50)
        for t in (36, 60, 120)
        for c in ("low", "moderate", "block", "high")
        if n / t >= NT_MIN
    )
    assert STRATUM_CELLS == expected
    assert len(STRATUM_CELLS) == 16


def test_projection_respects_cap_and_budget():
    raw = np.array([0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = project_to_constraints(raw, 8)
    assert out is not None
    assert abs(out.sum() - 1) < 1e-8
    assert out.min() >= -1e-12
    assert out.max() <= cap_for(8) + 1e-8


def test_cap_rule_is_feasible_and_nonfinite_falls_back():
    for n in (8, 20, 50):
        assert cap_for(n) * n >= 1.0 - 1e-12
    assert project_to_constraints(np.array([np.nan, 0.2, 0.2]), 3) is None


def test_allocators_share_the_cap():
    returns, _ = make_synthetic_panel(8, 40, "block", 1)
    cov = sample_cov(returns)
    for series, ok in (min_variance(cov), hrp(cov, "single"), hrp(cov, "average")):
        w = series.to_numpy()
        assert ok
        assert abs(w.sum() - 1) < 1e-6
        assert w.min() >= -1e-8
        assert w.max() <= WEIGHT_CAP + 1e-8


def test_factor_covariance_is_the_same_order_as_sample():
    returns, meta = make_synthetic_panel(12, 80, "moderate", 2)
    sample = sample_cov(returns).to_numpy()
    factor = factor_cov(returns).to_numpy()
    true = meta["true_sigmas"][-1]
    assert np.linalg.norm(factor - true) < 5 * np.linalg.norm(sample - true)


def test_no_lookahead():
    returns, meta = make_synthetic_panel(6, 48, "low", 3)
    base = walk_forward(returns, meta["true_sigmas"], 24)
    poisoned = returns.copy()
    poisoned.iloc[-1] = poisoned.iloc[-1] + 5
    alt = walk_forward(poisoned, meta["true_sigmas"], 24)
    date = returns.index[-1]
    b = base[base.date == date].set_index("method")["ret_gross"]
    # The last return changed, so ret_gross may change. The previous date must not.
    prev = returns.index[-2]
    left = base[base.date == prev].set_index("method")[["frobenius", "eff_bets", "max_weight"]]
    right = alt[alt.date == prev].set_index("method")[["frobenius", "eff_bets", "max_weight"]]
    pd.testing.assert_frame_equal(left, right)
    assert not b.empty


def test_detector_ignores_the_future():
    returns, _ = make_synthetic_panel(8, 80, "moderate", 4, shock="vol")
    train = returns.iloc[:60]
    assert detect_stress(train) == detect_stress(train.copy())
    longer = returns.iloc[:70]
    # Appending future rows can change a later decision, not the function of the original train.
    assert detect_stress(train) in ("calm", "stress")
    assert detect_stress(longer) in ("calm", "stress")


def test_decision_refuses_a_short_sample():
    assert decide_h1(1.0, 0.5, 1.0, 0.99, n_seeds=3, grid_complete=False) == "NOT_A_DECISION_SAMPLE"
    assert decide_h1(0.2, 0.05, 1.0, 0.5, n_seeds=30, grid_complete=True) == "INCONCLUSIVE_UNDERPOWERED"
    assert decide_h1(0.2, 0.05, 1.0, 0.9, n_seeds=30, grid_complete=True) == "HOLDS"
    assert decide_h1(0.05, 0.01, 1.0, 0.9, n_seeds=30, grid_complete=True) == "DETECTED_BELOW_MDE"
    assert decide_h1(-0.1, -0.2, 1.0, 0.9, n_seeds=30, grid_complete=True) == "NOT_SUPPORTED"
    assert decide_h2(1.5, 0.2, 30, True) == "HOLDS"
    assert h3_reversal(0.2, -0.2)
    assert not h3_reversal(0.01, -0.2)


def test_theta_is_the_weighted_mean_of_medians():
    deltas = {(20, 36, "low"): np.array([1.0, 3.0]), (50, 36, "low"): np.array([10.0, 10.0])}
    weights = {k: k[0] / k[1] for k in deltas}
    # medians 2 and 10, weights 20/36 and 50/36
    w1, w2 = 20 / 36, 50 / 36
    expected = (2 * w1 + 10 * w2) / (w1 + w2)
    assert abs(theta_from_seed_deltas(deltas, weights) - expected) < 1e-12
