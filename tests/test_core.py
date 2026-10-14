import numpy as np
import pandas as pd
import pytest
from robust_allocator.data import make_synthetic_returns, load_returns_csv
from robust_allocator.estimators import ledoit_wolf_cov, diagonal_shrink_cov
from robust_allocator.allocators import equal_weight, inverse_vol, min_variance, hrp
from robust_allocator.backtest import walk_forward_backtest

def test_allocators_are_long_only_and_fully_invested():
    r, _ = make_synthetic_returns(100, 3)
    cov = ledoit_wolf_cov(r)
    for w in [equal_weight(r), inverse_vol(r), min_variance(r, cov=cov), hrp(r, cov=cov)]:
        assert np.isfinite(w).all()
        assert (w >= -1e-10).all()
        assert abs(w.sum() - 1) < 1e-6

def test_covariance_shrinkage_psd():
    r, _ = make_synthetic_returns(80, 4)
    c = diagonal_shrink_cov(r, 0.3).to_numpy()
    assert np.linalg.eigvalsh(c).min() > -1e-10

def test_walk_forward_outputs_are_aligned_and_finite():
    r, _ = make_synthetic_returns(110, 5)
    out, weights = walk_forward_backtest(r, lookback=60, cost_bps=5,
        methods=("equal_weight", "hrp", "lw_min_variance"))
    assert len(out) == 50
    assert np.isfinite(out.to_numpy()).all()
    for w in weights.values():
        assert np.allclose(w.sum(axis=1), 1)

def test_input_validation():
    r, _ = make_synthetic_returns(80, 6)
    with pytest.raises(ValueError): walk_forward_backtest(r, lookback=10)
