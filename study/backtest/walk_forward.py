"""Walk-forward engine that records estimation error against known Sigma."""
from __future__ import annotations
import numpy as np
import pandas as pd
from estimators.covariance import sample_cov, ledoit_wolf_cov, constant_corr_cov, factor_cov
from allocators.weights import equal_weight, inverse_vol, min_variance, hrp


METHODS = {
    "equal_weight": ("none", equal_weight),
    "inverse_vol": ("none", inverse_vol),
    "sample_mv": ("sample", min_variance),
    "lw_mv": ("lw", min_variance),
    "const_corr_mv": ("cc", min_variance),
    "factor_mv": ("factor", min_variance),
    "hrp": ("lw", hrp),
}


def _get_cov(method_key: str, train: pd.DataFrame) -> pd.DataFrame | None:
    if method_key == "sample":
        return sample_cov(train)
    if method_key == "lw":
        return ledoit_wolf_cov(train)
    if method_key == "cc":
        return constant_corr_cov(train)
    if method_key == "factor":
        return factor_cov(train)
    return None


def walk_forward(
    returns: pd.DataFrame,
    true_sigmas: list[np.ndarray],
    lookback: int,
    cost_bps: float = 0.0,
    methods: list[str] | None = None,
) -> dict:
    methods = methods or list(METHODS)
    n = len(returns)
    out_returns = {m: [] for m in methods}
    out_weights = {m: [] for m in methods}
    out_errors = {m: [] for m in methods}
    prev = {m: None for m in methods}

    for t in range(lookback, n):
        train = returns.iloc[t - lookback : t]
        true_S = true_sigmas[t]  # covariance that generates r_t

        for m in methods:
            key, fn = METHODS[m]
            cov = _get_cov(key, train)
            w = fn(train, cov=cov) if cov is not None else fn(train)
            w = w.reindex(returns.columns).fillna(0)
            w = w / w.sum()

            # Estimation error
            if cov is not None:
                fro = np.linalg.norm(cov.to_numpy() - true_S, ord="fro")
            else:
                fro = np.nan
            out_errors[m].append(fro)

            # Turnover & cost
            old = prev[m]
            to = 0.5 * np.abs(w - old).sum() if old is not None else 0.5 * w.abs().sum()
            prev[m] = w
            gross = float(w.to_numpy() @ returns.iloc[t].to_numpy())
            net = (1 + gross) * (1 - to * cost_bps / 1e4) - 1
            out_returns[m].append(net)
            out_weights[m].append(w)

    idx = returns.index[lookback:]
    return {
        "returns": pd.DataFrame(out_returns, index=idx),
        "weights": {m: pd.DataFrame(out_weights[m], index=idx) for m in methods},
        "errors": pd.DataFrame(out_errors, index=idx),
    }
