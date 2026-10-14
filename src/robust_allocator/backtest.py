from __future__ import annotations
from collections.abc import Callable
import numpy as np
import pandas as pd
from .estimators import sample_cov, ledoit_wolf_cov
from .allocators import equal_weight, inverse_vol, min_variance, hrp

Allocator = Callable[..., pd.Series]
ALLOCATORS: dict[str, Allocator] = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_vol,
    "sample_min_variance": min_variance,
    "lw_min_variance": min_variance,
    "hrp": hrp,
}

def walk_forward_backtest(returns: pd.DataFrame, lookback: int = 60, rebalance_every: int = 1,
                          cost_bps: float = 5.0, max_weight: float = 0.35,
                          methods: tuple[str, ...] = tuple(ALLOCATORS)) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Monthly/periodic rebalance; weights estimated at t-1 are applied to return at t.

    Cost convention: one-way turnover = 0.5 * sum(abs(weight change)); cost deducted
    at rebalance. This is a stylized proportional-cost model, not market-impact modeling.
    """
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("returns index must be unique and sorted ascending")
    if returns.isna().any().any(): raise ValueError("returns contain NaNs")
    if lookback < 12: raise ValueError("lookback must be at least 12 periods")
    if cost_bps < 0: raise ValueError("cost_bps cannot be negative")
    n = len(returns)
    strategy_returns: dict[str, list[tuple[pd.Timestamp, float]]] = {m: [] for m in methods}
    weights_log: dict[str, list[tuple[pd.Timestamp, pd.Series]]] = {m: [] for m in methods}
    prev = {m: None for m in methods}
    for t in range(lookback, n):
        if (t - lookback) % rebalance_every == 0 or t == lookback:
            train = returns.iloc[t-lookback:t].copy()  # excludes current period
            for method in methods:
                if method == "sample_min_variance": cov = sample_cov(train)
                elif method == "lw_min_variance": cov = ledoit_wolf_cov(train)
                elif method == "hrp": cov = ledoit_wolf_cov(train)
                else: cov = None
                fn = ALLOCATORS[method]
                kw = {"max_weight": max_weight}
                if method in ("sample_min_variance", "lw_min_variance", "hrp"): kw["cov"] = cov
                w = fn(train, **kw).reindex(returns.columns).fillna(0)
                w = w / w.sum()
                weights_log[method].append((returns.index[t], w.copy()))
                old = prev[method]
                turnover = float(w.abs().sum()) if old is None else float((w - old).abs().sum() / 2)
                prev[method] = w
                cost = turnover * cost_bps / 10000.0
                gross = float(w.to_numpy() @ returns.iloc[t].to_numpy())
                net = (1 + gross) * (1 - cost) - 1
                strategy_returns[method].append((returns.index[t], net))
        else:
            for method in methods:
                w = prev[method]
                gross = float(w.to_numpy() @ returns.iloc[t].to_numpy())
                strategy_returns[method].append((returns.index[t], gross))
    ret_df = pd.DataFrame({m: pd.Series(dict(v)).sort_index() for m, v in strategy_returns.items()})
    weights_df: dict[str, pd.DataFrame] = {}
    for method, items in weights_log.items():
        weights_df[method] = pd.DataFrame({dt: w for dt, w in items}).T.sort_index()
    return ret_df, weights_df
