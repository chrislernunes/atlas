"""Evaluation metrics and block bootstrap."""
from __future__ import annotations
import numpy as np
import pandas as pd


def frobenius(error_series: pd.Series) -> float:
    return float(error_series.mean())


def portfolio_stats(r: pd.Series, periods: int = 12) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    vol = r.std(ddof=1) * np.sqrt(periods)
    ann = (1 + r).prod() ** (periods / len(r)) - 1
    wealth = (1 + r).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    return {
        "ann_return": float(ann),
        "ann_vol": float(vol),
        "sharpe": float(ann / vol) if vol > 0 else np.nan,
        "max_dd": float(dd),
    }


def effective_bets(w: pd.DataFrame) -> float:
    return float((1 / (w ** 2).sum(axis=1)).mean())


def block_bootstrap_sharpe(r: pd.Series, n_boot: int = 200, block: int = 6, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    r = r.dropna().to_numpy()
    n = len(r)
    if n < block * 2:
        return np.nan, np.nan
    sharpes = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, size=n // block + 1)
        sample = np.concatenate([r[s : s + block] for s in starts])[:n]
        vol = sample.std(ddof=1) * np.sqrt(12)
        ann = (1 + sample).prod() ** (12 / len(sample)) - 1
        sharpes.append(ann / vol if vol > 0 else np.nan)
    return float(np.nanpercentile(sharpes, 5)), float(np.nanpercentile(sharpes, 95))
