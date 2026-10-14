from __future__ import annotations
import numpy as np
import pandas as pd

def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1).min())

def effective_number_of_bets(weights: pd.Series) -> float:
    w = weights.to_numpy(dtype=float)
    s = np.sum(w * w)
    return float(1 / s) if s > 0 else 0.0

def summarize(returns: pd.Series, weights: pd.DataFrame | None = None, periods_per_year: int = 12) -> dict[str, float]:
    r = returns.dropna()
    if r.empty: return {}
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan")
    ann_return = float((np.prod(1 + r) ** (periods_per_year / len(r))) - 1) if (1 + r).gt(0).all() else float("nan")
    result = {"annualized_return": ann_return, "annualized_volatility": vol,
              "sharpe_rf0": float(ann_return / vol) if vol > 0 else float("nan"),
              "max_drawdown": max_drawdown(r), "periods": float(len(r))}
    if weights is not None and len(weights):
        result["average_turnover_one_way"] = float(weights.diff().abs().sum(axis=1).fillna(0).mean() / 2)
        result["average_effective_bets"] = float(weights.apply(effective_number_of_bets, axis=1).mean())
        result["average_max_weight"] = float(weights.max(axis=1).mean())
    return result
