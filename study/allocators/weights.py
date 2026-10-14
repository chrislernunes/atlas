"""Allocation methods (long-only, fully invested)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize


def _normalise(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1 / len(w))


def equal_weight(returns: pd.DataFrame, **_) -> pd.Series:
    n = returns.shape[1]
    return pd.Series(np.full(n, 1 / n), index=returns.columns)


def inverse_vol(returns: pd.DataFrame, **_) -> pd.Series:
    vol = returns.std(ddof=1).replace(0, np.nan).fillna(returns.std().median())
    return pd.Series(_normalise(1 / vol.to_numpy()), index=returns.columns)


def min_variance(returns: pd.DataFrame, cov: pd.DataFrame, max_weight: float = 0.40, **_) -> pd.Series:
    names = list(returns.columns)
    mat = cov.loc[names, names].to_numpy()
    n = len(names)
    cap = max(max_weight, 1 / n)
    x0 = np.full(n, 1 / n)
    res = minimize(
        lambda w: w @ mat @ w,
        x0,
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"maxiter": 400, "ftol": 1e-12},
    )
    w = res.x if res.success else x0
    return pd.Series(_normalise(w), index=names)


def hrp(returns: pd.DataFrame, cov: pd.DataFrame, **_) -> pd.Series:
    names = list(returns.columns)
    c = cov.loc[names, names].to_numpy()
    sd = np.sqrt(np.maximum(np.diag(c), 1e-12))
    corr = c / np.outer(sd, sd)
    corr = np.clip((corr + corr.T) / 2, -0.999, 0.999)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.maximum((1 - corr) / 2, 0))
    tree = linkage(squareform(dist, checks=False), method="single")
    ordered = leaves_list(tree).tolist()
    weights = np.ones(len(names))

    def cluster_var(idx):
        sub = c[np.ix_(idx, idx)]
        iv = 1 / np.maximum(np.diag(sub), 1e-12)
        iv /= iv.sum()
        return float(iv @ sub @ iv)

    clusters = [ordered]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            mid = len(cl) // 2
            left, right = cl[:mid], cl[mid:]
            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / (vl + vr + 1e-16)
            weights[left] *= alpha
            weights[right] *= 1 - alpha
            nxt.extend([left, right])
        clusters = nxt
    return pd.Series(_normalise(weights), index=names)
