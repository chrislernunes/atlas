from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from .estimators import regularize_cov

def _normalise(w: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    s = w.sum()
    return w / s if s > 0 else np.repeat(1 / len(w), len(w))

def equal_weight(returns: pd.DataFrame, **_) -> pd.Series:
    return pd.Series(1 / returns.shape[1], index=returns.columns)

def inverse_vol(returns: pd.DataFrame, **_) -> pd.Series:
    vol = returns.std(ddof=1).replace(0, np.nan).fillna(returns.std(ddof=1).median()).to_numpy()
    return pd.Series(_normalise(1 / np.maximum(vol, 1e-8)), index=returns.columns)

def min_variance(returns: pd.DataFrame, cov: pd.DataFrame | None = None,
                 max_weight: float = 0.35, **_) -> pd.Series:
    """Long-only, fully invested min-variance portfolio with per-asset caps."""
    names = list(returns.columns)
    c = regularize_cov(cov if cov is not None else returns.cov())
    mat = c.loc[names, names].to_numpy()
    n = len(names)
    cap = max(max_weight, 1 / n)
    x0 = np.repeat(1 / n, n)
    res = minimize(lambda w: float(w @ mat @ w), x0, method="SLSQP",
                   bounds=[(0.0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        # A feasible equal-weight fallback is preferable to silently returning invalid weights.
        return pd.Series(x0, index=names)
    return pd.Series(_normalise(res.x), index=names)

def hrp(returns: pd.DataFrame, cov: pd.DataFrame | None = None, **_) -> pd.Series:
    """Hierarchical Risk Parity: correlation clustering + recursive bisection."""
    names = list(returns.columns)
    if len(names) == 1: return pd.Series([1.0], index=names)
    c = regularize_cov(cov if cov is not None else returns.cov())
    c = c.loc[names, names]
    sd = np.sqrt(np.maximum(np.diag(c.to_numpy()), 1e-16))
    corr = c.to_numpy() / np.outer(sd, sd)
    corr = np.clip((corr + corr.T) / 2, -1, 1)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.maximum((1 - corr) / 2, 0))
    np.fill_diagonal(dist, 0.0)
    tree = linkage(squareform(dist, checks=False), method="single", optimal_ordering=True)
    ordered = leaves_list(tree).tolist()
    weights = pd.Series(1.0, index=ordered)
    def cluster_var(cluster: list[int]) -> float:
        sub = c.to_numpy()[np.ix_(cluster, cluster)]
        inv_diag = 1 / np.maximum(np.diag(sub), 1e-16)
        ivp = inv_diag / inv_diag.sum()
        return float(ivp @ sub @ ivp)
    clusters = [ordered]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1: continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / max(vl + vr, 1e-16)
            weights.loc[left] *= alpha
            weights.loc[right] *= (1 - alpha)
            next_clusters.extend([left, right])
        clusters = next_clusters
    out = np.zeros(len(names))
    for pos, original_idx in enumerate(ordered): out[original_idx] = weights.loc[original_idx]
    return pd.Series(_normalise(out), index=names)
