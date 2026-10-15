"""Long-only, fully invested weights with one shared constraint set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from study.protocol import (
    FALLBACK,
    LINKAGE_DEFAULT,
    SOLVER,
    SOLVER_FTOL,
    SOLVER_MAXITER,
    WEIGHT_CAP,
)


def cap_for(n: int) -> float:
    return max(WEIGHT_CAP, 1.0 / n)


def project_to_constraints(raw: np.ndarray, n: int | None = None) -> np.ndarray | None:
    """Euclidean projection onto sum(w)=1, 0 <= w_i <= cap.

    Returns None when the box cannot contain the simplex.
    """
    w = np.asarray(raw, dtype=float)
    n = int(n if n is not None else w.size)
    cap = cap_for(n)
    if cap * n < 1.0 - 1e-10 or not np.isfinite(w).all():
        return None
    lo = float(np.min(w) - cap)
    hi = float(np.max(w))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.clip(w - mid, 0.0, cap).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    out = np.clip(w - hi, 0.0, cap)
    if not np.isfinite(out).all() or abs(float(out.sum()) - 1.0) > 1e-6:
        return None
    return out / out.sum()


def equal_weight(n: int, names: pd.Index) -> pd.Series:
    return pd.Series(np.full(n, 1.0 / n), index=names)


def inverse_vol(returns: pd.DataFrame) -> tuple[pd.Series, bool]:
    vol = returns.std(ddof=1).replace(0, np.nan)
    vol = vol.fillna(float(vol.median()) if np.isfinite(vol.median()) else 1.0).to_numpy()
    raw = 1.0 / np.clip(vol, 1e-8, None)
    proj = project_to_constraints(raw / raw.sum(), len(raw))
    if proj is None:
        return equal_weight(returns.shape[1], returns.columns), False
    return pd.Series(proj, index=returns.columns), True


def min_variance(cov: pd.DataFrame, x0: np.ndarray | None = None) -> tuple[pd.Series, bool]:
    names = cov.index
    n = len(names)
    mat = cov.to_numpy(dtype=float)
    mat = (mat + mat.T) / 2
    cap = cap_for(n)
    start = np.full(n, 1.0 / n) if x0 is None else np.asarray(x0, dtype=float)
    start_proj = project_to_constraints(start, n)
    start = start_proj if start_proj is not None else np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        return float(w @ mat @ w)

    res = minimize(
        objective,
        start,
        method=SOLVER,
        bounds=[(0.0, cap)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": SOLVER_MAXITER, "ftol": SOLVER_FTOL, "disp": False},
    )
    if not res.success or not np.isfinite(res.x).all():
        return equal_weight(n, names), False
    proj = project_to_constraints(res.x, n)
    if proj is None:
        return equal_weight(n, names), False
    return pd.Series(proj, index=names), True


def hrp(cov: pd.DataFrame, method: str = LINKAGE_DEFAULT) -> tuple[pd.Series, bool]:
    if method not in ("single", "average"):
        raise ValueError(method)
    names = list(cov.index)
    n = len(names)
    c = cov.to_numpy(dtype=float)
    c = (c + c.T) / 2
    sd = np.sqrt(np.clip(np.diag(c), 1e-18, None))
    corr = c / np.outer(sd, sd)
    corr = np.clip((corr + corr.T) / 2, -0.999, 0.999)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0, None))
    np.fill_diagonal(dist, 0.0)
    tree = linkage(squareform(dist, checks=False), method=method)
    ordered = leaves_list(tree).tolist()
    weights = np.ones(n)

    def cluster_var(idx: list[int]) -> float:
        sub = c[np.ix_(idx, idx)]
        iv = 1.0 / np.clip(np.diag(sub), 1e-18, None)
        iv = iv / iv.sum()
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
            alpha = 1.0 - vl / (vl + vr + 1e-16)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            nxt.extend([left, right])
        clusters = nxt
    proj = project_to_constraints(weights, n)
    if proj is None:
        return equal_weight(n, cov.index), False
    return pd.Series(proj, index=cov.index), True


FALLBACK_NAME = FALLBACK
