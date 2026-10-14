from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov().reindex(index=returns.columns, columns=returns.columns)

def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    est = LedoitWolf().fit(returns.to_numpy())
    return pd.DataFrame(est.covariance_, index=returns.columns, columns=returns.columns)

def diagonal_shrink_cov(returns: pd.DataFrame, intensity: float = 0.25) -> pd.DataFrame:
    """Transparent constant-intensity shrinkage toward the diagonal; intensity in [0,1]."""
    if not 0 <= intensity <= 1: raise ValueError("intensity must be in [0, 1]")
    s = sample_cov(returns)
    target = pd.DataFrame(np.diag(np.diag(s.to_numpy())), index=s.index, columns=s.columns)
    return (1 - intensity) * s + intensity * target

def regularize_cov(cov: pd.DataFrame, ridge: float = 1e-8) -> pd.DataFrame:
    arr = cov.to_numpy(dtype=float)
    arr = (arr + arr.T) / 2
    floor = max(float(np.trace(arr) / max(len(arr), 1)) * ridge, 1e-12)
    arr += np.eye(len(arr)) * floor
    return pd.DataFrame(arr, index=cov.index, columns=cov.columns)
