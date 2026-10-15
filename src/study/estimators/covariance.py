"""Covariance estimators. No free shrinkage grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from study.protocol import FACTORS_FIXED


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    c = returns.cov()
    return pd.DataFrame((c.to_numpy() + c.to_numpy().T) / 2, index=c.index, columns=c.columns)


def ledoit_wolf_cov(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    est = LedoitWolf().fit(returns.to_numpy())
    c = pd.DataFrame(est.covariance_, index=returns.columns, columns=returns.columns)
    return c, float(est.shrinkage_)


def constant_corr_cov(returns: pd.DataFrame) -> pd.DataFrame:
    s = sample_cov(returns).to_numpy()
    std = np.sqrt(np.clip(np.diag(s), 1e-18, None))
    corr = s / np.outer(std, std)
    n = len(std)
    avg = (corr.sum() - n) / (n * (n - 1))
    r = np.full((n, n), avg)
    np.fill_diagonal(r, 1.0)
    out = np.outer(std, std) * r
    return pd.DataFrame((out + out.T) / 2, index=returns.columns, columns=returns.columns)


def factor_cov(returns: pd.DataFrame, n_factors: int = FACTORS_FIXED) -> pd.DataFrame:
    """Spectral truncation of the sample covariance. n_factors is fixed at 1."""
    c = np.cov(returns.to_numpy(), rowvar=False, ddof=1)
    c = (c + c.T) / 2
    evals, evecs = np.linalg.eigh(c)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    k = min(n_factors, len(evals))
    loadings = evecs[:, :k] * np.sqrt(np.clip(evals[:k], 0, None))
    resid = np.clip(np.diag(c) - np.sum(loadings ** 2, axis=1), 0, None)
    sigma = loadings @ loadings.T + np.diag(resid)
    sigma = (sigma + sigma.T) / 2
    return pd.DataFrame(sigma, index=returns.columns, columns=returns.columns)
