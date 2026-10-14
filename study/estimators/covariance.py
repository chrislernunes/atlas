"""Covariance estimators."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov()


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    est = LedoitWolf().fit(returns.to_numpy())
    return pd.DataFrame(est.covariance_, index=returns.columns, columns=returns.columns)


def constant_corr_cov(returns: pd.DataFrame) -> pd.DataFrame:
    s = returns.cov()
    std = np.sqrt(np.diag(s))
    corr = s.to_numpy() / np.outer(std, std)
    avg = (corr.sum() - len(corr)) / (len(corr) * (len(corr) - 1))
    R = np.full_like(corr, avg)
    np.fill_diagonal(R, 1.0)
    return pd.DataFrame(np.outer(std, std) * R, index=s.index, columns=s.columns)


def factor_cov(returns: pd.DataFrame, n_factors: int = 1) -> pd.DataFrame:
    """Single-factor covariance (first PC + residual variances)."""
    X = returns.to_numpy()
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    loadings = Vt[:n_factors].T * S[:n_factors]
    resid_var = np.var(Xc - U[:, :n_factors] @ np.diag(S[:n_factors]) @ Vt[:n_factors], axis=0, ddof=1)
    Sigma = loadings @ loadings.T + np.diag(resid_var)
    return pd.DataFrame(Sigma, index=returns.columns, columns=returns.columns)
