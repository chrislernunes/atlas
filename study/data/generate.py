"""Controlled synthetic data generator with known true covariance."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _equicorr(n: int, rho: float) -> np.ndarray:
    R = np.full((n, n), rho)
    np.fill_diagonal(R, 1.0)
    return R


def _block_corr(n: int, n_blocks: int = 2, intra: float = 0.60, inter: float = 0.05) -> np.ndarray:
    R = np.full((n, n), inter)
    size = n // n_blocks
    for b in range(n_blocks):
        sl = slice(b * size, (b + 1) * size if b < n_blocks - 1 else n)
        R[sl, sl] = intra
    np.fill_diagonal(R, 1.0)
    return R


def make_synthetic_panel(
    n_assets: int,
    n_periods: int,
    corr_type: str,
    lookback_ref: int,
    seed: int,
    shock_vol: bool = False,
    shock_corr: bool = False,
    student_t: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns monthly returns and a dict containing the true covariance
    at every date (for estimation-error measurement).
    """
    rng = np.random.default_rng(seed)
    if corr_type == "low":
        R0 = _equicorr(n_assets, 0.10)
    elif corr_type == "moderate":
        R0 = _equicorr(n_assets, 0.30)
    elif corr_type == "block":
        R0 = _block_corr(n_assets)
    elif corr_type == "high":
        R0 = _equicorr(n_assets, 0.60)
    else:
        raise ValueError(corr_type)

    vols = rng.uniform(0.08, 0.25, size=n_assets) / np.sqrt(12)  # monthly
    Sigma0 = np.outer(vols, vols) * R0

    dates = pd.date_range("2000-01-31", periods=n_periods, freq="ME")
    returns = np.zeros((n_periods, n_assets))
    true_sigmas = []

    shock_start = n_periods // 2
    shock_end = shock_start + 24

    for t in range(n_periods):
        R, scale = R0.copy(), 1.0
        if shock_vol and shock_start <= t < shock_end:
            scale = 2.0
        if shock_corr and shock_start <= t < shock_end:
            R = _equicorr(n_assets, 0.70)
        Sigma = np.outer(vols * scale, vols * scale) * R
        # Ensure PD
        eig, vec = np.linalg.eigh(Sigma)
        Sigma = vec @ np.diag(np.maximum(eig, 1e-8)) @ vec.T
        true_sigmas.append(Sigma)

        if student_t:
            z = rng.standard_t(7, size=n_assets)
            z = z / np.sqrt(7 / 5)  # rough unit variance
        else:
            z = rng.standard_normal(n_assets)
        L = np.linalg.cholesky(Sigma)
        returns[t] = L @ z

    ret_df = pd.DataFrame(returns, index=dates, columns=[f"A{i}" for i in range(n_assets)])
    meta = {
        "true_sigmas": true_sigmas,
        "vols": vols,
        "shock_start": shock_start,
        "shock_end": shock_end,
        "corr_type": corr_type,
        "n_assets": n_assets,
    }
    return ret_df, meta
