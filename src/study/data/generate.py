"""Controlled returns with a known unconditional covariance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from study.protocol import (
    BLOCK_COUNT,
    BLOCK_INTER,
    BLOCK_INTRA,
    CORR_SPIKE,
    MIX_PROB,
    MIX_SCALE,
    MIX_VAR_FACTOR,
    RHO,
    SHOCK_LEN,
    T4_NU,
    VOL_JUMP,
)


def _equicorr(n: int, rho: float) -> np.ndarray:
    r = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(r, 1.0)
    return r


def _block_corr(n: int) -> np.ndarray:
    r = np.full((n, n), BLOCK_INTER, dtype=float)
    size = n // BLOCK_COUNT
    for b in range(BLOCK_COUNT):
        sl = slice(b * size, (b + 1) * size if b < BLOCK_COUNT - 1 else n)
        r[sl, sl] = BLOCK_INTRA
    np.fill_diagonal(r, 1.0)
    return r


def _random_corr(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(n, n))
    m = a @ a.T
    d = np.sqrt(np.diag(m))
    return m / np.outer(d, d)


def _pd(sigma: np.ndarray) -> np.ndarray:
    sigma = (sigma + sigma.T) / 2
    eig, vec = np.linalg.eigh(sigma)
    return vec @ np.diag(np.maximum(eig, 1e-12)) @ vec.T


def make_synthetic_panel(
    n_assets: int,
    n_periods: int,
    corr_type: str,
    seed: int,
    tail: str = "gaussian",
    shock: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Monthly panel. `true_sigmas[t]` is the unconditional covariance of r_t.

    The mixture multiplier is latent. Error is measured against the
    unconditional covariance, not against the latent draw.
    `shock` is None, 'vol', or 'corr'. The two shocks are never combined.
    """
    if tail not in ("gaussian", "t4", "mix"):
        raise ValueError(tail)
    if shock not in (None, "vol", "corr"):
        raise ValueError(shock)
    rng = np.random.default_rng(seed)
    if corr_type in RHO:
        base_r = _equicorr(n_assets, RHO[corr_type])
    elif corr_type == "block":
        base_r = _block_corr(n_assets)
    elif corr_type == "random":
        base_r = _random_corr(n_assets, rng)
    else:
        raise ValueError(corr_type)

    annual = rng.uniform(0.08, 0.25, size=n_assets)
    vols = annual / np.sqrt(12.0)
    dates = pd.date_range("2000-01-31", periods=n_periods, freq="ME")
    shock_start = n_periods // 2
    shock_end = min(n_periods, shock_start + SHOCK_LEN)
    rows = np.zeros((n_periods, n_assets))
    truth = []
    var_factor = MIX_VAR_FACTOR if tail == "mix" else 1.0

    for t in range(n_periods):
        scale = VOL_JUMP if shock == "vol" and shock_start <= t < shock_end else 1.0
        corr = _equicorr(n_assets, CORR_SPIKE) if shock == "corr" and shock_start <= t < shock_end else base_r
        sigma = _pd(np.outer(vols * scale, vols * scale) * corr)
        truth.append(var_factor * sigma)
        if tail == "t4":
            z = rng.standard_t(T4_NU, size=n_assets) / np.sqrt(T4_NU / (T4_NU - 2))
        else:
            z = rng.standard_normal(n_assets)
        draw = np.linalg.cholesky(sigma) @ z
        if tail == "mix" and rng.random() < MIX_PROB:
            draw = draw * MIX_SCALE
        rows[t] = draw

    frame = pd.DataFrame(rows, index=dates, columns=[f"A{i}" for i in range(n_assets)])
    meta = {
        "true_sigmas": truth,
        "shock_start": shock_start,
        "shock_end": shock_end,
        "corr_type": corr_type,
        "tail": tail,
        "shock": shock,
        "n_assets": n_assets,
    }
    return frame, meta
