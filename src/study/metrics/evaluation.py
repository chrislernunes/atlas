"""Estimands. H1 is a weighted mean of seed-medians, not a cell-count vote."""
from __future__ import annotations

import numpy as np
import pandas as pd

from study.protocol import (
    H3_SHARPE_FLOOR,
    MDE_EFFECTIVE_BETS,
    MDE_RELATIVE_FROBENIUS,
    POWER_MIN,
    SEEDS_PRIMARY,
    STRATUM_CELLS,
)


def frobenius(hat: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(hat - true, ord="fro"))


def eigenvalue_l1(hat: np.ndarray, true: np.ndarray) -> float:
    eh = np.sort(np.linalg.eigvalsh((hat + hat.T) / 2))[::-1]
    et = np.sort(np.linalg.eigvalsh((true + true.T) / 2))[::-1]
    return float(np.sum(np.abs(eh - et)))


def risk_forecast_abs(weights: np.ndarray, hat: np.ndarray, true: np.ndarray) -> float:
    pred = float(weights @ hat @ weights)
    tgt = float(weights @ true @ weights)
    return abs(pred - tgt)


def effective_bets(weights: np.ndarray) -> float:
    s = float(np.dot(weights, weights))
    return 1.0 / s if s > 0 else 0.0


def sharpe(returns: np.ndarray, periods: int = 12) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    vol = float(r.std(ddof=1) * np.sqrt(periods))
    ann = float(np.prod(1.0 + r) ** (periods / r.size) - 1.0)
    return ann / vol if vol > 0 else float("nan")


def max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def stratum_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Gaussian primary-correlation rows inside the locked N/T list."""
    keys = pd.MultiIndex.from_tuples(STRATUM_CELLS, names=["n_assets", "lookback", "corr_type"])
    out = panel[(panel["innovation"] == "gaussian") & (panel["shock"].isna() | (panel["shock"] == ""))]
    idx = pd.MultiIndex.from_frame(out[["n_assets", "lookback", "corr_type"]])
    return out.loc[idx.isin(keys)].copy()


def theta_from_seed_deltas(deltas: dict[tuple, np.ndarray], weights: dict[tuple, float]) -> float:
    meds = []
    wts = []
    for key, values in deltas.items():
        meds.append(float(np.median(values)))
        wts.append(weights[key])
    return float(np.average(meds, weights=wts))


def seed_bootstrap_theta(
    deltas: dict[tuple, np.ndarray],
    weights: dict[tuple, float],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    keys = list(deltas)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        meds = []
        wts = []
        for key in keys:
            values = deltas[key]
            draw = values[rng.integers(0, values.size, size=values.size)]
            meds.append(float(np.median(draw)))
            wts.append(weights[key])
        stats[b] = np.average(meds, weights=wts)
    return float(np.percentile(stats, 5)), float(np.percentile(stats, 95))


def decide_h1(
    theta: float,
    theta_p05: float,
    mean_sample_frobenius: float,
    power: float,
    n_seeds: int,
    grid_complete: bool,
) -> str:
    """The only H1 label. Verification samples cannot say HOLDS."""
    if n_seeds < len(SEEDS_PRIMARY) or not grid_complete:
        return "NOT_A_DECISION_SAMPLE"
    if not np.isfinite(power) or power < POWER_MIN:
        return "INCONCLUSIVE_UNDERPOWERED"
    if mean_sample_frobenius <= 0 or not np.isfinite(theta):
        return "NOT_SUPPORTED"
    relative = theta / mean_sample_frobenius
    if theta_p05 > 0 and relative >= MDE_RELATIVE_FROBENIUS:
        return "HOLDS"
    if theta_p05 > 0:
        return "DETECTED_BELOW_MDE"
    return "NOT_SUPPORTED"


def decide_h2(theta: float, theta_p05: float, n_seeds: int, grid_complete: bool) -> str:
    if n_seeds < len(SEEDS_PRIMARY) or not grid_complete:
        return "NOT_A_DECISION_SAMPLE"
    if theta_p05 > 0 and theta >= MDE_EFFECTIVE_BETS:
        return "HOLDS"
    if theta_p05 > 0:
        return "DETECTED_BELOW_MDE"
    return "NOT_SUPPORTED"


def h3_reversal(sharpe_gap_0: float, sharpe_gap_20: float) -> bool:
    if not np.isfinite(sharpe_gap_0) or not np.isfinite(sharpe_gap_20):
        return False
    if abs(sharpe_gap_0) < H3_SHARPE_FLOOR or abs(sharpe_gap_20) < H3_SHARPE_FLOOR:
        return False
    return np.sign(sharpe_gap_0) != np.sign(sharpe_gap_20)


def block_bootstrap_sharpe(returns: np.ndarray, n_boot: int, block: int, seed: int) -> tuple[float, float]:
    """Within-path interval. Not used to decide H1."""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < block * 2:
        return float("nan"), float("nan")
    stats = []
    n_blocks = int(np.ceil(n / block))
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([r[s : s + block] for s in starts])[:n]
        stats.append(sharpe(sample))
    return float(np.nanpercentile(stats, 5)), float(np.nanpercentile(stats, 95))


def parametric_power(
    mu: float,
    sigma: float,
    mde_abs: float,
    n_seeds: int,
    n_cells: int,
    cell_weights: np.ndarray,
    n_draws: int,
    seed: int,
    boot_reps: int = 200,
) -> float:
    """Power of the compound rule under iid Gaussian seed-level deltas.

    Cells are independent and share one variance. If variance rises with
    N/T, the number is an upper bound for the N=50 cells.
    """
    if not np.isfinite(sigma) or sigma <= 0 or n_seeds < 2 or n_cells < 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    weights = np.asarray(cell_weights, dtype=float)
    weights = weights / weights.sum()
    hits = 0
    done = 0
    chunk = 50
    while done < n_draws:
        m = min(chunk, n_draws - done)
        sample = rng.normal(mu, sigma, size=(m, n_cells, n_seeds))
        theta = np.median(sample, axis=2) @ weights
        idx = rng.integers(0, n_seeds, size=(m, boot_reps, n_cells, n_seeds))
        picked = np.take_along_axis(sample[:, None, :, :], idx, axis=3)
        boot = np.median(picked, axis=3) @ weights
        p05 = np.percentile(boot, 5, axis=1)
        hits += int(np.sum((p05 > 0.0) & (theta >= mde_abs)))
        done += m
    return hits / n_draws
