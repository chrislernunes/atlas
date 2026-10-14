from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_ASSETS = [
    "US_Equity", "Dev_Equity", "EM_Equity", "Govt_Bonds", "Inflation_Linkers",
    "IG_Credit", "HY_Credit", "Commodities", "Gold", "Listed_REITs",
    "Infrastructure_Equity", "Managed_Futures"
]

def load_returns_csv(path: str) -> pd.DataFrame:
    """Load a date-indexed CSV of periodic simple returns; first column must be dates."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if df.index.has_duplicates:
        raise ValueError("Date index contains duplicates")
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise ValueError("Returns contain missing/non-numeric values; clean explicitly before loading")
    if (df <= -1).any().any():
        raise ValueError("Simple returns must be greater than -100%")
    return df

def make_synthetic_returns(n_periods: int = 360, seed: int = 7,
                           assets: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic synthetic monthly returns with two latent regimes. NOT market data."""
    assets = assets or DEFAULT_ASSETS
    rng = np.random.default_rng(seed)
    n = len(assets)
    # Persistent regimes with occasional stress transitions.
    regime = np.zeros(n_periods, dtype=int)
    for t in range(1, n_periods):
        p_stress = 0.035 if regime[t-1] == 0 else 0.18
        regime[t] = 1 if rng.random() < p_stress else (0 if regime[t-1] == 1 and rng.random() < 0.35 else regime[t-1])
    # Shared equity, duration, inflation and trend factors.
    base_corr = np.eye(n) * 0.55 + np.ones((n, n)) * 0.08
    equity_idx = [i for i, a in enumerate(assets) if any(k in a for k in ["Equity", "REIT", "Credit"])]
    for i in equity_idx:
        for j in equity_idx:
            if i != j: base_corr[i, j] += 0.18
    base_corr = (base_corr + base_corr.T) / 2
    eig, vec = np.linalg.eigh(base_corr)
    base_corr = vec @ np.diag(np.maximum(eig, 0.05)) @ vec.T
    d = np.sqrt(np.diag(base_corr)); base_corr = base_corr / np.outer(d, d)
    vol = np.array([0.045 if "Equity" in a else 0.025 if "Bond" in a or "Linker" in a else 0.035 for a in assets])
    vol = np.where(np.array(["Credit" in a for a in assets]), 0.025, vol)
    means = np.array([0.006 if "Equity" in a else 0.002 if "Bond" in a else 0.003 for a in assets])
    rows = []
    for t in range(n_periods):
        stress = regime[t] == 1
        scale = 1.8 if stress else 1.0
        corr = base_corr.copy()
        if stress:
            corr += 0.16 * np.outer(np.array([1.0 if i in equity_idx else 0.25 for i in range(n)]),
                                     np.array([1.0 if i in equity_idx else 0.25 for i in range(n)]))
            np.fill_diagonal(corr, 1.0)
            eig, vec = np.linalg.eigh(corr); corr = vec @ np.diag(np.maximum(eig, 0.02)) @ vec.T
            d = np.sqrt(np.diag(corr)); corr = corr / np.outer(d, d)
        cov = np.outer(vol * scale, vol * scale) * corr
        draw = rng.multivariate_normal(means + (-0.006 if stress else 0), cov)
        # Mild heavy tails to avoid a Gaussian-only demo.
        draw += rng.standard_t(df=7, size=n) * vol * (0.12 if stress else 0.05)
        rows.append(draw)
    dates = pd.date_range("1996-01-31", periods=n_periods, freq="ME")
    return pd.DataFrame(rows, index=dates, columns=assets), pd.Series(regime, index=dates, name="latent_stress_regime")
