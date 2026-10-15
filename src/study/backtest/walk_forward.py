"""Walk-forward. Weights at t use returns strictly before t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from study.allocators.weights import equal_weight, hrp, inverse_vol, min_variance
from study.estimators.covariance import constant_corr_cov, factor_cov, ledoit_wolf_cov, sample_cov
from study.metrics.evaluation import effective_bets, eigenvalue_l1, frobenius, risk_forecast_abs
from study.protocol import INNER_MIN_FIT, INNER_MONTHS, LINKAGE_CANDIDATES, LINKAGE_DEFAULT, STRESS_LOOKBACK
from study.regime.detector import detect_stress

COV_METHODS = ("sample_mv", "lw_mv", "const_corr_mv", "factor_mv", "hrp")


def _covs(train: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, float]]:
    sample = sample_cov(train)
    lw, intensity = ledoit_wolf_cov(train)
    return {
        "sample_mv": (sample, np.nan),
        "lw_mv": (lw, intensity),
        "const_corr_mv": (constant_corr_cov(train), np.nan),
        "factor_mv": (factor_cov(train), np.nan),
        "hrp": (lw, intensity),
    }


def _select_linkage(train: pd.DataFrame) -> tuple[str, bool]:
    if len(train) < INNER_MIN_FIT + INNER_MONTHS:
        return LINKAGE_DEFAULT, True
    fit = train.iloc[: -INNER_MONTHS]
    hold = train.iloc[-INNER_MONTHS :]
    if len(fit) < INNER_MIN_FIT:
        return LINKAGE_DEFAULT, True
    best = None
    best_var = np.inf
    cov, _ = ledoit_wolf_cov(fit)
    for name in LINKAGE_CANDIDATES:
        weights, _ = hrp(cov, method=name)
        var = float(np.var(hold.to_numpy() @ weights.to_numpy(), ddof=1))
        if var < best_var - 1e-12 or (abs(var - best_var) <= 1e-12 and name == LINKAGE_DEFAULT):
            best = name
            best_var = var
    return best if best is not None else LINKAGE_DEFAULT, False


def walk_forward(
    returns: pd.DataFrame,
    true_sigmas: list[np.ndarray],
    lookback: int,
    cost_bps: float = 0.0,
    use_detector: bool = False,
) -> pd.DataFrame:
    n = len(returns)
    rows = []
    prev: dict[str, np.ndarray | None] = {}
    for t in range(lookback, n):
        full = returns.iloc[t - lookback : t]
        state = detect_stress(full) if use_detector else "off"
        used = min(STRESS_LOOKBACK, lookback) if state == "stress" else lookback
        train = full.iloc[-used:]
        linkage, skipped = _select_linkage(train)
        covs = _covs(train)
        true = true_sigmas[t]
        gross_ret = returns.iloc[t].to_numpy()
        for method in ("equal_weight", "inverse_vol", *COV_METHODS):
            failed = False
            intensity = np.nan
            link = ""
            if method == "equal_weight":
                w = equal_weight(returns.shape[1], returns.columns).to_numpy()
                cov = None
            elif method == "inverse_vol":
                series, ok = inverse_vol(train)
                w = series.to_numpy()
                failed = not ok
                cov = None
            elif method == "hrp":
                cov, intensity = covs["hrp"]
                series, ok = hrp(cov, method=linkage)
                w = series.to_numpy()
                failed = not ok
                link = linkage
            else:
                cov, intensity = covs[method]
                x0 = prev.get(method)
                series, ok = min_variance(cov, x0=x0)
                w = series.to_numpy()
                failed = not ok
            prev[method] = w
            prev_w = prev.get("_last_" + method)
            if prev_w is None:
                turnover = 0.5 * float(np.abs(w).sum())
            else:
                turnover = 0.5 * float(np.abs(w - prev_w).sum())
            prev["_last_" + method] = w.copy()
            gross = float(w @ gross_ret)
            net = (1.0 + gross) * (1.0 - turnover * cost_bps / 1e4) - 1.0
            hat = None if cov is None else cov.to_numpy()
            rows.append(
                {
                    "date": returns.index[t],
                    "method": method,
                    "ret_gross": gross,
                    "ret_net": net,
                    "turnover": turnover,
                    "frobenius": np.nan if hat is None else frobenius(hat, true),
                    "eig_l1": np.nan if hat is None else eigenvalue_l1(hat, true),
                    "risk_abs": np.nan if hat is None else risk_forecast_abs(w, hat, true),
                    "eff_bets": effective_bets(w),
                    "max_weight": float(np.max(w)),
                    "optimizer_failed": failed,
                    "linkage": link,
                    "inner_skipped": bool(skipped) if method == "hrp" else False,
                    "detector": state,
                    "lookback_used": used,
                    "shrinkage": intensity,
                }
            )
    return pd.DataFrame(rows)
