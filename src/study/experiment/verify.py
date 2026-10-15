"""Verification sample. This module refuses to label a hypothesis as decided."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from study.backtest.walk_forward import walk_forward
from study.data.generate import make_synthetic_panel
from study.metrics.evaluation import (
    decide_h1,
    decide_h2,
    h3_reversal,
    parametric_power,
    seed_bootstrap_theta,
    sharpe,
    theta_from_seed_deltas,
)
from study.protocol import (
    MDE_RELATIVE_FROBENIUS,
    POWER_ALTERNATIVE_MULTIPLE,
    POWER_DRAWS,
    SEEDS_PILOT,
    SEEDS_PRIMARY,
    STRATUM_CELLS,
    VERIFY_BOOT,
    VERIFY_CORRS,
    VERIFY_LOOKBACKS,
    VERIFY_N,
    VERIFY_PERIODS,
    VERIFY_SEEDS,
    VERIFY_TAILS,
)


def _run_cell(n, lookback, corr, seed, tail="gaussian", shock=None, use_detector=False, periods=VERIFY_PERIODS):
    returns, meta = make_synthetic_panel(n, periods, corr, seed, tail=tail, shock=shock)
    path = walk_forward(returns, meta["true_sigmas"], lookback, cost_bps=0.0, use_detector=use_detector)
    rows = []
    for method, g in path.groupby("method"):
        rows.append(
            {
                "n_assets": n,
                "lookback": lookback,
                "corr_type": corr,
                "seed": seed,
                "innovation": tail,
                "shock": shock or "",
                "detector": use_detector,
                "method": method,
                "frobenius": float(g["frobenius"].mean()),
                "eig_l1": float(g["eig_l1"].mean()),
                "risk_abs": float(g["risk_abs"].mean()),
                "eff_bets": float(g["eff_bets"].mean()),
                "max_weight": float(g["max_weight"].mean()),
                "turnover": float(g["turnover"].mean()),
                "sharpe_gross": sharpe(g["ret_gross"].to_numpy()),
                "sharpe_0": sharpe(g["ret_net"].to_numpy()),
                "sharpe_20": sharpe(((1 + g["ret_gross"]) * (1 - g["turnover"] * 20 / 1e4) - 1).to_numpy()),
                "var": float(g["ret_gross"].var(ddof=1)),
                "failed_frac": float(g["optimizer_failed"].mean()),
                "linkage_single_frac": float((g["linkage"] == "single").mean()) if method == "hrp" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_verification() -> dict[str, pd.DataFrame | dict]:
    frames = []
    for n in VERIFY_N:
        for lookback in VERIFY_LOOKBACKS:
            for corr in VERIFY_CORRS:
                for seed in VERIFY_SEEDS:
                    frames.append(_run_cell(n, lookback, corr, seed))
    for tail in ("t4", "mix"):
        for seed in VERIFY_SEEDS:
            frames.append(_run_cell(20, 36, "moderate", seed, tail=tail))
    for shock in ("vol", "corr"):
        for seed in VERIFY_SEEDS:
            frames.append(_run_cell(20, 36, "moderate", seed, shock=shock))
            frames.append(_run_cell(20, 36, "moderate", seed, shock=shock, use_detector=True))
    for seed in VERIFY_SEEDS:
        frames.append(_run_cell(20, 36, "moderate", seed, use_detector=True))
    panel = pd.concat(frames, ignore_index=True)

    pilot_rows = [_run_cell(20, 36, "moderate", seed) for seed in SEEDS_PILOT]
    pilot = pd.concat(pilot_rows, ignore_index=True)
    return {"panel": panel, "pilot": pilot, "decision": decision_record(panel, pilot)}


def _pair(frame: pd.DataFrame, method_a: str, method_b: str, column: str) -> dict[tuple, np.ndarray]:
    a = frame[frame.method == method_a].set_index(["n_assets", "lookback", "corr_type", "seed"])[column]
    b = frame[frame.method == method_b].set_index(["n_assets", "lookback", "corr_type", "seed"])[column]
    delta = (a - b).dropna()
    out = {}
    for key, g in delta.groupby(level=[0, 1, 2]):
        out[key] = g.to_numpy()
    return out


def _filled(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["shock"] = out["shock"].fillna("")
    out["detector"] = out["detector"].astype(bool)
    return out


def decision_record(panel: pd.DataFrame, pilot: pd.DataFrame) -> dict:
    panel = _filled(panel)
    pilot = _filled(pilot)
    base = panel[(panel["innovation"] == "gaussian") & (panel["shock"] == "") & (~panel["detector"])]
    stratum_keys = set(STRATUM_CELLS)
    present = base[base.apply(lambda r: (r.n_assets, r.lookback, r.corr_type) in stratum_keys, axis=1)]
    deltas = _pair(present, "sample_mv", "lw_mv", "frobenius")
    # Keep only locked stratum keys that actually appear.
    deltas = {k: v for k, v in deltas.items() if k in stratum_keys and v.size}
    weights = {k: k[0] / k[1] for k in deltas}
    theta = theta_from_seed_deltas(deltas, weights) if deltas else float("nan")
    p05, p95 = seed_bootstrap_theta(deltas, weights, VERIFY_BOOT, 11) if deltas else (float("nan"), float("nan"))
    sample_level = present[present.method == "sample_mv"]
    # Mean sample Frobenius on the same cells, seed-mean then cell-mean with the same weights.
    sample_cell = sample_level.groupby(["n_assets", "lookback", "corr_type"]).frobenius.mean()
    sample_cell = sample_cell.loc[[k for k in deltas]]
    mean_sample = float(np.average(sample_cell.to_numpy(), weights=[weights[k] for k in sample_cell.index])) if len(sample_cell) else float("nan")

    pilot_s = pilot[(pilot.method == "sample_mv")].set_index("seed").frobenius
    pilot_l = pilot[(pilot.method == "lw_mv")].set_index("seed").frobenius
    pilot_delta = (pilot_s - pilot_l).to_numpy()
    sigma = float(np.std(pilot_delta, ddof=1))
    pilot_fro = float(pilot_s.mean())
    mde_abs = MDE_RELATIVE_FROBENIUS * pilot_fro
    design_mu = POWER_ALTERNATIVE_MULTIPLE * mde_abs
    cell_w = np.array([n / t for n, t, _ in STRATUM_CELLS], dtype=float)
    power_design = parametric_power(design_mu, sigma, mde_abs, len(SEEDS_PRIMARY), len(STRATUM_CELLS), cell_w, POWER_DRAWS, 12)
    power_at_mde = parametric_power(mde_abs, sigma, mde_abs, len(SEEDS_PRIMARY), len(STRATUM_CELLS), cell_w, POWER_DRAWS, 13)
    h1 = decide_h1(theta, p05, mean_sample, power_design, n_seeds=present.seed.nunique(), grid_complete=set(deltas) == stratum_keys)

    bet_deltas = _pair(present, "hrp", "sample_mv", "eff_bets")
    bet_deltas = {k: v for k, v in bet_deltas.items() if k in stratum_keys and v.size}
    bet_w = {k: 1.0 for k in bet_deltas}
    theta_h2 = theta_from_seed_deltas(bet_deltas, bet_w) if bet_deltas else float("nan")
    h2_p05, h2_p95 = seed_bootstrap_theta(bet_deltas, bet_w, VERIFY_BOOT, 14) if bet_deltas else (float("nan"), float("nan"))
    h2 = decide_h2(theta_h2, h2_p05, n_seeds=present.seed.nunique(), grid_complete=set(bet_deltas) == stratum_keys)

    reversals = []
    wide0 = base.pivot_table(index=["n_assets", "lookback", "corr_type", "seed"], columns="method", values="sharpe_0")
    wide20 = base.pivot_table(index=["n_assets", "lookback", "corr_type", "seed"], columns="method", values="sharpe_20")
    gap0 = (wide0["hrp"] - wide0["sample_mv"]).groupby(level=[0, 1, 2]).median()
    gap20 = (wide20["hrp"] - wide20["sample_mv"]).groupby(level=[0, 1, 2]).median()
    for key in gap0.index.intersection(gap20.index):
        if h3_reversal(float(gap0.loc[key]), float(gap20.loc[key])):
            reversals.append(key)

    return {
        "h1_status": h1,
        "theta_h1": theta,
        "theta_h1_p05": p05,
        "theta_h1_p95": p95,
        "mean_sample_frobenius": mean_sample,
        "relative_h1": theta / mean_sample if mean_sample else float("nan"),
        "pilot_sigma": sigma,
        "pilot_mean_sample_frobenius": pilot_fro,
        "mde_abs": mde_abs,
        "power_at_design_alternative": power_design,
        "power_at_mde": power_at_mde,
        "h2_status": h2,
        "theta_h2": theta_h2,
        "theta_h2_p05": h2_p05,
        "theta_h2_p95": h2_p95,
        "h3_reversal_cells": len(reversals),
        "stratum_cells_present": len(deltas),
        "stratum_cells_required": len(STRATUM_CELLS),
        "seeds_present": int(present.seed.nunique()) if len(present) else 0,
        "seeds_required": len(SEEDS_PRIMARY),
    }


def write_outputs(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    result["panel"].to_csv(out / "verification_panel.csv", index=False)
    result["pilot"].to_csv(out / "pilot_panel.csv", index=False)
    pd.Series(result["decision"]).to_csv(out / "decision_record.csv", header=["value"])


if __name__ == "__main__":
    result = run_verification()
    write_outputs(result, Path("outputs/verification"))
    for k, v in result["decision"].items():
        print(f"{k}: {v}")
