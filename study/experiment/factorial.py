"""Small factorial runner for demonstration (primary grid subset)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from data.generate import make_synthetic_panel
from backtest.walk_forward import walk_forward
from metrics.evaluation import portfolio_stats, frobenius, effective_bets, block_bootstrap_sharpe


def run_cell(n_assets, corr_type, lookback, seed, cost_bps=5.0, n_periods=180):
    rets, meta = make_synthetic_panel(n_assets, n_periods, corr_type, lookback, seed)
    res = walk_forward(rets, meta["true_sigmas"], lookback, cost_bps=cost_bps)
    rows = []
    for m in res["returns"].columns:
        stats = portfolio_stats(res["returns"][m])
        lo, hi = block_bootstrap_sharpe(res["returns"][m], n_boot=100, seed=seed)
        rows.append({
            "n_assets": n_assets,
            "corr_type": corr_type,
            "lookback": lookback,
            "seed": seed,
            "method": m,
            "mean_frobenius": frobenius(res["errors"][m]),
            "eff_bets": effective_bets(res["weights"][m]),
            "sharpe_lo": lo,
            "sharpe_hi": hi,
            **stats,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    grid = []
    for n in [8, 20]:
        for corr in ["low", "moderate", "high"]:
            for T in [36, 60]:
                for seed in range(3):  # small for demo
                    grid.append(run_cell(n, corr, T, seed))
    df = pd.concat(grid, ignore_index=True)
    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "factorial_demo.parquet", index=False)
    print(df.groupby(["n_assets", "corr_type", "method"])[["ann_vol", "sharpe", "mean_frobenius", "eff_bets"]].mean().round(3))
