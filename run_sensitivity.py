"""Multi-seed/cost sensitivity on synthetic data; not a market backtest."""
from pathlib import Path
import pandas as pd
from robust_allocator.data import make_synthetic_returns
from robust_allocator.backtest import walk_forward_backtest
from robust_allocator.metrics import summarize

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
rows = []
for seed in range(5):
    returns, _ = make_synthetic_returns(n_periods=180, seed=seed)
    for cost_bps in (0.0, 10.0, 25.0):
        strategy_returns, weights = walk_forward_backtest(returns, lookback=60, rebalance_every=3, cost_bps=cost_bps)
        for strategy in strategy_returns:
            rows.append({"seed": seed, "cost_bps": cost_bps, "strategy": strategy,
                         **summarize(strategy_returns[strategy], weights[strategy])})
raw = pd.DataFrame(rows)
raw.to_csv(OUT / "synthetic_sensitivity_all_runs_NOT_MARKET_DATA.csv", index=False)
summary = raw.groupby(["cost_bps", "strategy"]).agg(
    median_ann_return=("annualized_return", "median"),
    median_ann_vol=("annualized_volatility", "median"),
    median_sharpe=("sharpe_rf0", "median"),
    median_max_drawdown=("max_drawdown", "median"),
    median_turnover=("average_turnover_one_way", "median"),
    median_effective_bets=("average_effective_bets", "median"),
    positive_return_fraction=("annualized_return", lambda x: float((x > 0).mean())),
)
summary.to_csv(OUT / "synthetic_sensitivity_summary_NOT_MARKET_DATA.csv")
print(summary.round(4).to_string())
