from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from robust_allocator.data import make_synthetic_returns
from robust_allocator.backtest import walk_forward_backtest
from robust_allocator.metrics import summarize

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
returns, latent_regime = make_synthetic_returns(n_periods=360, seed=7)
returns.to_csv(OUT / "synthetic_returns_NOT_MARKET_DATA.csv")
latent_regime.to_csv(OUT / "synthetic_latent_regime_NOT_MARKET_DATA.csv")
strategy_returns, weights = walk_forward_backtest(returns, lookback=60, rebalance_every=1, cost_bps=5)
strategy_returns.to_csv(OUT / "walk_forward_strategy_returns.csv")
rows = []
for name in strategy_returns:
    rows.append({"strategy": name, **summarize(strategy_returns[name], weights[name])})
summary = pd.DataFrame(rows).set_index("strategy")
summary.to_csv(OUT / "summary.csv")
ax = (1 + strategy_returns).cumprod().plot(figsize=(12, 6), title="Synthetic demonstration only — not market performance")
ax.set_ylabel("Growth of $1")
ax.figure.tight_layout(); ax.figure.savefig(OUT / "equity_curves_SYNTHETIC.png", dpi=160); plt.close(ax.figure)
ax = weights["hrp"].plot.area(figsize=(12, 6), title="HRP weights (synthetic demonstration)")
ax.set_ylabel("Portfolio weight"); ax.figure.tight_layout(); ax.figure.savefig(OUT / "hrp_weights_SYNTHETIC.png", dpi=160); plt.close(ax.figure)
print(summary.round(4).to_string())
print(f"\nSaved outputs to {OUT.resolve()}")
