# Robust Hierarchical Multi-Asset Allocation under Estimation Error and Regime Uncertainty

## Abstract
This project is a reproducible research scaffold for evaluating whether hierarchical allocation and covariance shrinkage improve out-of-sample portfolio behavior relative to simple baselines. It treats stability, turnover and risk control as first-class outcomes rather than assuming a higher backtest Sharpe proves value. The bundled dataset is synthetic and is used only to validate plumbing; it is not evidence of a market anomaly or an investable result.

## Hypotheses
- H1: covariance shrinkage reduces out-of-sample risk-estimation error versus raw sample covariance when the asset count is large relative to the lookback.
- H2: HRP produces less concentrated or more stable allocations than constrained sample-covariance minimum variance in some noisy-covariance settings.
- H3: these differences may disappear or reverse after turnover costs and under regime shifts.

Each hypothesis must be evaluated over repeated seeds, sample sizes, asset counts, and controlled shifts. A single synthetic run is not a result.

## Methods in this starter
- Long-only, fully invested equal-weight and inverse-volatility baselines.
- Constrained minimum variance using sample covariance or Ledoit-Wolf shrinkage.
- HRP using correlation-distance hierarchical clustering, quasi-diagonal ordering, and recursive bisection.
- Expanding-time walk-forward evaluation with a trailing estimation window; weights at period t use data strictly before t.
- Proportional transaction cost charged at rebalance, plus turnover, max drawdown, annualized volatility, zero-risk-free Sharpe, effective number of bets and average maximum weight.

## Important limitations
1. The synthetic data generator has known stylized regimes; it does not mimic all features of actual markets.
2. The current engine applies proportional costs only. It does not model nonlinear market impact, spreads by asset, capacity, borrow, financing, FX hedging, futures rolls, taxes, or execution scheduling.
3. Private-market assets are not represented as investable private assets. Public listed proxies cannot reproduce stale NAV marks, appraisal smoothing, capital calls, gates, or valuation lag. Do not relabel synthetic or listed data as private-market returns.
4. The engine does not currently implement a regime detector. Regime uncertainty should first be evaluated as a controlled stress experiment; only then should an estimated model be added and validated against a no-regime baseline.
5. There is no claim that HRP or Ledoit-Wolf wins. If it does not, report that.
6. Metrics are descriptive. Sharpe ratios on short, non-normal, serially correlated or synthetic samples have substantial uncertainty.

## Required full study before any investment conclusion
- Use timestamped total-return indices with explicit currency and hedging conventions; align calendars and corporate-action treatment.
- Freeze universe membership point-in-time and document survivorship, backfill, stale pricing and missing-data policies.
- Compare sample covariance, Ledoit-Wolf, constant-correlation/factor covariance, HRP, inverse volatility, equal weight and constrained minimum variance.
- Measure covariance error against a known covariance in simulations; on real data use realized risk forecasts and calibration diagnostics.
- Run nested walk-forward tuning, untouched final holdout, block bootstrap, parameter sweeps, correlation shocks, volatility jumps, liquidity/cost shocks and estimation-window sensitivity.
- Report gross/net return, volatility, drawdown, expected shortfall, turnover, concentration, effective bets, risk contributions, weight drift, cost attribution and uncertainty intervals.
- Add a regime model only if it improves pre-registered out-of-sample decisions; avoid using smoothed full-sample state probabilities.

## Interpretation
The most credible outcome may be that the robust method sacrifices return or does not dominate the baselines. The project succeeds if the conclusion is reproducible, assumptions are explicit, and failure modes are honestly documented—not if one strategy wins every metric.
