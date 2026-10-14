# Robust Hierarchical Multi-Asset Allocation under Estimation Error and Regime Uncertainty
## Pre-registered Experimental Design and Analysis Framework (Synthetic Data Only)

**Document type:** Internal research note / pre-registration  
**Status:** Design freeze before any results  
**Data restriction:** Synthetic only. No real-market claims are permitted or implied.  
**Date:** 2026-10-15

### 1. Revised Hypotheses and Pre-registered Success Criteria

#### Primary hypotheses (pre-registered)
- **H1 (Estimation error).** When the number of assets \(N\) is large relative to the estimation window \(T\) (\(N/T \gtrsim 0.4\)), Ledoit–Wolf shrinkage and hierarchical risk parity (HRP) produce lower out-of-sample covariance prediction error and lower realized portfolio variance than sample-covariance minimum variance.
- **H2 (Stability / concentration).** Under moderate-to-high estimation noise, HRP produces (a) higher effective number of bets, (b) lower maximum weight, and (c) lower weight turnover than constrained sample-covariance minimum variance, holding transaction costs at zero.
- **H3 (Cost and regime robustness).** The ranking between methods on net-of-cost Sharpe, maximum drawdown, and expected shortfall is not invariant to (a) proportional transaction costs of 5–20 bps one-way and (b) controlled volatility and correlation shocks. Methods that dominate on gross risk metrics may fail to dominate after costs or under shocks.

#### Secondary hypotheses
- **H4.** A constant-correlation or single-factor covariance estimator will be competitive with Ledoit–Wolf when the true correlation structure is close to constant or low-rank.
- **H5.** An estimated regime detector (fit only on data available at \(t\)) improves pre-registered out-of-sample risk metrics relative to a no-regime baseline only when the injected regime signal-to-noise ratio is high and the detector is evaluated strictly out-of-sample.

#### Pre-registered success criteria
A method is declared to dominate a baseline on a given metric only if:
1. The difference survives the outer-loop walk-forward (no look-ahead).
2. The 90 % block-bootstrap interval for the difference excludes zero in the direction of improvement.
3. The result holds on at least 70 % of the factorial cells in the pre-specified primary grid (asset-count × correlation structure × window length).
4. The ranking is reported separately for gross and net-of-cost returns.

Failure to meet any of the above is recorded as a negative or mixed result for that comparison. No method is claimed to “work” in general.

### 2. Complete Synthetic Data-Generating Process Specification

All experiments use a fully controlled multivariate Gaussian (or Student-\(t\)) factor model with known true covariance \(\Sigma_t\).

#### Asset universe
- \(N \in \{8, 20, 50\}\).  
- Asset names are labels only; no economic interpretation is attached.

#### Factor structure (controlled correlation)
Four pre-specified true correlation regimes:
1. **Low:** equicorrelation \(\rho = 0.10\).
2. **Moderate:** equicorrelation \(\rho = 0.30\).
3. **Block:** two or three equal-sized blocks with intra-block \(\rho = 0.60\), inter-block \(\rho = 0.05\).
4. **High:** equicorrelation \(\rho = 0.60\).

True covariance is formed as \(\Sigma = D\, R\, D\) where \(D\) is diagonal of annualized volatilities scaled to monthly. Base annualized volatilities are drawn once per seed from \(\mathrm{Unif}(0.08, 0.25)\) and then held fixed except under explicit shocks.

#### Returns
\[
r_t = \mu + L_t\, z_t, \quad z_t \sim \mathcal{N}(0,I) \text{ or } t_\nu\ (\nu=7)
\]
where \(L_t L_t^\top = \Sigma_t\). Mean \(\mu\) is zero in the primary design (or a small constant vector fixed across seeds).

#### Controlled shocks (exogenous regimes)
- **Volatility jump:** at a known date, multiply all diagonal elements of \(\Sigma\) by 2.0 for 24 periods, then revert.
- **Correlation spike:** at a known date, replace \(R\) with a higher-correlation matrix (e.g., equicorrelation 0.70) for 24 periods, then revert.
- Shocks are non-overlapping and their timing is known to the experimentalist but invisible to any estimator or regime detector under test.

#### Estimation windows and seeds
- Lookback \(T \in \{36, 60, 120\}\) months.
- 30 independent random seeds for the primary grid; 10 seeds for secondary sensitivity.
- All random number generation uses `numpy.random.Generator` with explicit seeds.

#### Known quantities recorded
At every rebalance date the true \(\Sigma_t\) (and therefore true correlation and eigenvalues) is stored so that estimation error can be computed exactly.

### 3. Exact Evaluation Protocol

#### Nested validation structure
- **Outer loop:** walk-forward. At date \(t\), all estimation and hyperparameter selection use only returns \(\{r_s : s < t\}\).
- **Inner loop (hyperparameters only):** for any method that has free parameters (shrinkage intensity if not Ledoit–Wolf, linkage method, window length if not fixed), a further rolling validation window inside the outer training set is used. The selected parameter is then frozen for the out-of-sample step.
- No full-sample quantities are ever used.

#### Methods (all long-only, fully invested, weight caps optional and pre-specified)
1. Equal-weight.
2. Inverse-volatility.
3. Constrained minimum variance – sample covariance.
4. Constrained minimum variance – Ledoit–Wolf.
5. HRP (correlation distance, single or average linkage, recursive bisection).
6. Constrained minimum variance – constant-correlation covariance (average pairwise correlation).
7. Constrained minimum variance – single-factor covariance (first principal component + residual variances).

#### Transaction costs and diagnostics
- Proportional one-way cost: \(c \in \{0, 5, 10, 20\}\) bps applied to \(0.5 \sum_i |\Delta w_i|\).
- Full turnover series, effective number of bets \(1 / \sum w_i^2\), maximum weight, risk-contribution vector, and weight-drift (average absolute weight change between rebalances) are recorded.

#### Covariance estimation error (primary diagnostic)
Against the known true \(\Sigma_t\):
- Frobenius norm \(\|\hat{\Sigma}_t - \Sigma_t\|_F\).
- Eigenvalue error (sorted absolute differences, sum and max).
- Predictive risk-forecast error: realized portfolio variance over the subsequent holding period versus the ex-ante forecast \(w^\top \hat{\Sigma} w\).

#### Statistical uncertainty
Every reported metric (Sharpe, volatility, drawdown, turnover, estimation error) is accompanied by a 90 % interval obtained from a stationary block bootstrap (block length = 6 months) of the outer-loop strategy returns or of the estimation-error series. Bootstrap is performed independently per seed and then aggregated.

#### Regime detector evaluation (secondary, only after stress results are documented)
A simple two-state detector (e.g., rolling realized volatility threshold or a short-window change-point statistic) is fit exclusively on data before \(t\). Its decisions are compared to the no-regime baseline on the same pre-registered metrics. Full-sample smoothed probabilities are forbidden.

### 4. Metric Definitions and Required Tables / Figures

#### Required tables (all synthetic, all with bootstrap intervals)
- Table 1: Estimation error (Frobenius, eigenvalue, predictive) by method, \(N/T\) ratio, and correlation structure.
- Table 2: Gross portfolio metrics (ann. vol, max DD, ES 5 %, effective bets, max weight).
- Table 3: Net-of-cost metrics at 0 / 5 / 20 bps.
- Table 4: Ranking stability across the factorial grid (fraction of cells in which method A beats method B).
- Table 5: Degradation under explicit volatility and correlation shocks.

#### Required figures
- Figure 1: Estimation error vs. \(N/T\) for sample, Ledoit–Wolf, constant-correlation.
- Figure 2: Effective number of bets and turnover distributions (HRP vs. sample MV).
- Figure 3: Cumulative net wealth under the primary seed set (median path + interquartile band).
- Figure 4: Performance difference (HRP – sample MV) across correlation structures, with bootstrap intervals.
- Figure 5: Shock window performance (pre-shock, shock, post-shock).

All figures carry the label “Synthetic data only – not market performance.”

### 5. Failure-Mode Checklist

The following must be explicitly tested and reported, even if negative:
- [ ] HRP or shrinkage increases rather than decreases estimation error when \(N/T\) is small.
- [ ] HRP produces higher turnover than sample MV under some correlation structures.
- [ ] After 10–20 bps costs the net ranking reverses relative to the gross ranking.
- [ ] Under a pure correlation spike the hierarchical clustering distance becomes unstable and weight changes are large.
- [ ] The constant-correlation estimator dominates Ledoit–Wolf when the true structure is equicorrelation.
- [ ] An estimated regime detector fitted only on past data degrades out-of-sample risk metrics relative to the no-regime baseline.
- [ ] Bootstrap intervals for the primary differences include zero on a majority of seeds.

### 6. What a Negative or Mixed Result Would Still Teach

- That the benefit of shrinkage or hierarchical allocation is conditional on the \(N/T\) ratio and the true correlation structure, and disappears outside that region.
- That transaction costs of realistic magnitude can erase any gross-risk advantage of more adaptive methods.
- That controlled correlation spikes can produce larger weight instability in clustering-based methods than in simple shrinkage.
- That a regime detector estimated in real time may not improve decisions even when the experimentalist knows the true regime sequence exists.
- That the practical value of the methods under study is limited to specific, measurable regions of the estimation-error / cost / shock space rather than being general.

These statements remain scientifically useful because they are falsifiable, quantified, and obtained under a known data-generating process.

### 7. Minimal Reproducible Code Architecture

```
data/
  generate.py          # make_synthetic_panel(n_assets, corr_type, shock_schedule, seed) -> returns, true_Sigma
estimators/
  covariance.py        # sample, ledoit_wolf, constant_corr, factor
allocators/
  weights.py           # equal, inv_vol, min_var, hrp, ...
backtest/
  walk_forward.py      # outer loop; returns strategy_returns, weights, estimation_errors
metrics/
  evaluation.py        # frobenius, eigenvalue_error, predictive_risk, portfolio_stats, block_bootstrap
experiment/
  factorial.py         # grid over N, T, corr_type, seeds; writes results.parquet
  regime_stress.py     # inject shocks, re-run, compare
notebooks/
  pre_registration.ipynb   # this document rendered + empty result placeholders
  results.ipynb            # loads saved results only; no estimation
```

**Data flow**
1. `generate.py` produces returns and the true \(\Sigma_t\) series (saved once per cell).
2. `walk_forward.py` consumes only past returns, calls the chosen estimator and allocator, records weights, turnover, and \(\|\hat{\Sigma}-\Sigma\|_F\).
3. `evaluation.py` computes all pre-registered metrics and bootstrap intervals.
4. `factorial.py` loops over the grid, writes a single results table.
5. The results notebook reads only the saved table; it never re-estimates.

All random seeds, window lengths, cost levels, and hyperparameter grids are declared in a single configuration object before any loop is executed.

### Limitations of the Synthetic Design (explicit)

The design can claim:
- Relative ordering of methods under known \(N/T\), known correlation structure, and known shocks.
- Exact covariance estimation error.
- Sensitivity of net performance to proportional costs.

The design cannot claim:
- Performance on real markets.
- Behavior under missing data, asynchronous observations, or non-stationary factor loadings beyond the injected shocks.
- Capacity, market impact, or financing effects.
- That any method will remain superior once the true data-generating process is unknown.

End of pre-registration.
