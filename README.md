# Multi-Asset Allocator Research Scaffold

A research codebase for testing hierarchical allocation, covariance shrinkage, estimation error and regime-shift sensitivity in a walk-forward portfolio setting.

**Important:** bundled data and generated charts are synthetic demonstration artifacts, not market data or evidence of investment performance. No ADIA affiliation or institutional deployment is implied.

## Quick start (Python 3.10+)

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e '.[test]'
pytest
python run_experiment.py
jupyter notebook notebooks/robust_allocator_research.ipynb
```

The demo saves CSVs, a metrics table and charts under `outputs/`.

## Current implementation
- Equal-weight and inverse-volatility baselines.
- Long-only constrained minimum variance using sample covariance and Ledoit-Wolf shrinkage.
- HRP with correlation-distance clustering and recursive bisection.
- Strict trailing-window walk-forward backtest; weights used for period *t* are computed from returns ending at *t-1*.
- Stylized proportional turnover costs and portfolio diagnostics.
- Synthetic two-regime data generator for repeatable plumbing tests.

## Data contract
`load_returns_csv(path)` expects a CSV with dates in the first column and periodic simple returns in each subsequent column. Use total-return series where appropriate; do not feed price levels. Missing observations are rejected instead of silently forward-filled.

## What is not implemented yet
This is a tested baseline scaffold, not the entire proposed institutional research system. The current version does not implement a fitted HMM/change-point model, Bayesian uncertainty integration, private asset cash-flow/NAV models, nonlinear market impact, or multi-trillion-dollar capacity analysis. These are intentionally excluded until the baseline experiment is validated on point-in-time real data.

See [`docs/RESEARCH_NOTE.md`](docs/RESEARCH_NOTE.md) for the original research design, limitations and required next steps.

## Pre-registered synthetic study (2026-10-15)

[`docs/RESEARCH_NOTE_UPGRADED.md`](docs/RESEARCH_NOTE_UPGRADED.md) freezes hypotheses, the controlled data-generating process, nested walk-forward protocol, estimation-error metrics, and failure modes **before** results. Executable modules live under [`study/`](study/). Synthetic data only. A reduced-grid notebook with saved outputs is at [`study/notebooks/pre_registration_demo_executed.ipynb`](study/notebooks/pre_registration_demo_executed.ipynb). Nothing in that notebook is a claim that any method dominates.

For a multi-seed, transaction-cost sensitivity run (takes longer):

```bash
python run_sensitivity.py
```

It runs 10 synthetic seeds at 0, 5 and 20 bps and reports medians. This is a software stress test, not evidence of real-market performance.
