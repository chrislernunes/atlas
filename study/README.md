# Pre-registered synthetic study

Controlled factorial design for hierarchical allocation under known covariance error and exogenous regime shocks. Synthetic data only. This does not replace the baseline scaffold in `src/robust_allocator/`; it is the next experimental layer specified in `docs/RESEARCH_NOTE_UPGRADED.md`.

## Run a single cell

From the repository root:

```bash
PYTHONPATH=study python -c "
from data.generate import make_synthetic_panel
from backtest.walk_forward import walk_forward
rets, meta = make_synthetic_panel(12, 100, 'block', 36, 7)
res = walk_forward(rets, meta['true_sigmas'], 36, cost_bps=5)
print(res['errors'].mean())
"
```

## Reduced factorial demo

```bash
PYTHONPATH=study python study/experiment/factorial.py
```

The executed notebook is `study/notebooks/pre_registration_demo_executed.ipynb`. Rankings in that notebook are a software check on a reduced grid, not a pre-registered conclusion.

No real-market performance is claimed.
