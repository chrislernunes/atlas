# Atlas

Long-only walk-forward allocation on a locked synthetic design.

Protocol, estimands, power, and the verification tables: <https://chrislernunes.github.io/atlas/>

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Test

```bash
pytest
```

## Run

```bash
PYTHONPATH=src python -m study.experiment.verify
python run_experiment.py
python run_sensitivity.py
```

`study.experiment.verify` writes `outputs/verification/`. That run is not the decision sample. Hypothesis labels stay `NOT_A_DECISION_SAMPLE` until the locked grid in `src/study/protocol.py` is complete and the power gate passes.

Synthetic returns only.
