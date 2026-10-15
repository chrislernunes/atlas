"""Pre-registered constants. Changing these changes the study.

The verification sample is not allowed to decide a hypothesis.
"""
from __future__ import annotations

N_ASSETS = (8, 20, 50)
LOOKBACKS = (36, 60, 120)
CORR_PRIMARY = ("low", "moderate", "block", "high")
CORR_SENSITIVITY = ("random",)
TAILS_PRIMARY = ("gaussian",)
TAILS_STRESS = ("t4", "mix")
SEEDS_PRIMARY = tuple(range(30))
SEEDS_PILOT = tuple(range(9000, 9005))

NT_MIN = 0.40
STRATUM_CELLS = tuple(
    (n, t, c)
    for n in N_ASSETS
    for t in LOOKBACKS
    for c in CORR_PRIMARY
    if n / t >= NT_MIN
)

# H1 estimand is Frobenius. Relative MDE is vs sample-covariance Frobenius.
MDE_RELATIVE_FROBENIUS = 0.15
# H2 estimand is effective-number-of-bets(HRP) - effective-number-of-bets(sample MV).
MDE_EFFECTIVE_BETS = 1.0
# H3 sign-flip must clear this absolute net-Sharpe gap at both costs.
H3_SHARPE_FLOOR = 0.05
POWER_MIN = 0.80
POWER_DRAWS = 1000
# HOLDS also requires the point estimate to clear the MDE, so power is
# evaluated at twice the MDE. Power exactly at the MDE is reported and is
# not the gate.
POWER_ALTERNATIVE_MULTIPLE = 2.0

BOOT_SEED_REPS = 1000
BOOT_BLOCK = 6
BOOT_BLOCK_REPS = 1000

LINKAGE_CANDIDATES = ("single", "average")
LINKAGE_DEFAULT = "single"
INNER_MONTHS = 12
INNER_MIN_FIT = 24
INNER_METRIC = "realized_variance"
FACTORS_FIXED = 1
# No shrinkage grid. Ledoit-Wolf intensity is the analytic estimate.
SHRINKAGE_GRID = ()

WEIGHT_CAP = 0.25
SOLVER = "SLSQP"
SOLVER_FTOL = 1e-12
SOLVER_MAXITER = 500
FALLBACK = "equal_weight"

DETECTOR_FEATURE_MONTHS = 12
DETECTOR_QUANTILE = 0.80
DETECTOR_MIN_CALIBRATION = 12
STRESS_LOOKBACK = 24
H5_SHOCK_VAR_REDUCTION = 0.10
H5_CALM_VAR_INCREASE_MAX = 0.05

COSTS_BPS = (0, 5, 10, 20)
H3_COSTS_BPS = (0, 20)

RHO = {"low": 0.10, "moderate": 0.30, "high": 0.60}
BLOCK_INTRA = 0.60
BLOCK_INTER = 0.05
BLOCK_COUNT = 2
VOL_JUMP = 2.0
CORR_SPIKE = 0.70
SHOCK_LEN = 24
T4_NU = 4
MIX_PROB = 0.05
MIX_SCALE = 3.0
MIX_VAR_FACTOR = (1.0 - MIX_PROB) + MIX_PROB * (MIX_SCALE ** 2)

# Verification is a code path. It is not the decision sample.
VERIFY_N = (8, 20)
VERIFY_LOOKBACKS = (36, 60)
VERIFY_CORRS = CORR_PRIMARY + CORR_SENSITIVITY
VERIFY_TAILS = ("gaussian", "t4", "mix")
VERIFY_SEEDS = (0, 1, 2)
VERIFY_PERIODS = 96
VERIFY_COSTS = (0, 20)
VERIFY_BOOT = 300
