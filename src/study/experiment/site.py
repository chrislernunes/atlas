"""Write the GitHub Pages note from a verification directory."""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from study.protocol import (
    H3_SHARPE_FLOOR,
    H5_CALM_VAR_INCREASE_MAX,
    H5_SHOCK_VAR_REDUCTION,
    INNER_METRIC,
    INNER_MIN_FIT,
    INNER_MONTHS,
    LINKAGE_CANDIDATES,
    MDE_EFFECTIVE_BETS,
    MDE_RELATIVE_FROBENIUS,
    POWER_ALTERNATIVE_MULTIPLE,
    POWER_MIN,
    SEEDS_PRIMARY,
    STRATUM_CELLS,
    WEIGHT_CAP,
)


COV_METHODS = ["const_corr_mv", "factor_mv", "hrp", "lw_mv", "sample_mv"]
X_URL = "https://x.com/chrislernunes"
LINKEDIN_URL = "https://www.linkedin.com/in/chrisler-nunes-335333161"

_SOCIAL = """
<nav class="social" aria-label="Profiles">
  <a href="https://x.com/chrislernunes" aria-label="X">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.7 10.3 22.1 2h-1.8l-6.4 7.2L8.8 2H2.2l7.8 11L2 22h1.8l6.8-7.7L15.1 22h6.6l-8-11.7Zm-2.4 2.7-.8-1.1L4.6 3.3h2.7l5.1 7 .8 1.1 6.6 9.2h-2.7l-5.4-7.6Z"/></svg>
  </a>
  <a href="https://www.linkedin.com/in/chrisler-nunes-335333161" aria-label="LinkedIn">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.98 3.5A2.5 2.5 0 1 1 2.5 6a2.5 2.5 0 0 1 2.48-2.5ZM3 8.5H6.5V21H3Zm6 0h3.4v1.7h.05c.47-.9 1.62-1.85 3.34-1.85 3.57 0 4.23 2.35 4.23 5.4V21H16.5v-5.55c0-1.32-.02-3.02-1.84-3.02-1.84 0-2.12 1.44-2.12 2.92V21H9Z"/></svg>
  </a>
</nav>
"""


def _cell(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        number = float(value)
        if digits == 0:
            return f"{number:.0f}"
        return f"{number:.{digits}f}"
    return str(value)


def _table(frame: pd.DataFrame, digits: dict[str, int] | None = None) -> str:
    digits = digits or {}
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for col, value in row.items():
            cells.append(f"<td>{html.escape(_cell(value, digits.get(col, 4)))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def _charts(panel: pd.DataFrame, out: Path) -> None:
    panel = panel.copy()
    panel["shock"] = panel["shock"].fillna("")
    panel["detector"] = panel["detector"].astype(bool)
    base = panel[(panel["innovation"] == "gaussian") & (panel["shock"] == "") & (~panel["detector"])]
    est = base[base.method.isin(["sample_mv", "lw_mv", "const_corr_mv", "factor_mv"])]
    g = est.groupby(["n_assets", "lookback", "method"], as_index=False).frobenius.mean()
    g["nt"] = g.n_assets / g.lookback
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    for method, sub in g.groupby("method"):
        sub = sub.sort_values("nt")
        ax.plot(sub.nt, sub.frobenius, marker="o", label=method)
    ax.set_xlabel("N / T")
    ax.set_ylabel("Mean Frobenius error")
    ax.set_title("Verification sample. Not a decision.")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "frobenius.png")
    plt.close(fig)

    bets = base[base.method.isin(["hrp", "sample_mv", "equal_weight"])]
    b = bets.groupby(["n_assets", "method"], as_index=False).eff_bets.mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    methods = ["equal_weight", "hrp", "sample_mv"]
    ns = sorted(b.n_assets.unique())
    width = 0.25
    for i, method in enumerate(methods):
        vals = [float(b[(b.n_assets == n) & (b.method == method)].eff_bets.mean()) for n in ns]
        ax.bar([x + i * width for x in range(len(ns))], vals, width=width, label=method)
    ax.set_xticks([x + width for x in range(len(ns))], [str(n) for n in ns])
    ax.set_xlabel("N")
    ax.set_ylabel("Effective number of bets")
    ax.set_title("Verification sample. Cap is 0.25.")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "bets.png")
    plt.close(fig)


def write_site(verification: Path, dest: Path) -> None:
    panel = pd.read_csv(verification / "verification_panel.csv")
    panel["shock"] = panel["shock"].fillna("")
    panel["detector"] = panel["detector"].astype(bool)
    decision = pd.read_csv(verification / "decision_record.csv", index_col=0).squeeze()
    dest.mkdir(parents=True, exist_ok=True)
    _charts(panel, dest)

    base = panel[(panel["innovation"] == "gaussian") & (panel["shock"] == "") & (~panel["detector"])]
    err = (
        base[base.method.isin(COV_METHODS)]
        .groupby("method")[["frobenius", "eig_l1", "risk_abs", "eff_bets", "turnover", "failed_frac"]]
        .mean()
        .reindex(COV_METHODS)
        .reset_index()
    )
    weights = (
        base.groupby("method")[["eff_bets", "max_weight", "turnover", "failed_frac"]]
        .mean()
        .reindex(["equal_weight", "inverse_vol", *COV_METHODS])
        .reset_index()
    )
    tail_src = panel[(panel["shock"] == "") & (~panel["detector"]) & (panel["n_assets"] == 20) & (panel["lookback"] == 36) & (panel["corr_type"] == "moderate") & (panel["method"].isin(COV_METHODS))]
    tail = tail_src.groupby(["innovation", "method"]).frobenius.mean().unstack("method").reindex(columns=COV_METHODS).reset_index()
    shock = panel[(panel["n_assets"] == 20) & (panel["lookback"] == 36) & (panel["corr_type"] == "moderate") & (panel["innovation"] == "gaussian") & (panel["method"] == "lw_mv")]
    shock_tbl = shock.groupby(["shock", "detector"]).agg(variance=("var", "mean"), frobenius=("frobenius", "mean")).reset_index()
    shock_tbl["shock"] = shock_tbl["shock"].replace({"": "none"})
    shock_tbl["detector"] = shock_tbl["detector"].map({True: "on", False: "off"})

    cells = ", ".join(f"N={n}, T={t}, {c}" for n, t, c in STRATUM_CELLS)
    def n(key, spec):
        return format(float(decision[key]), spec)

    dec_rows = pd.DataFrame(
        [
            ["H1", str(decision["h1_status"])],
            ["Theta, Frobenius", n("theta_h1", ".6f")],
            ["5th percentile", n("theta_h1_p05", ".6f")],
            ["95th percentile", n("theta_h1_p95", ".6f")],
            ["Relative to sample", f"{float(decision['relative_h1']):.1%}"],
            ["Sample Frobenius", n("mean_sample_frobenius", ".5f")],
            ["Pilot sigma", n("pilot_sigma", ".6f")],
            ["Absolute minimum effect", n("mde_abs", ".5f")],
            ["Power at minimum effect", n("power_at_mde", ".2f")],
            ["Power at twice the minimum effect", n("power_at_design_alternative", ".2f")],
            ["H2", str(decision["h2_status"])],
            ["Theta, effective bets", n("theta_h2", ".2f")],
            ["H2 5th percentile", n("theta_h2_p05", ".2f")],
            ["H2 95th percentile", n("theta_h2_p95", ".2f")],
            ["H3 reversals", str(int(float(decision["h3_reversal_cells"])))],
            ["Stratum cells", f"{int(float(decision['stratum_cells_present']))} / {int(float(decision['stratum_cells_required']))}"],
            ["Seeds", f"{int(float(decision['seeds_present']))} / {int(float(decision['seeds_required']))}"],
        ],
        columns=["quantity", "value"],
    )
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Atlas protocol</title>
<style>
body {{ background:#fff; color:#111; font: 17px/1.55 Georgia, "Iowan Old Style", Palatino, serif; max-width: 860px; margin: 0 auto; padding: 0 1.2rem 3rem; }}
header, footer {{ display: flex; align-items: center; justify-content: space-between; padding: 1.1rem 0; }}
header {{ border-bottom: 1px solid #111; }}
footer {{ border-top: 1px solid #ddd; margin-top: 2.5rem; color: #444; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px; }}
.mark {{ font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px; letter-spacing: 0.14em; text-transform: uppercase; }}
.social {{ display: flex; gap: 0.85rem; }}
.social a {{ display: flex; color: #111; }}
.social svg {{ width: 18px; height: 18px; fill: currentColor; }}
h1,h2,h3 {{ font-weight: 500; letter-spacing: -0.02em; }}
h1 {{ font-size: 1.7rem; margin-top: 1.6rem; }}
a {{ color: #111; }}
code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.86rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px; font-variant-numeric: tabular-nums; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
img {{ width: 100%; height: auto; }}
.note {{ border-bottom: 1px solid #111; padding: 0 0 0.8rem; }}
.caption {{ font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px; color: #444; margin: -0.6rem 0 1.2rem; }}
</style>
</head>
<body>
<header>
  <div class="mark">Atlas</div>
  {_SOCIAL}
</header>
<h1>Hierarchical allocation under estimation error</h1>
<p class="note">Synthetic data only. The numbers below are the verification sample in this repository. They are not a hypothesis decision. The verification label is <code>{html.escape(str(decision["h1_status"]))}</code>.</p>

<h2>What is being estimated</h2>
<p>For each locked cell and seed, Δ is the paired difference of mean out-of-sample Frobenius error: sample covariance minus Ledoit–Wolf. The cell summary is the median across seeds. The primary estimand is the N/T-weighted mean of those medians on the stratum N/T ≥ 0.40:</p>
<pre>Θ_H1 = Σ_{{c in stratum}} (N/T)_c · median_s(Δ_{{c,s}}) / Σ_{{c in stratum}} (N/T)_c</pre>
<p>The stratum is fixed: {html.escape(cells)}. Sixteen cells. Seeds 0–29 are the decision sample. Seeds 9000–9004 are the pilot and are not reused.</p>
<p>Uncertainty for Θ is a seed bootstrap: resample seeds inside each cell, recompute medians, recompute Θ. A block bootstrap of a single path is not the test. Block length 6 is used only for within-path Sharpe intervals.</p>
<p>The minimum practical effect is a {MDE_RELATIVE_FROBENIUS:.0%} reduction in sample-covariance Frobenius. <code>HOLDS</code> requires the 5th percentile of Θ to be positive and the point estimate to clear that 15% bar. Power of this compound rule is computed at {POWER_ALTERNATIVE_MULTIPLE:.0f}× the minimum effect, under a Gaussian homoskedastic model fit to the pilot. The target is {POWER_MIN:.0%}. Power exactly at the 15% boundary is reported and is not the gate: at the boundary the point-estimate rule rejects about half the time even when the interval excludes zero. If pilot variance understates the N=50 cells, reported power is an upper bound. If design power is below {POWER_MIN:.0%}, a complete sample is still <code>INCONCLUSIVE_UNDERPOWERED</code>.</p>
<p>A 70% cell-count vote is not used. It had no power calculation and it can be passed by many tiny effects.</p>

<h2>Hyperparameters that are not free</h2>
<p>Linkage candidates are exactly {LINKAGE_CANDIDATES}. The inner window is the last {INNER_MONTHS} months of the training sample. The fit window must be at least {INNER_MIN_FIT} months, otherwise linkage is <code>single</code> and the skip is recorded. The inner metric is {INNER_METRIC}. Ties go to <code>single</code>. Ledoit–Wolf has no shrinkage grid; the intensity is the analytic estimate and is stored, not chosen. The factor estimator keeps {1} principal component. Lookbacks {36, 60, 120} are experimental factors, not tuned parameters.</p>

<h2>Constraints</h2>
<p>Every method, including HRP, is projected onto the same set: weights sum to 1, weights are non-negative, and no weight exceeds max({WEIGHT_CAP}, 1/N). The quadratic programs use SLSQP, ftol 1e-12, maxiter 500, with those bounds. If the solver fails or the projection is non-finite, the weight is equal-weight and <code>optimizer_failed</code> is true. The cap rule is feasible for every N in the grid because N·cap ≥ 1.</p>

<h2>Regime detector</h2>
<p>This is not a hidden state model. On the training sample only, 12-month mean volatility and mean absolute correlation are computed. Quantiles at 0.80 are taken from windows that end before the decision window. Stress is declared if either current feature exceeds its quantile. Otherwise the state is calm. If fewer than 12 calibration points exist, the state is calm. Stress shortens the covariance window to the last 24 months. Calm uses the full lookback. Full-sample smoothed probabilities are not computed.</p>
<p>H5 is a comparison of Ledoit–Wolf minimum variance with that switch against the same estimator with the switch off. It is supported only if shocked cells show a variance reduction of at least {H5_SHOCK_VAR_REDUCTION:.0%} and unshocked cells do not show a variance increase above {H5_CALM_VAR_INCREASE_MAX:.0%}. H5 is not folded into Θ_H1.</p>

<h2>Other hypotheses</h2>
<p>H2 is the unweighted mean, across stratum cells, of the seed-median gap in effective number of bets (HRP minus sample minimum variance) at zero cost. The practical bar is +{MDE_EFFECTIVE_BETS:.0f} bet. H3 is not a performance claim. A reversal is a sign change in the seed-median net-Sharpe gap between HRP and sample minimum variance at 0 bps and at 20 bps, and both gaps must clear {H3_SHARPE_FLOOR} in absolute value.</p>

<h2>What the claims are conditional on</h2>
<p>Θ_H1 uses Gaussian draws and only the four named correlation structures (equicorrelation 0.10, 0.30, 0.60, and a two-block matrix). It does not include the random full-rank correlation, the t distribution with 4 degrees of freedom, or the scale mixture (probability 0.05 of multiplying the draw by 3; unconditional variance factor 1.40). Those are stresses. A result that changes sign there limits the claim to the primary design. It is not averaged back into Θ after being seen.</p>

<h2>Verification record</h2>
<p>Decision seeds required: {len(SEEDS_PRIMARY)}. This run has {decision["seeds_present"]} and {decision["stratum_cells_present"]} of {decision["stratum_cells_required"]} stratum cells. The status function is required to return <code>NOT_A_DECISION_SAMPLE</code>. On the cells that were run, the partial Θ is {float(decision["relative_h1"]):.1%} of sample Frobenius, against a 15% bar, and the seed-bootstrap 5th percentile is {float(decision["theta_h1_p05"]):.6f}. That interval includes non-positive values. Power at the minimum effect is {float(decision["power_at_mde"]):.3f}. Power at twice the minimum effect is {float(decision["power_at_design_alternative"]):.3f} under the pilot variance; that second number is not evidence about N=50.</p>
{_table(dec_rows)}
<h3>Covariance error, Gaussian paths, no injected shock</h3>
<p class="caption">Equal weight and inverse volatility do not form a covariance, so they are not in this table.</p>
{_table(err, {"frobenius": 4, "eig_l1": 4, "risk_abs": 4, "eff_bets": 2, "turnover": 4, "failed_frac": 2})}
<img src="frobenius.png" alt="Frobenius error against N/T">
<img src="bets.png" alt="Effective number of bets">
<h3>Weights, same paths</h3>
{_table(weights, {"eff_bets": 2, "max_weight": 2, "turnover": 4, "failed_frac": 2})}
<h3>Tail stress, N=20, T=36, moderate correlation</h3>
<p class="caption">Frobenius error. Covariance methods only.</p>
{_table(tail, {name: 4 for name in COV_METHODS})}
<h3>Ledoit–Wolf minimum variance under an injected shock</h3>
{_table(shock_tbl, {"variance": 4, "frobenius": 4})}
<footer>
  <div>Constants in <code>src/study/protocol.py</code>. <a href="https://github.com/chrislernunes/atlas">Repository</a>.</div>
  {_SOCIAL}
</footer>
</body>
</html>
"""
    (dest / "index.html").write_text(page)


if __name__ == "__main__":
    write_site(Path("outputs/verification"), Path("site"))
