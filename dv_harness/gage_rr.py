"""Simulated crossed gage R&R (measurement system analysis), ANOVA method.

Design: `n_parts` parts x `n_operators` operators x `n_trials` repeat
trials, fully crossed (every operator measures every part `n_trials`
times). This is the standard AIAG MSA crossed gage R&R study shape.

Generative model (what `simulate_study` actually draws):

    measurement[p, o, t] = mu
                          + part_effect[p]              ~ N(0, sigma_part^2)
                          + operator_effect[o]           ~ N(0, sigma_operator^2)
                          + part_operator_interaction[p,o] ~ N(0, sigma_po^2)
                          + repeatability_noise[p,o,t]   ~ N(0, sigma_repeat^2)

ANOVA decomposition: `statsmodels.formula.api.ols` fits
`measurement ~ C(part) * C(operator)`, and `statsmodels.stats.anova_lm`
(type II) gives the part, operator, part:operator, and residual
(repeatability) mean squares. Variance components are recovered from
those mean squares with the standard AIAG ANOVA-method formulas:

    repeatability_var  = MS_error
    interaction_var    = max(0, (MS_po - MS_error) / n_trials)
    operator_var       = max(0, (MS_operator - MS_po) / (n_parts * n_trials))
    part_var           = max(0, (MS_part - MS_po) / (n_operators * n_trials))

    grr_var   = repeatability_var + operator_var + interaction_var
    total_var = grr_var + part_var
    pct_grr   = 100 * sqrt(grr_var / total_var)
    ndc       = floor(1.41 * sqrt(part_var) / sqrt(grr_var))   # number of distinct categories

Why the ANOVA method rather than the classical range method: the range
method (AIAG "short method", using d2*-scaled ranges) cannot separate the
part*operator interaction from repeatability at all, so a measurement
system whose failure mode is specifically "operators are inconsistent on
some parts but not others" (interaction-dominated) is invisible to it by
construction. This project's seeded failure scenarios deliberately
include an interaction-dominated failure mode (see gage_rr_scenarios.py),
so the ANOVA method, which estimates the interaction term directly from
its own line in the ANOVA table, is required, not just preferred.

Acceptance rule: %GRR <= 30 AND NDC >= 5. AIAG MSA (4th edition) treats
%GRR < 10% as acceptable, 10-30% as conditionally acceptable depending on
application, and > 30% as unacceptable; this project uses the outer 30%
boundary as the pass/fail line (a stricter 10% line would also flag some
of the "conditionally acceptable" real-world measurement systems as
failing, which is not what "engineered to fail" means here) plus the
companion NDC >= 5 rule (also AIAG), which independently catches poor
part-to-part discrimination even when %GRR alone is borderline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

PCT_GRR_THRESHOLD = 30.0
NDC_THRESHOLD = 5


def simulate_study(n_parts: int, n_operators: int, n_trials: int,
                    sigma_part: float, sigma_operator: float, sigma_po: float,
                    sigma_repeat: float, mu: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Simulate one crossed gage R&R study and return a long-format
    DataFrame with columns part, operator, trial, measurement."""
    rng = np.random.default_rng(seed)
    part_effect = rng.normal(0.0, sigma_part, size=n_parts)
    operator_effect = rng.normal(0.0, sigma_operator, size=n_operators)
    po_interaction = rng.normal(0.0, sigma_po, size=(n_parts, n_operators))

    rows = []
    for p in range(n_parts):
        for o in range(n_operators):
            for t in range(n_trials):
                noise = rng.normal(0.0, sigma_repeat)
                y = mu + part_effect[p] + operator_effect[o] + po_interaction[p, o] + noise
                rows.append({"part": f"P{p:02d}", "operator": f"O{o}", "trial": t, "measurement": y})
    return pd.DataFrame(rows)


def run_anova(df: pd.DataFrame) -> pd.DataFrame:
    """Fit part * operator two-way ANOVA (type II) and return the
    statsmodels anova table (sum_sq, df, F, PR(>F) per term)."""
    model = ols("measurement ~ C(part) * C(operator)", data=df).fit()
    return anova_lm(model, typ=2)


@dataclass(frozen=True)
class GageRRResult:
    n_parts: int
    n_operators: int
    n_trials: int
    repeatability_var: float
    operator_var: float
    interaction_var: float
    part_var: float
    grr_var: float
    total_var: float
    pct_grr: float
    ndc: int
    passed: bool
    reasons: tuple


def compute_variance_components(anova_table: pd.DataFrame, n_parts: int, n_operators: int,
                                 n_trials: int) -> GageRRResult:
    ms_part = anova_table.loc["C(part)", "sum_sq"] / anova_table.loc["C(part)", "df"]
    ms_operator = anova_table.loc["C(operator)", "sum_sq"] / anova_table.loc["C(operator)", "df"]
    ms_po = anova_table.loc["C(part):C(operator)", "sum_sq"] / anova_table.loc["C(part):C(operator)", "df"]
    ms_error = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]

    repeatability_var = max(0.0, ms_error)
    interaction_var = max(0.0, (ms_po - ms_error) / n_trials)
    operator_var = max(0.0, (ms_operator - ms_po) / (n_parts * n_trials))
    part_var = max(0.0, (ms_part - ms_po) / (n_operators * n_trials))

    grr_var = repeatability_var + operator_var + interaction_var
    total_var = grr_var + part_var
    pct_grr = 100.0 * math.sqrt(grr_var / total_var) if total_var > 0 else float("inf")

    if grr_var > 0:
        ndc = math.floor(1.41 * math.sqrt(part_var) / math.sqrt(grr_var))
    else:
        ndc = 10 ** 6  # perfect repeatability: not a realistic case, treat as unbounded discrimination

    reasons = []
    if pct_grr > PCT_GRR_THRESHOLD:
        reasons.append(f"%GRR={pct_grr:.2f} exceeds threshold {PCT_GRR_THRESHOLD}")
    if ndc < NDC_THRESHOLD:
        reasons.append(f"NDC={ndc} below threshold {NDC_THRESHOLD}")
    passed = len(reasons) == 0

    return GageRRResult(
        n_parts=n_parts, n_operators=n_operators, n_trials=n_trials,
        repeatability_var=repeatability_var, operator_var=operator_var,
        interaction_var=interaction_var, part_var=part_var,
        grr_var=grr_var, total_var=total_var, pct_grr=pct_grr, ndc=ndc,
        passed=passed, reasons=tuple(reasons),
    )


def evaluate_scenario(n_parts: int, n_operators: int, n_trials: int,
                       sigma_part: float, sigma_operator: float, sigma_po: float,
                       sigma_repeat: float, seed: int) -> GageRRResult:
    """Convenience wrapper: simulate + fit ANOVA + compute components + classify."""
    df = simulate_study(n_parts, n_operators, n_trials, sigma_part, sigma_operator,
                         sigma_po, sigma_repeat, seed=seed)
    anova_table = run_anova(df)
    return compute_variance_components(anova_table, n_parts, n_operators, n_trials)
