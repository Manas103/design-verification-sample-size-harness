"""Nested (hierarchical) gage R&R for destructive tests, where a specimen
cannot be remeasured.

The crossed design in `gage_rr.py` requires every operator to measure
every part; that is impossible when the act of measuring destroys the
part (pull-to-failure, burst, peel, cut-and-etch). The standard AIAG
answer for this case is a nested design: each part (a lot or batch,
represented here by several nominally-identical specimens) is assigned to
exactly one operator, never shared, and repeatability is estimated from
`n_trials` replicate specimens drawn from that same part rather than
repeat measurements of the same physical specimen.

Design: `n_operators` operators x `n_parts_per_operator` parts, each part
unique to its operator (never crossed), x `n_trials` replicate specimens
per part.

Generative model (what `simulate_nested_study` actually draws):

    measurement[o, p, t] = mu
                          + operator_effect[o]  ~ N(0, sigma_operator^2)
                          + part_effect[o, p]   ~ N(0, sigma_part^2)
                          + repeatability_noise[o, p, t] ~ N(0, sigma_repeat^2)

Because no part is ever measured by more than one operator, there is no
data from which to estimate a part*operator interaction term at all; any
real interaction effect is structurally confounded into the operator
term. This is not a limitation of the ANOVA method used here, it is a
property of the design itself, and is the reason nested gage R&R studies
are understood to be less diagnostic than crossed ones, and are only used
when the crossed design is physically impossible (see Limitations).

ANOVA decomposition: computed directly from the classical nested-ANOVA
sum-of-squares decomposition (operator means, part-within-operator means,
grand mean), not through statsmodels' `C(a) + C(b)` formula interface.
That was the first thing tried here, and it does not work: once
`part_within_operator` is coded as a globally unique label per
(operator, part) pair, its treatment-coded dummy columns already fully
determine operator membership, so `C(operator) + C(part_in_operator)` is
rank-deficient (statsmodels raises `SingularMatrixWarning` and silently
returns a non-unique, wrong coefficient solution rather than failing
loudly; see Findings). The hand-computed sum-of-squares below has no such
ambiguity because it never forms an overparametrized joint design matrix:

    repeatability_var = MS_error
    part_var          = max(0, (MS_part_in_op - MS_error) / n_trials)
    operator_var       = max(0, (MS_operator - MS_part_in_op) / (n_parts_per_operator * n_trials))

    grr_var   = repeatability_var + operator_var
    total_var = grr_var + part_var
    pct_grr   = 100 * sqrt(grr_var / total_var)
    ndc       = floor(1.41 * sqrt(part_var) / sqrt(grr_var))

Acceptance rule: the same AIAG %GRR <= 30 AND NDC >= 5 bright line used by
the crossed study in `gage_rr.py`, applied to the nested variance
components. Using the same numeric thresholds for both designs is a
deliberate choice: the nested design is a different way of estimating the
same repeatability/reproducibility-vs-part-variation ratio, not a
different acceptance standard.

Operator-term pooling (found necessary, see Findings). With only
`n_operators` levels, the operator mean square has `n_operators - 1`
degrees of freedom, which is a genuinely noisy chi-squared estimate: a
first implementation that subtracted MS_part_in_op from MS_operator and
clamped only at zero from below produced false failures on a healthy
(true operator variance exactly zero) measurement system at roughly the
rate an unguarded point-estimate difference implies, regardless of how
many parts or trials were added, because the noise lives in the
numerator's small degrees of freedom, not in the sample size. The fix
used here is the same one AIAG MSA's ANOVA method already applies to a
non-significant part*operator interaction term: pool a term into the
next-lower level of the hierarchy unless its F-ratio against the next
term down is significant at a stated alpha. `ALPHA_POOL = 0.05` is used
here (stricter than AIAG's usual 0.25 for the interaction term, because
this design has no interaction term to also catch real problems, so the
operator term is the only thing standing between a real reproducibility
problem and a value of zero; a looser alpha measurably increased the
false-flag rate in testing, see Findings).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

PCT_GRR_THRESHOLD = 30.0
NDC_THRESHOLD = 5
ALPHA_POOL = 0.05


def simulate_nested_study(n_operators: int, n_parts_per_operator: int, n_trials: int,
                           sigma_part: float, sigma_operator: float, sigma_repeat: float,
                           mu: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Simulate one nested (destructive-test) gage R&R study and return a
    long-format DataFrame with columns operator, part_in_operator, trial,
    measurement. `part_in_operator` is globally unique per (operator,
    part) pair, i.e. genuinely nested, not crossed."""
    rng = np.random.default_rng(seed)
    operator_effect = rng.normal(0.0, sigma_operator, size=n_operators)
    part_effect = rng.normal(0.0, sigma_part, size=(n_operators, n_parts_per_operator))

    rows = []
    for o in range(n_operators):
        for p in range(n_parts_per_operator):
            for t in range(n_trials):
                noise = rng.normal(0.0, sigma_repeat)
                y = mu + operator_effect[o] + part_effect[o, p] + noise
                rows.append({
                    "operator": f"O{o}",
                    "part_in_operator": f"O{o}_P{p:02d}",
                    "trial": t,
                    "measurement": y,
                })
    return pd.DataFrame(rows)


def run_nested_anova(df: pd.DataFrame) -> dict:
    """Hand-computed nested ANOVA table: operator, part(operator), and
    residual (repeatability) sums of squares, degrees of freedom, and mean
    squares, from the classical hierarchical decomposition:

        SS_operator      = n_trials * n_parts_per_operator * sum_o (mean_o - grand_mean)^2
        SS_part_in_op    = n_trials * sum_{o,p} (mean_op - mean_o)^2
        SS_error         = sum_{o,p,t} (y_opt - mean_op)^2

    Each measurement contributes to exactly one term, so
    SS_operator + SS_part_in_op + SS_error == the total corrected sum of
    squares, by construction (verified in tests)."""
    grand_mean = df["measurement"].mean()

    operator_means = df.groupby("operator")["measurement"].mean()
    part_means = df.groupby(["operator", "part_in_operator"])["measurement"].mean()
    n_per_operator = df.groupby("operator")["measurement"].count()
    n_per_part = df.groupby(["operator", "part_in_operator"])["measurement"].count()

    n_operators = df["operator"].nunique()
    n_parts_total = df["part_in_operator"].nunique()
    n_obs = len(df)

    ss_operator = sum(n_per_operator[o] * (operator_means[o] - grand_mean) ** 2 for o in operator_means.index)
    df_operator = n_operators - 1

    ss_part_in_op = 0.0
    for (o, p), mean_op in part_means.items():
        ss_part_in_op += n_per_part[(o, p)] * (mean_op - operator_means[o]) ** 2
    df_part_in_op = n_parts_total - n_operators

    merged = df.merge(
        part_means.rename("part_mean"), left_on=["operator", "part_in_operator"], right_index=True
    )
    ss_error = float(((merged["measurement"] - merged["part_mean"]) ** 2).sum())
    df_error = n_obs - n_parts_total

    return {
        "ss_operator": ss_operator, "df_operator": df_operator,
        "ss_part_in_op": ss_part_in_op, "df_part_in_op": df_part_in_op,
        "ss_error": ss_error, "df_error": df_error,
    }


@dataclass(frozen=True)
class NestedGageRRResult:
    n_operators: int
    n_parts_per_operator: int
    n_trials: int
    repeatability_var: float
    operator_var: float
    part_var: float
    grr_var: float
    total_var: float
    pct_grr: float
    ndc: int
    passed: bool
    reasons: tuple
    operator_term_pooled: bool
    operator_f_pvalue: float


def compute_nested_variance_components(anova_table: dict, n_operators: int,
                                        n_parts_per_operator: int, n_trials: int) -> NestedGageRRResult:
    ms_operator = anova_table["ss_operator"] / anova_table["df_operator"]
    ms_part_in_op = anova_table["ss_part_in_op"] / anova_table["df_part_in_op"]
    ms_error = anova_table["ss_error"] / anova_table["df_error"]

    f_operator = ms_operator / ms_part_in_op if ms_part_in_op > 0 else float("inf")
    operator_f_pvalue = float(stats.f.sf(f_operator, anova_table["df_operator"], anova_table["df_part_in_op"]))
    operator_significant = operator_f_pvalue < ALPHA_POOL

    repeatability_var = max(0.0, ms_error)
    part_var = max(0.0, (ms_part_in_op - ms_error) / n_trials)
    if operator_significant:
        operator_var = max(0.0, (ms_operator - ms_part_in_op) / (n_parts_per_operator * n_trials))
    else:
        operator_var = 0.0

    grr_var = repeatability_var + operator_var
    total_var = grr_var + part_var
    pct_grr = 100.0 * math.sqrt(grr_var / total_var) if total_var > 0 else float("inf")

    if grr_var > 0:
        ndc = math.floor(1.41 * math.sqrt(part_var) / math.sqrt(grr_var))
    else:
        ndc = 10 ** 6

    reasons = []
    if pct_grr > PCT_GRR_THRESHOLD:
        reasons.append(f"%GRR={pct_grr:.2f} exceeds threshold {PCT_GRR_THRESHOLD}")
    if ndc < NDC_THRESHOLD:
        reasons.append(f"NDC={ndc} below threshold {NDC_THRESHOLD}")
    passed = len(reasons) == 0

    return NestedGageRRResult(
        n_operators=n_operators, n_parts_per_operator=n_parts_per_operator, n_trials=n_trials,
        repeatability_var=repeatability_var, operator_var=operator_var, part_var=part_var,
        grr_var=grr_var, total_var=total_var, pct_grr=pct_grr, ndc=ndc,
        passed=passed, reasons=tuple(reasons),
        operator_term_pooled=(not operator_significant), operator_f_pvalue=operator_f_pvalue,
    )


def evaluate_nested_scenario(n_operators: int, n_parts_per_operator: int, n_trials: int,
                              sigma_part: float, sigma_operator: float, sigma_repeat: float,
                              seed: int) -> NestedGageRRResult:
    """Convenience wrapper: simulate + fit nested ANOVA + compute components + classify."""
    df = simulate_nested_study(n_operators, n_parts_per_operator, n_trials,
                                sigma_part, sigma_operator, sigma_repeat, seed=seed)
    anova_table = run_nested_anova(df)
    return compute_nested_variance_components(anova_table, n_operators, n_parts_per_operator, n_trials)
