"""Measurement uncertainty for the wafer-shape metrology in wafer_shape.py,
sized with a simulated gage R&R study, plus a model-to-measurement decision
rule that refuses to call a model validated without enough repeats.

This module reuses the existing crossed-design gage R&R machinery in
gage_rr.py (simulate a parts x operators x trials study, fit
`measurement ~ C(part) * C(operator)` in statsmodels, decompose the ANOVA
mean squares into repeatability/operator/interaction/part variance
components) rather than reimplementing it: `run_anova` and
`compute_variance_components` from gage_rr.py are called directly on a
DataFrame built here. What is new is what "one measurement" means: instead
of a single scalar reading, one measurement is one simulated noisy
interferometer scan of a wafer, fit with `wafer_shape.fit_shape_lstsq`,
reduced to the fitted bow (or cyl, or saddle) coefficient. One scan yields
all three shape terms at once (a single fit produces the whole
ShapeCoeffs), so one simulated study produces a per-term %GRR/NDC/variance
report for bow, cylindrical and saddle simultaneously from the same
underlying scans, which is both more efficient and more physically honest
than drawing three independent studies.

Generative model for one simulated scan
----------------------------------------
    true height map   = the part's true ShapeCoeffs
                       + an operator-specific bias in (bow, cyl, saddle)
                         only [operator_bias ~ N(0, sigma_operator^2) per
                         term per operator, held fixed across that
                         operator's trials -- representing a systematic
                         reference-surface/calibration difference between
                         instruments or operators in the shape channels
                         specifically]
    simulated scan     = true height map sampled on the measurement grid
                       + i.i.d. per-point noise ~ N(0, sigma_repeat_point^2)
                         [repeatability: interferometer/instrument noise,
                         independent on every trial]
    measurement        = wafer_shape.fit_shape_lstsq(scan)

Tip, tilt and piston are not given an operator bias term: a real wafer
chuck's kinematic mount references the same three points on every load, so
the rigid-body terms are assumed comparably well controlled across
operators; this is a stated simplifying assumption, not a measured fact
about any real tool.

Decision rule (claim: "refuses to call a model validated without enough
repeats")
----------------------------------------------------------------------
A gage R&R study's repeatability variance estimate comes from the ANOVA
residual mean square, whose degrees of freedom are
`n_parts * n_operators * (n_trials - 1)`. A mean square with few degrees
of freedom is a high-variance estimate of the true measurement noise
regardless of the magnitude of that noise (this is exactly the failure
mode documented in this project's original gage_rr Findings entry, where a
3-operator study gave the operator mean square only 2 degrees of freedom).
`MIN_RESIDUAL_DF` is the minimum residual degrees of freedom this project
is willing to trust for that estimate; a study below it is refused outright
(`INSUFFICIENT_DATA`), not silently allowed to produce a "validated"
verdict on a noisy variance estimate. This threshold is chosen, not
discovered: at `MIN_RESIDUAL_DF = 15`, the relative standard error of a
chi-squared-based variance estimate is `sqrt(2/15) ~= 36.5%`, which is
already generous; anything looser than this was judged to defeat the
point of gating on repeats at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dv_harness.gage_rr import GageRRResult, compute_variance_components, run_anova
from dv_harness.wafer_shape import (
    SHAPE_TERM_NAMES,
    ShapeCoeffs,
    fit_shape_lstsq,
    make_measurement_grid,
    synth_height_map,
)

MIN_RESIDUAL_DF = 15


@dataclass(frozen=True)
class WaferPart:
    name: str
    coeffs: ShapeCoeffs


def make_parts(n_parts: int, sigma_part: dict, seed: int) -> list[WaferPart]:
    """Generate n_parts simulated wafers with randomized true shape
    coefficients, representing ordinary lot-to-lot wafer-shape variation.
    sigma_part gives the part-to-part standard deviation per term (mm)."""
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_parts):
        coeffs = ShapeCoeffs(
            piston=rng.normal(0.0, sigma_part.get("piston", 0.01)),
            tip=rng.normal(0.0, sigma_part.get("tip", 0.005)),
            tilt=rng.normal(0.0, sigma_part.get("tilt", 0.005)),
            bow=rng.normal(0.0, sigma_part.get("bow", 0.020)),
            cyl=rng.normal(0.0, sigma_part.get("cyl", 0.010)),
            saddle=rng.normal(0.0, sigma_part.get("saddle", 0.010)),
        )
        parts.append(WaferPart(name=f"W{i:02d}", coeffs=coeffs))
    return parts


def simulate_one_scan(part: WaferPart, operator_bias: dict, sigma_repeat_point: float,
                       x: np.ndarray, y: np.ndarray, radius: float, rng: np.random.Generator) -> ShapeCoeffs:
    """One simulated scan of `part` by one operator/instrument: the part's
    true shape, plus that operator's fixed bow/cyl/saddle bias, plus fresh
    per-point repeatability noise, fit back to ShapeCoeffs."""
    biased = part.coeffs.with_term_offset("bow", operator_bias["bow"]) \
                         .with_term_offset("cyl", operator_bias["cyl"]) \
                         .with_term_offset("saddle", operator_bias["saddle"])
    z = synth_height_map(biased, x, y, radius, noise_sigma=sigma_repeat_point, rng=rng)
    return fit_shape_lstsq(x, y, z, radius)


def build_multiresponse_frame(parts: list, n_operators: int, n_trials: int, sigma_operator: float,
                               sigma_repeat_point: float, radius: float, n_per_axis: int,
                               seed: int) -> pd.DataFrame:
    """Simulate the full crossed parts x operators x trials study and return
    a long-format DataFrame with columns part, operator, trial, bow, cyl,
    saddle: one row per simulated scan, all three shape terms from the same
    fit."""
    rng = np.random.default_rng(seed)
    x, y = make_measurement_grid(n_per_axis=n_per_axis, radius=radius)

    operator_biases = {}
    for o in range(n_operators):
        operator_biases[o] = {
            "bow": rng.normal(0.0, sigma_operator),
            "cyl": rng.normal(0.0, sigma_operator),
            "saddle": rng.normal(0.0, sigma_operator),
        }

    rows = []
    for part in parts:
        for o in range(n_operators):
            for t in range(n_trials):
                fitted = simulate_one_scan(part, operator_biases[o], sigma_repeat_point, x, y, radius, rng)
                rows.append({
                    "part": part.name, "operator": f"O{o}", "trial": t,
                    "bow": fitted.bow, "cyl": fitted.cyl, "saddle": fitted.saddle,
                })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ShapeGageRRStudy:
    n_parts: int
    n_operators: int
    n_trials: int
    residual_df: int
    results: dict  # term -> GageRRResult


def residual_degrees_of_freedom(n_parts: int, n_operators: int, n_trials: int) -> int:
    return n_parts * n_operators * (n_trials - 1)


def repeats_are_sufficient(n_parts: int, n_operators: int, n_trials: int,
                            min_residual_df: int = MIN_RESIDUAL_DF) -> bool:
    return residual_degrees_of_freedom(n_parts, n_operators, n_trials) >= min_residual_df


def run_shape_gage_rr_study(frame: pd.DataFrame, n_parts: int, n_operators: int,
                             n_trials: int) -> ShapeGageRRStudy:
    """For each of bow/cyl/saddle, reuse gage_rr.run_anova and
    gage_rr.compute_variance_components (imported, not reimplemented) on the
    same simulated frame, one shape term at a time."""
    results = {}
    for term in SHAPE_TERM_NAMES:
        term_frame = frame.rename(columns={term: "measurement"})[["part", "operator", "trial", "measurement"]]
        anova_table = run_anova(term_frame)
        results[term] = compute_variance_components(anova_table, n_parts, n_operators, n_trials)
    return ShapeGageRRStudy(
        n_parts=n_parts, n_operators=n_operators, n_trials=n_trials,
        residual_df=residual_degrees_of_freedom(n_parts, n_operators, n_trials),
        results=results,
    )


@dataclass(frozen=True)
class TermVerdict:
    term: str
    status: str  # "VALIDATED", "NOT VALIDATED", "INSUFFICIENT_DATA"
    z_score: float | None
    reason: str


@dataclass(frozen=True)
class ModelValidationVerdict:
    overall_status: str  # "MODEL VALIDATED", "MODEL NOT VALIDATED", "INSUFFICIENT_DATA"
    failing_terms: tuple
    term_verdicts: dict


def standard_error_of_mean(result: GageRRResult, n_operators: int, n_trials: int) -> float:
    """The standard error of the grand mean of an n_operators x n_trials
    repeated measurement, decomposed correctly by which variance component
    averages down with which count.

    This project's first attempt at this formula used
    `sqrt(grr_var / (n_operators * n_trials))`, treating all of
    repeatability, operator and interaction variance as if they were one
    pool of i.i.d. noise that shrinks with every additional trial. That is
    wrong here: an operator's bias is drawn once and held fixed across that
    operator's n_trials repeats (see the generative model in this module's
    docstring), so running more trials on the same 3 operators narrows the
    repeatability contribution but does nothing to narrow the
    operator-to-operator contribution, which only narrows by measuring with
    more distinct operators. The correct decomposition is

        Var(grand mean) = operator_var / n_operators
                         + (interaction_var + repeatability_var) / (n_operators * n_trials)

    This was found and fixed after the naive formula produced a 12/40 false
    flag rate on clean lots at a nominally 4-standard-error threshold, which
    is wildly above the sub-1% rate a correctly-sized 4-sigma threshold
    should give; see the README Findings entry for the full account."""
    var_grand_mean = (result.operator_var / n_operators
                       + (result.interaction_var + result.repeatability_var) / (n_operators * n_trials))
    return math.sqrt(var_grand_mean) if var_grand_mean > 0 else 0.0


def validate_model_term(term: str, model_coeff: float, measured_mean_coeff: float, se: float,
                         n_operators: int, n_trials: int, n_parts: int, z_threshold: float,
                         min_residual_df: int = MIN_RESIDUAL_DF) -> TermVerdict:
    """The refusal rule for one shape term: refuse (INSUFFICIENT_DATA) if the
    gage R&R study behind `se` does not have enough residual degrees of
    freedom to trust its measurement-noise estimate; otherwise compare the
    model-vs-measured discrepancy, in standard errors of the measured mean
    (`se`, from standard_error_of_mean), against z_threshold."""
    residual_df = residual_degrees_of_freedom(n_parts, n_operators, n_trials)
    if residual_df < min_residual_df:
        return TermVerdict(
            term=term, status="INSUFFICIENT_DATA", z_score=None,
            reason=(f"only {residual_df} residual degrees of freedom from {n_parts} parts x "
                    f"{n_operators} operators x {n_trials} trials, below the minimum "
                    f"{min_residual_df} required to trust the {term} measurement-noise estimate"),
        )
    if se == 0.0:
        z = float("inf") if model_coeff != measured_mean_coeff else 0.0
    else:
        z = abs(model_coeff - measured_mean_coeff) / se
    if z > z_threshold:
        return TermVerdict(
            term=term, status="NOT VALIDATED", z_score=z,
            reason=f"{term} discrepancy {z:.2f} measurement standard errors exceeds threshold {z_threshold}",
        )
    return TermVerdict(
        term=term, status="VALIDATED", z_score=z,
        reason=f"{term} discrepancy {z:.2f} measurement standard errors within threshold {z_threshold}",
    )


def validate_model(model_coeffs: ShapeCoeffs, measured_mean_coeffs: ShapeCoeffs, se_by_term: dict,
                    n_operators: int, n_trials: int, n_parts: int, z_threshold: float,
                    min_residual_df: int = MIN_RESIDUAL_DF) -> ModelValidationVerdict:
    """Runs validate_model_term for bow, cyl and saddle and rolls the three
    up into one overall verdict. If any term is refused for insufficient
    repeats, the whole model is refused (a model cannot be honestly called
    validated on some terms and unmeasured on others); otherwise the model
    is validated only if all three terms individually validate, and the
    failing terms are named explicitly, not just reported as a single
    boolean. se_by_term maps each shape term to its pre-computed
    standard_error_of_mean from a calibration gage R&R study."""
    term_verdicts = {}
    for term in SHAPE_TERM_NAMES:
        term_verdicts[term] = validate_model_term(
            term, model_coeffs.get(term), measured_mean_coeffs.get(term), se_by_term[term],
            n_operators, n_trials, n_parts, z_threshold, min_residual_df,
        )

    if any(v.status == "INSUFFICIENT_DATA" for v in term_verdicts.values()):
        return ModelValidationVerdict(overall_status="INSUFFICIENT_DATA", failing_terms=(), term_verdicts=term_verdicts)

    failing = tuple(t for t, v in term_verdicts.items() if v.status == "NOT VALIDATED")
    overall = "MODEL NOT VALIDATED" if failing else "MODEL VALIDATED"
    return ModelValidationVerdict(overall_status=overall, failing_terms=failing, term_verdicts=term_verdicts)
