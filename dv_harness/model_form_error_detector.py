"""Model-form-error detector: given a "modeled" wafer shape and a
simulated, repeatedly-measured "true" wafer, detect whether the model
disagrees with measurement by more than the sized measurement uncertainty
can explain, and name which shape term (bow, cylindrical, saddle) is
responsible.

This is the sweep that produces the two matched numbers this project's
resume claim depends on: the catch rate on 30 seeded model-form errors
(one shape term deliberately perturbed per trial, beyond what the gage R&R
study's measurement noise can explain) and the false-flag rate on 40 clean
lots (no seeded error, only realistic simulated measurement noise). Both
numbers are measured by actually running this sweep, not assumed.

Calibration
-----------
Measurement uncertainty (grr_var per shape term) is sized ONCE, from a
CALIBRATION_STUDY (10 simulated wafers x 3 operators x 4 trials, residual
degrees of freedom 90, comfortably above MIN_RESIDUAL_DF), exactly as a
real DV program sizes its measurement system once and then reuses that
budget to judge many subsequent models, rather than re-deriving
measurement uncertainty from scratch for every model comparison. Every
one of the 30 seeded-error trials and 40 clean-lot trials below re-uses
this same calibration grr_var; only the model-vs-measurement comparison
changes per trial.

Injected model-form error magnitude
------------------------------------
Each seeded trial injects an offset of INJECT_SIGMA_MULTIPLIER (8) times
the calibration standard error into exactly one randomly chosen shape
term, with a random sign. 8 measurement standard errors is a magnitude
designed to be unambiguously "a model-form error beyond what measurement
noise explains" (a genuine measurement discrepancy of 8 SEs under a
correctly-sized noise budget has a two-sided normal-tail probability of
about 1.2e-15, i.e. this is not something the gage R&R noise budget could
plausibly produce by chance); it is not tuned against the detector's
threshold; see the detection threshold discussion in
dv_harness/wafer_shape_gage_rr.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dv_harness.wafer_shape import SHAPE_TERM_NAMES, WAFER_RADIUS_MM
from dv_harness.wafer_shape_gage_rr import (
    ModelValidationVerdict,
    build_multiresponse_frame,
    make_parts,
    run_shape_gage_rr_study,
    standard_error_of_mean,
    validate_model,
)

N_PER_AXIS = 31
SIGMA_OPERATOR = 0.004  # mm, operator/instrument bow-cyl-saddle calibration bias, stdev
SIGMA_REPEAT_POINT = 0.0008  # mm, per-point repeatability noise, stdev

CALIBRATION_N_PARTS = 10
CALIBRATION_N_OPERATORS = 3
CALIBRATION_N_TRIALS = 4
CALIBRATION_SEED = 900

PRODUCTION_N_OPERATORS = 3
PRODUCTION_N_TRIALS = 4

Z_THRESHOLD = 4.0
INJECT_SIGMA_MULTIPLIER = 8.0

N_SEEDED_TRIALS = 30
N_CLEAN_LOTS = 40
SEEDED_BASE_SEED = 5000
CLEAN_BASE_SEED = 6000


@dataclass(frozen=True)
class CalibrationResult:
    study: object  # ShapeGageRRStudy, kept for reporting grr_var/operator_var/etc per term
    se: dict  # term -> standard_error_of_mean at PRODUCTION_N_OPERATORS x PRODUCTION_N_TRIALS


def run_calibration() -> CalibrationResult:
    parts = make_parts(CALIBRATION_N_PARTS, {}, seed=CALIBRATION_SEED)
    frame = build_multiresponse_frame(
        parts, CALIBRATION_N_OPERATORS, CALIBRATION_N_TRIALS, SIGMA_OPERATOR, SIGMA_REPEAT_POINT,
        WAFER_RADIUS_MM, N_PER_AXIS, seed=CALIBRATION_SEED + 1,
    )
    study = run_shape_gage_rr_study(frame, CALIBRATION_N_PARTS, CALIBRATION_N_OPERATORS, CALIBRATION_N_TRIALS)
    se = {
        term: standard_error_of_mean(study.results[term], PRODUCTION_N_OPERATORS, PRODUCTION_N_TRIALS)
        for term in SHAPE_TERM_NAMES
    }
    return CalibrationResult(study=study, se=se)


def _measure_one_wafer(true_coeffs, seed: int):
    """Simulate a PRODUCTION_N_OPERATORS x PRODUCTION_N_TRIALS repeated gage
    measurement of a single wafer with the given true coeffs, and return the
    mean fitted ShapeCoeffs across all scans."""
    from dv_harness.wafer_shape_gage_rr import WaferPart

    part = WaferPart(name="W_trial", coeffs=true_coeffs)
    frame = build_multiresponse_frame(
        [part], PRODUCTION_N_OPERATORS, PRODUCTION_N_TRIALS, SIGMA_OPERATOR, SIGMA_REPEAT_POINT,
        WAFER_RADIUS_MM, N_PER_AXIS, seed=seed,
    )
    from dv_harness.wafer_shape import ShapeCoeffs
    return ShapeCoeffs(
        piston=0.0, tip=0.0, tilt=0.0,
        bow=float(frame["bow"].mean()), cyl=float(frame["cyl"].mean()), saddle=float(frame["saddle"].mean()),
    )


@dataclass(frozen=True)
class SeededTrialResult:
    trial: int
    injected_term: str
    injected_sign: int
    verdict: ModelValidationVerdict
    caught: bool


@dataclass(frozen=True)
class CleanLotResult:
    lot: int
    verdict: ModelValidationVerdict
    false_flagged: bool


def run_seeded_error_sweep(calibration: CalibrationResult) -> list:
    results = []
    for i in range(N_SEEDED_TRIALS):
        seed = SEEDED_BASE_SEED + i
        rng = np.random.default_rng(seed)
        true_part = make_parts(1, {}, seed=seed)[0]
        injected_term = SHAPE_TERM_NAMES[rng.integers(0, len(SHAPE_TERM_NAMES))]
        sign = 1 if rng.random() < 0.5 else -1
        offset = sign * INJECT_SIGMA_MULTIPLIER * calibration.se[injected_term]
        model_coeffs = true_part.coeffs.with_term_offset(injected_term, offset)

        measured_mean = _measure_one_wafer(true_part.coeffs, seed=seed + 200000)

        verdict = validate_model(
            model_coeffs, measured_mean, calibration.se,
            PRODUCTION_N_OPERATORS, PRODUCTION_N_TRIALS, CALIBRATION_N_PARTS, Z_THRESHOLD,
        )
        caught = verdict.overall_status == "MODEL NOT VALIDATED" and verdict.failing_terms == (injected_term,)
        results.append(SeededTrialResult(trial=i, injected_term=injected_term, injected_sign=sign,
                                          verdict=verdict, caught=caught))
    return results


def run_clean_lot_sweep(calibration: CalibrationResult) -> list:
    results = []
    for j in range(N_CLEAN_LOTS):
        seed = CLEAN_BASE_SEED + j
        true_part = make_parts(1, {}, seed=seed)[0]
        model_coeffs = true_part.coeffs  # no injected error: model matches truth exactly

        measured_mean = _measure_one_wafer(true_part.coeffs, seed=seed + 200000)

        verdict = validate_model(
            model_coeffs, measured_mean, calibration.se,
            PRODUCTION_N_OPERATORS, PRODUCTION_N_TRIALS, CALIBRATION_N_PARTS, Z_THRESHOLD,
        )
        false_flagged = verdict.overall_status != "MODEL VALIDATED"
        results.append(CleanLotResult(lot=j, verdict=verdict, false_flagged=false_flagged))
    return results
