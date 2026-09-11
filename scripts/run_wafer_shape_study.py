"""Runs the full wafer-shape reconciliation extension: the decomposition's
reference-oracle and rotation-invariant checks, the shape gage R&R study
(measurement uncertainty per term), the insufficient-repeats refusal
demonstration, and the 30-seeded-error / 40-clean-lot detector sweep.

Usage: venv\\Scripts\\python scripts\\run_wafer_shape_study.py > docs\\wafer_shape_output.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dv_harness.model_form_error_detector import (
    CALIBRATION_N_OPERATORS,
    CALIBRATION_N_PARTS,
    CALIBRATION_N_TRIALS,
    INJECT_SIGMA_MULTIPLIER,
    PRODUCTION_N_OPERATORS,
    PRODUCTION_N_TRIALS,
    Z_THRESHOLD,
    run_calibration,
    run_clean_lot_sweep,
    run_seeded_error_sweep,
)
from dv_harness.wafer_shape import (
    SHAPE_TERM_NAMES,
    WAFER_RADIUS_MM,
    ShapeCoeffs,
    design_matrix_condition_number,
    fit_shape_lstsq,
    fit_shape_normal_equations,
    make_measurement_grid,
    max_abs_diff,
    predict_coeffs_after_rotate_90,
    predict_coeffs_after_rotate_180,
    rotate_90,
    rotate_180,
    synth_height_map,
)
from dv_harness.wafer_shape_gage_rr import (
    MIN_RESIDUAL_DF,
    ModelValidationVerdict,
    build_multiresponse_frame,
    make_parts,
    residual_degrees_of_freedom,
    run_shape_gage_rr_study,
    standard_error_of_mean,
    validate_model,
)


def section(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def run_recovery_and_oracle_check() -> bool:
    section("1. Noiseless ground-truth recovery + lstsq vs normal-equations reference oracle")
    x, y = make_measurement_grid(n_per_axis=31, radius=WAFER_RADIUS_MM)
    cond = design_matrix_condition_number(x, y, WAFER_RADIUS_MM)
    print(f"grid points: {len(x)}, design matrix (X^T X) condition number: {cond:.3f}")

    rng = np.random.default_rng(42)
    max_recovery_err = 0.0
    max_oracle_diff = 0.0
    n_trials = 200
    for _ in range(n_trials):
        true_coeffs = ShapeCoeffs(*rng.normal(0.0, 0.02, size=6))
        noise_sigma = rng.choice([0.0, 0.0, 0.0, 0.0005, 0.002])  # mostly noiseless, some noisy
        z = synth_height_map(true_coeffs, x, y, WAFER_RADIUS_MM, noise_sigma=noise_sigma,
                              rng=rng if noise_sigma > 0 else None)
        fitted_lstsq = fit_shape_lstsq(x, y, z, WAFER_RADIUS_MM)
        fitted_normal = fit_shape_normal_equations(x, y, z, WAFER_RADIUS_MM)
        max_oracle_diff = max(max_oracle_diff, max_abs_diff(fitted_lstsq, fitted_normal))
        if noise_sigma == 0.0:
            max_recovery_err = max(max_recovery_err, max_abs_diff(fitted_lstsq, true_coeffs))

    print(f"noiseless ground-truth recovery, max abs coefficient error over {n_trials} random trials: "
          f"{max_recovery_err:.3e} mm")
    print(f"lstsq vs normal-equations reference oracle, max abs diff over {n_trials} random trials "
          f"(noisy and noiseless): {max_oracle_diff:.3e} mm")
    recovery_ok = max_recovery_err < 1e-9
    oracle_ok = max_oracle_diff < 1e-8
    print(f"recovery meets < 1e-9 mm tolerance: {recovery_ok}")
    print(f"oracle agreement meets < 1e-8 mm tolerance: {oracle_ok}")
    return recovery_ok and oracle_ok


def run_rotation_invariants() -> bool:
    section("2. Rotation-invariant checks (90 degree and 180 degree)")
    x, y = make_measurement_grid(n_per_axis=31, radius=WAFER_RADIUS_MM)
    rng = np.random.default_rng(7)
    max_err_90 = 0.0
    max_err_180 = 0.0
    n_trials = 100
    for _ in range(n_trials):
        coeffs = ShapeCoeffs(*rng.normal(0.0, 0.02, size=6))

        x90, y90 = rotate_90(x, y)
        z90 = synth_height_map(coeffs, x90, y90, WAFER_RADIUS_MM)
        fitted_90 = fit_shape_lstsq(x, y, z90, WAFER_RADIUS_MM)
        predicted_90 = predict_coeffs_after_rotate_90(coeffs)
        max_err_90 = max(max_err_90, max_abs_diff(fitted_90, predicted_90))

        x180, y180 = rotate_180(x, y)
        z180 = synth_height_map(coeffs, x180, y180, WAFER_RADIUS_MM)
        fitted_180 = fit_shape_lstsq(x, y, z180, WAFER_RADIUS_MM)
        predicted_180 = predict_coeffs_after_rotate_180(coeffs)
        max_err_180 = max(max_err_180, max_abs_diff(fitted_180, predicted_180))

    print(f"90 degree rotation: bow invariant, cyl/saddle sign-flip -- max abs diff from analytic "
          f"prediction over {n_trials} random trials: {max_err_90:.3e} mm")
    print(f"180 degree rotation: bow/cyl/saddle invariant, tip/tilt sign-flip -- max abs diff from "
          f"analytic prediction over {n_trials} random trials: {max_err_180:.3e} mm")
    ok = max_err_90 < 1e-9 and max_err_180 < 1e-9
    print(f"both invariants meet < 1e-9 mm tolerance: {ok}")
    return ok


def run_gage_rr_report():
    section("3. Measurement uncertainty: simulated gage R&R study for the shape decomposition")
    parts = make_parts(CALIBRATION_N_PARTS, {}, seed=900)
    frame = build_multiresponse_frame(
        parts, CALIBRATION_N_OPERATORS, CALIBRATION_N_TRIALS, sigma_operator=0.004,
        sigma_repeat_point=0.0008, radius=WAFER_RADIUS_MM, n_per_axis=31, seed=901,
    )
    study = run_shape_gage_rr_study(frame, CALIBRATION_N_PARTS, CALIBRATION_N_OPERATORS, CALIBRATION_N_TRIALS)
    print(f"study design: {CALIBRATION_N_PARTS} simulated wafers x {CALIBRATION_N_OPERATORS} operators x "
          f"{CALIBRATION_N_TRIALS} trials, residual df = {study.residual_df}")
    print(f"{'term':<8} {'repeat_var':>12} {'operator_var':>13} {'interaction_var':>16} {'grr_var':>10} "
          f"{'pct_grr':>8} {'ndc':>4} {'se(3x4 mean)':>13} {'AIAG pass':>10}")
    for term in SHAPE_TERM_NAMES:
        r = study.results[term]
        se = standard_error_of_mean(r, PRODUCTION_N_OPERATORS, PRODUCTION_N_TRIALS)
        print(f"{term:<8} {r.repeatability_var:12.3e} {r.operator_var:13.3e} {r.interaction_var:16.3e} "
              f"{r.grr_var:10.3e} {r.pct_grr:8.2f} {r.ndc:4d} {se:13.3e} {str(r.passed):>10}")
    print()
    print("AIAG %GRR<=30 / NDC>=5 is reported for context only; this claim is 'measurement uncertainty")
    print("sized', not 'measurement system passes AIAG acceptance', so a term failing the AIAG rule")
    print("(cyl on NDC, saddle on both) is reported honestly rather than hidden.")
    return study


def run_refusal_demo():
    section("4. Refusal rule: insufficient repeats must not be validated")
    # A study with only 1 trial per operator gives zero residual degrees of
    # freedom (n_parts * n_operators * (1 - 1) = 0), far below MIN_RESIDUAL_DF.
    insufficient_n_parts, insufficient_n_operators, insufficient_n_trials = 2, 3, 1
    residual_df = residual_degrees_of_freedom(insufficient_n_parts, insufficient_n_operators, insufficient_n_trials)
    print(f"scenario A (insufficient): {insufficient_n_parts} parts x {insufficient_n_operators} operators x "
          f"{insufficient_n_trials} trial, residual df = {residual_df} (minimum required: {MIN_RESIDUAL_DF})")

    dummy_model = ShapeCoeffs(0, 0, 0, 0.02, 0.01, 0.01)
    dummy_measured = ShapeCoeffs(0, 0, 0, 0.02, 0.01, 0.01)  # identical: would trivially "validate" if allowed
    dummy_se = {t: 0.001 for t in SHAPE_TERM_NAMES}
    verdict_insufficient = validate_model(
        dummy_model, dummy_measured, dummy_se,
        insufficient_n_operators, insufficient_n_trials, insufficient_n_parts, Z_THRESHOLD,
    )
    print(f"verdict: {verdict_insufficient.overall_status}")
    for term, v in verdict_insufficient.term_verdicts.items():
        print(f"  {term}: {v.status} ({v.reason})")
    refused = verdict_insufficient.overall_status == "INSUFFICIENT_DATA"
    print(f"refused correctly (even though model == measured exactly): {refused}")

    print()
    print(f"scenario B (sufficient, for contrast): {CALIBRATION_N_PARTS} parts x {PRODUCTION_N_OPERATORS} "
          f"operators x {PRODUCTION_N_TRIALS} trials, residual df = "
          f"{residual_degrees_of_freedom(CALIBRATION_N_PARTS, PRODUCTION_N_OPERATORS, PRODUCTION_N_TRIALS)}")
    verdict_sufficient = validate_model(
        dummy_model, dummy_measured, dummy_se,
        PRODUCTION_N_OPERATORS, PRODUCTION_N_TRIALS, CALIBRATION_N_PARTS, Z_THRESHOLD,
    )
    print(f"verdict: {verdict_sufficient.overall_status}")
    return refused and verdict_sufficient.overall_status == "MODEL VALIDATED"


def run_detector_sweep():
    section("5. Model-form-error detector: 30 seeded errors + 40 clean lots")
    calibration = run_calibration()
    print(f"calibration study standard errors (3x4 grand mean), used as the fixed measurement-uncertainty")
    print(f"budget for every trial below: {calibration.se}")
    print(f"injected offset magnitude: {INJECT_SIGMA_MULTIPLIER} x calibration SE, detection threshold: "
          f"{Z_THRESHOLD} standard errors")

    seeded = run_seeded_error_sweep(calibration)
    caught = sum(1 for r in seeded if r.caught)
    print()
    print(f"seeded model-form-error trials: {caught}/{len(seeded)} caught (correct term named)")
    for r in seeded:
        z = {t: f"{v.z_score:.2f}" for t, v in r.verdict.term_verdicts.items()}
        print(f"  trial {r.trial:2d} injected={r.injected_term:7s} sign={r.injected_sign:+d} "
              f"verdict={r.verdict.overall_status:20s} failing={r.verdict.failing_terms} "
              f"z={z} caught={r.caught}")

    clean = run_clean_lot_sweep(calibration)
    flagged = sum(1 for r in clean if r.false_flagged)
    print()
    print(f"clean lots (no injected error): {flagged}/{len(clean)} false flags")
    for r in clean:
        if r.false_flagged:
            z = {t: f"{v.z_score:.2f}" for t, v in r.verdict.term_verdicts.items()}
            print(f"  lot {r.lot:2d} verdict={r.verdict.overall_status} failing={r.verdict.failing_terms} z={z}")

    return caught, len(seeded), flagged, len(clean)


def main() -> int:
    ok_recovery_oracle = run_recovery_and_oracle_check()
    ok_rotation = run_rotation_invariants()
    run_gage_rr_report()
    ok_refusal = run_refusal_demo()
    caught, n_seeded, flagged, n_clean = run_detector_sweep()

    section("Summary")
    print(f"decomposition recovery + reference oracle: {'PASS' if ok_recovery_oracle else 'FAIL'}")
    print(f"rotation invariants: {'PASS' if ok_rotation else 'FAIL'}")
    print(f"refusal rule fires correctly on insufficient repeats: {'PASS' if ok_refusal else 'FAIL'}")
    print(f"seeded model-form-error catch rate: {caught}/{n_seeded}")
    print(f"clean-lot false-flag rate: {flagged}/{n_clean}")

    all_ok = ok_recovery_oracle and ok_rotation and ok_refusal
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
