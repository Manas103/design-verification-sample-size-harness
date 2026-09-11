import numpy as np
import pytest

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


def test_grid_is_within_wafer_radius_and_nonempty():
    x, y = make_measurement_grid(n_per_axis=31, radius=WAFER_RADIUS_MM)
    assert len(x) == len(y)
    assert len(x) > 100
    assert np.all(x ** 2 + y ** 2 <= WAFER_RADIUS_MM ** 2 + 1e-9)


def test_design_matrix_is_well_conditioned():
    x, y = make_measurement_grid(n_per_axis=31, radius=WAFER_RADIUS_MM)
    cond = design_matrix_condition_number(x, y, WAFER_RADIUS_MM)
    assert cond < 1000  # a well-posed fit; a degenerate/collinear basis would blow this up


@pytest.mark.parametrize("seed", range(10))
def test_noiseless_fit_recovers_injected_ground_truth_exactly(seed):
    rng = np.random.default_rng(seed)
    x, y = make_measurement_grid(n_per_axis=31, radius=WAFER_RADIUS_MM)
    true_coeffs = ShapeCoeffs(*rng.normal(0.0, 0.02, size=6))
    z = synth_height_map(true_coeffs, x, y, WAFER_RADIUS_MM)
    fitted = fit_shape_lstsq(x, y, z, WAFER_RADIUS_MM)
    assert max_abs_diff(fitted, true_coeffs) < 1e-9


@pytest.mark.parametrize("noise_sigma", [0.0, 0.0005, 0.002, 0.01])
def test_lstsq_matches_normal_equations_reference_oracle(noise_sigma):
    rng = np.random.default_rng(123)
    x, y = make_measurement_grid(n_per_axis=31, radius=WAFER_RADIUS_MM)
    true_coeffs = ShapeCoeffs(*rng.normal(0.0, 0.02, size=6))
    z = synth_height_map(true_coeffs, x, y, WAFER_RADIUS_MM, noise_sigma=noise_sigma,
                          rng=rng if noise_sigma > 0 else None)
    fitted_lstsq = fit_shape_lstsq(x, y, z, WAFER_RADIUS_MM)
    fitted_normal = fit_shape_normal_equations(x, y, z, WAFER_RADIUS_MM)
    assert max_abs_diff(fitted_lstsq, fitted_normal) < 1e-8


def test_synth_height_map_requires_rng_when_noisy():
    x, y = make_measurement_grid(n_per_axis=11, radius=WAFER_RADIUS_MM)
    coeffs = ShapeCoeffs(0, 0, 0, 0.01, 0.01, 0.01)
    with pytest.raises(ValueError):
        synth_height_map(coeffs, x, y, WAFER_RADIUS_MM, noise_sigma=0.001, rng=None)


@pytest.mark.parametrize("seed", range(20))
def test_rotate_90_invariant_bow_and_sign_flip_cyl_saddle(seed):
    rng = np.random.default_rng(seed)
    x, y = make_measurement_grid(n_per_axis=25, radius=WAFER_RADIUS_MM)
    coeffs = ShapeCoeffs(*rng.normal(0.0, 0.02, size=6))
    x90, y90 = rotate_90(x, y)
    z90 = synth_height_map(coeffs, x90, y90, WAFER_RADIUS_MM)
    fitted = fit_shape_lstsq(x, y, z90, WAFER_RADIUS_MM)
    predicted = predict_coeffs_after_rotate_90(coeffs)
    assert max_abs_diff(fitted, predicted) < 1e-9
    assert fitted.bow == pytest.approx(coeffs.bow, abs=1e-9)
    assert fitted.cyl == pytest.approx(-coeffs.cyl, abs=1e-9)
    assert fitted.saddle == pytest.approx(-coeffs.saddle, abs=1e-9)


@pytest.mark.parametrize("seed", range(20))
def test_rotate_180_invariant_bow_cyl_saddle_sign_flip_tip_tilt(seed):
    rng = np.random.default_rng(seed + 1000)
    x, y = make_measurement_grid(n_per_axis=25, radius=WAFER_RADIUS_MM)
    coeffs = ShapeCoeffs(*rng.normal(0.0, 0.02, size=6))
    x180, y180 = rotate_180(x, y)
    z180 = synth_height_map(coeffs, x180, y180, WAFER_RADIUS_MM)
    fitted = fit_shape_lstsq(x, y, z180, WAFER_RADIUS_MM)
    predicted = predict_coeffs_after_rotate_180(coeffs)
    assert max_abs_diff(fitted, predicted) < 1e-9
    assert fitted.bow == pytest.approx(coeffs.bow, abs=1e-9)
    assert fitted.cyl == pytest.approx(coeffs.cyl, abs=1e-9)
    assert fitted.saddle == pytest.approx(coeffs.saddle, abs=1e-9)
    assert fitted.tip == pytest.approx(-coeffs.tip, abs=1e-9)
    assert fitted.tilt == pytest.approx(-coeffs.tilt, abs=1e-9)


def test_with_term_offset_only_changes_the_named_term():
    coeffs = ShapeCoeffs(piston=0.01, tip=0.02, tilt=0.03, bow=0.04, cyl=0.05, saddle=0.06)
    offset = coeffs.with_term_offset("cyl", 0.5)
    assert offset.cyl == pytest.approx(0.55)
    assert offset.bow == pytest.approx(coeffs.bow)
    assert offset.saddle == pytest.approx(coeffs.saddle)
    assert offset.piston == pytest.approx(coeffs.piston)


def test_shape_term_names_are_the_three_canonical_warpage_terms():
    assert set(SHAPE_TERM_NAMES) == {"bow", "cyl", "saddle"}
