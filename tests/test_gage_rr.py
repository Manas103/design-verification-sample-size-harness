import pytest

from dv_harness.gage_rr import evaluate_scenario
from dv_harness.gage_rr_scenarios import N_OPERATORS, N_PARTS, N_TRIALS, SCENARIOS


def _run(scenario):
    return evaluate_scenario(
        N_PARTS, N_OPERATORS, N_TRIALS,
        scenario.sigma_part, scenario.sigma_operator, scenario.sigma_po, scenario.sigma_repeat,
        scenario.seed,
    )


def test_scenario_set_shape():
    engineered_fail = [s for s in SCENARIOS if s.engineered_to_fail]
    engineered_pass = [s for s in SCENARIOS if not s.engineered_to_fail]
    assert len(engineered_fail) == 12
    assert len(engineered_pass) >= 1


@pytest.mark.parametrize("scenario", [s for s in SCENARIOS if s.engineered_to_fail], ids=lambda s: s.name)
def test_engineered_failure_scenarios_are_flagged_failing(scenario):
    result = _run(scenario)
    assert result.passed is False, (
        f"{scenario.name} ({scenario.failure_mode}) was engineered to fail "
        f"but the harness passed it: pct_grr={result.pct_grr:.2f}, ndc={result.ndc}"
    )
    assert len(result.reasons) >= 1


@pytest.mark.parametrize("scenario", [s for s in SCENARIOS if not s.engineered_to_fail], ids=lambda s: s.name)
def test_engineered_pass_scenarios_are_flagged_passing(scenario):
    result = _run(scenario)
    assert result.passed is True, (
        f"{scenario.name} was engineered to pass but the harness flagged it failing: "
        f"pct_grr={result.pct_grr:.2f}, ndc={result.ndc}, reasons={result.reasons}"
    )
    assert result.reasons == ()


def test_catch_rate_is_12_of_12():
    fail_scenarios = [s for s in SCENARIOS if s.engineered_to_fail]
    caught = sum(1 for s in fail_scenarios if _run(s).passed is False)
    assert caught == 12
    assert len(fail_scenarios) == 12


def test_variance_components_are_nonnegative_and_total_is_consistent():
    for s in SCENARIOS:
        result = _run(s)
        assert result.repeatability_var >= 0
        assert result.operator_var >= 0
        assert result.interaction_var >= 0
        assert result.part_var >= 0
        assert result.total_var == pytest.approx(result.grr_var + result.part_var)


def test_rule_is_not_trivially_always_fail():
    # Guard against a degenerate "always fail" acceptance rule: at least
    # one scenario (the engineered-to-pass one) must actually pass.
    assert any(_run(s).passed for s in SCENARIOS if not s.engineered_to_fail)


def test_pct_grr_of_tolerance_is_computed_and_independent_of_study_sample():
    from dv_harness.gage_rr import pct_grr_of_tolerance
    result = _run(SCENARIOS[0])
    assert result.pct_grr_tolerance == pytest.approx(
        pct_grr_of_tolerance(result.grr_var, tolerance=6.0)
    )
    # Same grr_var, different tolerance widths must give different
    # percentages (this is the entire point of the metric: it does not
    # depend on how much part-to-part variation this study happened to
    # sample, only on the stated engineering tolerance).
    assert pct_grr_of_tolerance(result.grr_var, tolerance=3.0) == pytest.approx(
        2 * pct_grr_of_tolerance(result.grr_var, tolerance=6.0)
    )
