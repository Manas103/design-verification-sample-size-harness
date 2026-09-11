import pytest

from dv_harness.nested_gage_rr import evaluate_nested_scenario
from dv_harness.nested_gage_rr_scenarios import N_OPERATORS, N_PARTS_PER_OPERATOR, N_TRIALS, NESTED_SCENARIOS


def _run(scenario):
    return evaluate_nested_scenario(
        N_OPERATORS, N_PARTS_PER_OPERATOR, N_TRIALS,
        scenario.sigma_part, scenario.sigma_operator, scenario.sigma_repeat,
        scenario.seed,
    )


def test_scenario_set_shape():
    engineered_fail = [s for s in NESTED_SCENARIOS if s.engineered_to_fail]
    engineered_pass = [s for s in NESTED_SCENARIOS if not s.engineered_to_fail]
    assert len(engineered_fail) == 6
    assert len(engineered_pass) >= 1


@pytest.mark.parametrize("scenario", [s for s in NESTED_SCENARIOS if s.engineered_to_fail], ids=lambda s: s.name)
def test_engineered_failure_scenarios_are_flagged_failing(scenario):
    result = _run(scenario)
    assert result.passed is False, (
        f"{scenario.name} ({scenario.failure_mode}) was engineered to fail "
        f"but the harness passed it: pct_grr={result.pct_grr:.2f}, ndc={result.ndc}"
    )
    assert len(result.reasons) >= 1


@pytest.mark.parametrize("scenario", [s for s in NESTED_SCENARIOS if not s.engineered_to_fail], ids=lambda s: s.name)
def test_engineered_pass_scenarios_are_flagged_passing(scenario):
    result = _run(scenario)
    assert result.passed is True, (
        f"{scenario.name} was engineered to pass but the harness flagged it failing: "
        f"pct_grr={result.pct_grr:.2f}, ndc={result.ndc}, reasons={result.reasons}"
    )
    assert result.reasons == ()


def test_catch_rate_is_6_of_6():
    fail_scenarios = [s for s in NESTED_SCENARIOS if s.engineered_to_fail]
    caught = sum(1 for s in fail_scenarios if _run(s).passed is False)
    assert caught == 6
    assert len(fail_scenarios) == 6


def test_variance_components_are_nonnegative_and_total_is_consistent():
    for s in NESTED_SCENARIOS:
        result = _run(s)
        assert result.repeatability_var >= 0
        assert result.operator_var >= 0
        assert result.part_var >= 0
        assert result.total_var == pytest.approx(result.grr_var + result.part_var)


def test_sum_of_squares_partition_is_exact():
    # The hand-computed nested ANOVA's SS_operator + SS_part_in_op +
    # SS_error must equal the total corrected sum of squares exactly,
    # since every measurement's deviation from the grand mean is
    # partitioned into exactly those three pieces.
    from dv_harness.nested_gage_rr import run_nested_anova, simulate_nested_study
    df = simulate_nested_study(N_OPERATORS, N_PARTS_PER_OPERATOR, N_TRIALS, 1.0, 0.4, 0.3, seed=42)
    table = run_nested_anova(df)
    ss_total = float(((df["measurement"] - df["measurement"].mean()) ** 2).sum())
    ss_sum = table["ss_operator"] + table["ss_part_in_op"] + table["ss_error"]
    assert ss_sum == pytest.approx(ss_total, rel=1e-9)
    assert table["df_operator"] + table["df_part_in_op"] + table["df_error"] == len(df) - 1


def test_no_interaction_term_is_representable():
    # A nested design has no part*operator cell that repeats across
    # operators, so there is no way to name an "interaction" variance
    # component; NestedGageRRResult has no such field (unlike the
    # crossed GageRRResult), which this test pins down structurally.
    result = _run(NESTED_SCENARIOS[0])
    assert not hasattr(result, "interaction_var")
