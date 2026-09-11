import numpy as np
import pytest

from dv_harness.attribute_agreement import cohens_kappa, evaluate_attribute_study
from dv_harness.attribute_agreement_scenarios import ATTRIBUTE_SCENARIOS


def _run(scenario):
    return evaluate_attribute_study(scenario.n_parts, scenario.true_defect_rate,
                                     scenario.sensitivity, scenario.specificity, scenario.seed)


def test_scenario_set_shape():
    engineered_fail = [s for s in ATTRIBUTE_SCENARIOS if s.engineered_to_fail]
    engineered_pass = [s for s in ATTRIBUTE_SCENARIOS if not s.engineered_to_fail]
    assert len(engineered_fail) == 6
    assert len(engineered_pass) >= 1


@pytest.mark.parametrize("scenario", [s for s in ATTRIBUTE_SCENARIOS if s.engineered_to_fail], ids=lambda s: s.name)
def test_engineered_failure_scenarios_are_flagged_failing(scenario):
    result = _run(scenario)
    assert result.passed is False, (
        f"{scenario.name} ({scenario.failure_mode}) was engineered to fail "
        f"but the harness passed it: kappa={result.kappa:.4f}"
    )


@pytest.mark.parametrize("scenario", [s for s in ATTRIBUTE_SCENARIOS if not s.engineered_to_fail], ids=lambda s: s.name)
def test_engineered_pass_scenarios_are_flagged_passing(scenario):
    result = _run(scenario)
    assert result.passed is True, (
        f"{scenario.name} was engineered to pass but the harness flagged it failing: kappa={result.kappa:.4f}"
    )


def test_catch_rate_is_6_of_6():
    fail_scenarios = [s for s in ATTRIBUTE_SCENARIOS if s.engineered_to_fail]
    caught = sum(1 for s in fail_scenarios if _run(s).passed is False)
    assert caught == 6
    assert len(fail_scenarios) == 6


def test_kappa_perfect_agreement_is_one():
    truth = np.array([True, False, True, True, False, False, True, False])
    assert cohens_kappa(truth, truth) == pytest.approx(1.0)


def test_kappa_of_a_deterministic_opposite_rater_is_negative():
    truth = np.array([True, False] * 20)
    opposite = ~truth
    assert cohens_kappa(truth, opposite) < 0


def test_kappa_matches_hand_computed_confusion_matrix():
    # 10 parts: truth has 4 defects, appraiser calls 3 of them plus one
    # false positive on a good part. Computed by hand from the 2x2 table:
    # a=3 (true positive), b=1 (false positive), c=1 (false negative), d=5 (true negative)
    truth =     np.array([True, True, True, True, False, False, False, False, False, False])
    appraiser = np.array([True, True, True, False, True, False, False, False, False, False])
    po = (3 + 5) / 10  # 0.8
    p_truth = 4 / 10
    p_appr = 4 / 10
    pe = p_truth * p_appr + (1 - p_truth) * (1 - p_appr)
    expected_kappa = (po - pe) / (1 - pe)
    assert cohens_kappa(truth, appraiser) == pytest.approx(expected_kappa)


def test_percent_agreement_and_kappa_diverge_for_a_random_guesser():
    # At a low true defect rate, a rater that ignores the part entirely
    # and always calls "no defect" has high raw percent agreement but
    # kappa should be 0 (no better than chance, by construction: constant
    # calls carry no information at all).
    result = evaluate_attribute_study(n_parts=200, true_defect_rate=0.05,
                                       sensitivity=0.0, specificity=1.0, seed=7)
    assert result.percent_agreement > 0.9
    assert result.kappa == pytest.approx(0.0, abs=1e-9)
