"""The demo requirement set used by scripts/run_requirement_report.py.

Builds real evidence from the other three modules (no mocked pass/fail
booleans): the attribute and variables sample-size reference checks, and
the actual measured gage R&R catch rate over the 13 seeded scenarios.
REQ-005 is deliberately left without any attached evidence, to prove the
report generator's refusal behavior fires for real, not just in a unit
test.
"""
from __future__ import annotations

from dv_harness.attribute_sampling import zero_failure_sample_size
from dv_harness.gage_rr import evaluate_scenario
from dv_harness.gage_rr_scenarios import N_OPERATORS, N_PARTS, N_TRIALS, SCENARIOS
from dv_harness.requirements_report import Evidence, Requirement
from dv_harness.variables_sampling import one_sided_k_factor

REQUIREMENTS = [
    Requirement(
        id="REQ-001",
        description="Attribute (pass/fail) sample size shall be computable at 95% confidence and 95% reliability.",
        acceptance_criterion="zero-failure closed-form sample size at C=0.95, R=0.95 equals the published textbook value n=59",
    ),
    Requirement(
        id="REQ-002",
        description="Variables sample-size k-factor shall match published one-sided tolerance-interval table values.",
        acceptance_criterion="k(n=10, C=0.95, R=0.90) matches Natrella NBS Handbook 91 Table A-6 value 2.355 within 0.005",
    ),
    Requirement(
        id="REQ-003",
        description="The gage R&R harness shall flag every seeded measurement-system failure scenario as failing.",
        acceptance_criterion="all 12 engineered-to-fail scenarios are classified failed by the %GRR/NDC rule",
    ),
    Requirement(
        id="REQ-004",
        description="The gage R&R harness shall not flag a healthy measurement system as failing.",
        acceptance_criterion="the engineered-to-pass scenario is classified passed by the %GRR/NDC rule",
    ),
    Requirement(
        id="REQ-005",
        description="The verification report generator shall refuse to mark a requirement VERIFIED when no evidence has been attached.",
        acceptance_criterion="a requirement with zero attached evidence is reported NOT VERIFIED, reason 'no evidence attached'",
    ),
]


def build_evidence():
    """Runs the actual modules and returns dict[str, list[Evidence]].
    REQ-005 is intentionally omitted so its evidence list is empty."""
    evidence = {}

    n59 = zero_failure_sample_size(0.95, 0.95)
    evidence["REQ-001"] = [
        Evidence("REQ-001", passed=(n59 == 59),
                 detail=f"zero_failure_sample_size(0.95, 0.95) = {n59} (expected 59)",
                 source="dv_harness.attribute_sampling")
    ]

    k10 = one_sided_k_factor(10, 0.95, 0.90)
    evidence["REQ-002"] = [
        Evidence("REQ-002", passed=(abs(k10 - 2.355) <= 0.005),
                 detail=f"one_sided_k_factor(10, 0.95, 0.90) = {k10:.4f} (published 2.355, |diff|={abs(k10 - 2.355):.4f})",
                 source="dv_harness.variables_sampling")
    ]

    fail_scenarios = [s for s in SCENARIOS if s.engineered_to_fail]
    caught = 0
    for s in fail_scenarios:
        r = evaluate_scenario(N_PARTS, N_OPERATORS, N_TRIALS, s.sigma_part, s.sigma_operator,
                               s.sigma_po, s.sigma_repeat, s.seed)
        caught += int(not r.passed)
    evidence["REQ-003"] = [
        Evidence("REQ-003", passed=(caught == len(fail_scenarios)),
                 detail=f"caught {caught}/{len(fail_scenarios)} seeded failure scenarios",
                 source="dv_harness.gage_rr")
    ]

    pass_scenarios = [s for s in SCENARIOS if not s.engineered_to_fail]
    all_passed = True
    detail_parts = []
    for s in pass_scenarios:
        r = evaluate_scenario(N_PARTS, N_OPERATORS, N_TRIALS, s.sigma_part, s.sigma_operator,
                               s.sigma_po, s.sigma_repeat, s.seed)
        all_passed = all_passed and r.passed
        detail_parts.append(f"{s.name}: passed={r.passed}")
    evidence["REQ-004"] = [
        Evidence("REQ-004", passed=all_passed, detail="; ".join(detail_parts), source="dv_harness.gage_rr")
    ]

    # REQ-005: deliberately no evidence attached.

    return evidence
