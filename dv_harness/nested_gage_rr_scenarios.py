"""The 7 seeded nested (destructive-test) gage R&R scenarios: 6 engineered
to genuinely fail the %GRR / NDC acceptance rule, plus 1 engineered to
genuinely pass.

All scenarios use the same study shape (3 operators x 10 destructive
parts per operator x 3 replicate specimens per part = 90 measurements),
matching the crossed study's 90-measurement scale, and the same nominal
part-to-part standard deviation (sigma_part = 1.0, except the
poor-discrimination scenario, which deliberately shrinks it). Unlike the
crossed scenario set, there is no interaction-failure scenario here: a
nested design structurally cannot estimate a part*operator interaction
term (see nested_gage_rr.py), so an "inconsistent operators on specific
parts" failure mode is not representable and is not claimed to be
detectable by this design.
"""
from __future__ import annotations

from dataclasses import dataclass

N_OPERATORS = 3
N_PARTS_PER_OPERATOR = 10
N_TRIALS = 3


@dataclass(frozen=True)
class NestedScenario:
    name: str
    failure_mode: str
    engineered_to_fail: bool
    sigma_part: float
    sigma_operator: float
    sigma_repeat: float
    seed: int


NESTED_SCENARIOS = [
    NestedScenario("nested_repeatability_fail_moderate", "inflated repeatability across replicate specimens",
                   True, 1.0, 0.05, 0.60, seed=201),
    NestedScenario("nested_repeatability_fail_severe", "inflated repeatability across replicate specimens, severe",
                   True, 1.0, 0.05, 1.00, seed=202),
    NestedScenario("nested_operator_fail_moderate", "inflated operator-to-operator bias, confounded with any real interaction",
                   True, 1.0, 0.55, 0.05, seed=202),
    NestedScenario("nested_operator_fail_severe", "inflated operator-to-operator bias, severe",
                   True, 1.0, 0.95, 0.05, seed=204),
    NestedScenario("nested_combined_fail", "repeatability and operator bias both moderately inflated",
                   True, 1.0, 0.40, 0.40, seed=205),
    NestedScenario("nested_poor_discrimination_fail", "poor discrimination: part-to-part variation barely exceeds measurement noise",
                   True, 0.10, 0.05, 0.18, seed=206),
    NestedScenario("nested_well_designed_gage_pass", "healthy destructive-test measurement system: low repeatability and operator variance",
                   False, 1.0, 0.03, 0.06, seed=300),
]
