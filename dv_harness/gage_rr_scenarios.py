"""The 13 seeded gage R&R scenarios: 12 engineered to genuinely fail the
%GRR / NDC acceptance rule (each via a distinct, physically-motivated
measurement-system failure mode), plus 1 engineered to genuinely pass, so
the pass/fail rule is proven non-trivial in both directions.

All scenarios use the same study shape (10 parts x 3 operators x 3
trials = 90 measurements) and the same nominal part-to-part standard
deviation (sigma_part = 1.0, except scenario 10 which deliberately
shrinks it to model a poor-discrimination gage). Only the measurement
error components are varied per scenario. Each scenario has its own RNG
seed so results are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

N_PARTS = 10
N_OPERATORS = 3
N_TRIALS = 3


@dataclass(frozen=True)
class Scenario:
    name: str
    failure_mode: str
    engineered_to_fail: bool
    sigma_part: float
    sigma_operator: float
    sigma_po: float
    sigma_repeat: float
    seed: int


SCENARIOS = [
    Scenario("repeatability_fail_moderate", "inflated repeatability (poor equipment precision)",
             True, 1.0, 0.05, 0.05, 0.50, seed=1),
    Scenario("repeatability_fail_severe", "inflated repeatability, severe",
             True, 1.0, 0.05, 0.05, 0.90, seed=2),
    Scenario("operator_bias_fail_moderate", "inflated operator-to-operator bias",
             True, 1.0, 0.50, 0.05, 0.05, seed=4),
    Scenario("operator_bias_fail_severe", "inflated operator-to-operator bias, severe",
             True, 1.0, 0.90, 0.05, 0.05, seed=4),
    Scenario("interaction_fail_moderate", "inflated part x operator interaction (inconsistent operators on specific parts)",
             True, 1.0, 0.05, 0.50, 0.05, seed=5),
    Scenario("interaction_fail_severe", "inflated part x operator interaction, severe",
             True, 1.0, 0.05, 0.90, 0.05, seed=6),
    Scenario("combined_repeatability_operator", "inflated repeatability and operator bias together",
             True, 1.0, 0.35, 0.05, 0.35, seed=7),
    Scenario("combined_repeatability_interaction", "inflated repeatability and interaction together",
             True, 1.0, 0.05, 0.35, 0.35, seed=8),
    Scenario("combined_operator_interaction", "inflated operator bias and interaction together",
             True, 1.0, 0.35, 0.35, 0.05, seed=9),
    Scenario("poor_discrimination_low_part_variation", "poor discrimination/resolution: part-to-part variation barely exceeds measurement noise",
             True, 0.10, 0.05, 0.05, 0.15, seed=10),
    Scenario("combined_all_moderate", "repeatability, operator bias, and interaction all moderately inflated",
             True, 1.0, 0.30, 0.30, 0.30, seed=11),
    Scenario("combined_all_severe", "repeatability, operator bias, and interaction all severely inflated",
             True, 1.0, 0.60, 0.60, 0.60, seed=12),
    Scenario("well_designed_gage_pass", "healthy measurement system: low repeatability, operator, and interaction variance",
             False, 1.0, 0.03, 0.02, 0.05, seed=100),
]
