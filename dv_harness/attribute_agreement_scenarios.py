"""The 7 seeded attribute agreement scenarios: 6 engineered to genuinely
fail the Cohen's kappa acceptance rule, plus 1 engineered to genuinely
pass.

All scenarios use n_parts = 150 simulated visual pass/fail inspections
except `rare_defect_low_power_fail`, which deliberately uses the same
150 but a much lower true defect rate (3% instead of 20%) to demonstrate
a distinct, realistic failure mode: even a moderately competent appraiser
(85% sensitivity, 90% specificity) can produce an unacceptable kappa
purely because so few of the 150 parts are actually defective that
chance-corrected agreement has very little signal to work with (the
"kappa paradox" of rare-event attribute agreement).
"""
from __future__ import annotations

from dataclasses import dataclass

N_PARTS = 150
TRUE_DEFECT_RATE = 0.20


@dataclass(frozen=True)
class AttributeScenario:
    name: str
    failure_mode: str
    engineered_to_fail: bool
    n_parts: int
    true_defect_rate: float
    sensitivity: float
    specificity: float
    seed: int


ATTRIBUTE_SCENARIOS = [
    AttributeScenario("insensitive_rater_fail", "low sensitivity: appraiser misses real defects",
                       True, N_PARTS, TRUE_DEFECT_RATE, 0.55, 0.98, seed=401),
    AttributeScenario("overcalling_rater_fail", "low specificity: appraiser over-rejects good parts",
                       True, N_PARTS, TRUE_DEFECT_RATE, 0.97, 0.55, seed=401),
    AttributeScenario("random_guesser_fail", "appraiser calls are uncorrelated with the true condition",
                       True, N_PARTS, TRUE_DEFECT_RATE, 0.50, 0.50, seed=401),
    AttributeScenario("combined_poor_rater_fail", "sensitivity and specificity both moderately degraded",
                       True, N_PARTS, TRUE_DEFECT_RATE, 0.75, 0.75, seed=401),
    AttributeScenario("rare_defect_low_power_fail", "moderately competent appraiser, but true defect rate too low for kappa to have power",
                       True, N_PARTS, 0.03, 0.85, 0.90, seed=401),
    AttributeScenario("severe_bias_shift_fail", "severe specificity collapse: appraiser rejects nearly everything",
                       True, N_PARTS, TRUE_DEFECT_RATE, 0.97, 0.35, seed=401),
    AttributeScenario("well_calibrated_rater_pass", "healthy appraiser: high sensitivity and specificity",
                       False, N_PARTS, TRUE_DEFECT_RATE, 0.98, 0.98, seed=500),
]
