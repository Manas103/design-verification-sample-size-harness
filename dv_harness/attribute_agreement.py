"""Attribute agreement analysis (visual pass/fail) by Cohen's kappa.

Where the gage R&R modules validate a measurement system that produces a
continuous number, an attribute inspection (a visual cosmetic check, a
go/no-go fixture read) produces a binary call: pass or fail. Cohen's
kappa is the standard way to score an appraiser's calls against a known
reference standard while correcting for the agreement expected by chance
alone (a coin-flipping appraiser on a population that is 95% good parts
already agrees with the standard 90%+ of the time on raw percent
agreement; kappa is near zero for that appraiser, which is the entire
reason kappa rather than raw agreement is used here).

Generative model (what `simulate_attribute_study` actually draws): each
of `n_parts` simulated parts has a true condition (defect or not, drawn
Bernoulli at `true_defect_rate`); the appraiser's call is correct with
probability `sensitivity` when the truth is defect, and with probability
`specificity` when the truth is not-defect, independently per part.

    kappa = (p_observed - p_expected) / (1 - p_expected)

    p_observed = fraction of parts where appraiser call == truth
    p_expected = P(both call defect by chance) + P(both call not-defect by chance)
               = p_truth_defect * p_appraiser_defect
               + (1 - p_truth_defect) * (1 - p_appraiser_defect)

Acceptance rule: kappa >= KAPPA_THRESHOLD (0.75) is PASS. This project
uses 0.75 as a single bright-line threshold, the same "stated before any
scenario was run, not discovered afterward" discipline as the gage R&R
modules' %GRR/NDC thresholds. 0.75 falls inside the "substantial to
almost perfect" band of the commonly cited Landis and Koch (1977) kappa
scale (0.61-0.80 substantial, 0.81-1.00 almost perfect) and is a
conventional bright line for appraiser-vs-standard attribute agreement in
industrial practice; it is not derived from this project's own data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

KAPPA_THRESHOLD = 0.75


def simulate_attribute_study(n_parts: int, true_defect_rate: float,
                              sensitivity: float, specificity: float, seed: int) -> pd.DataFrame:
    """Simulate one attribute agreement study and return a DataFrame with
    columns part, truth_defect (bool), appraiser_defect (bool)."""
    rng = np.random.default_rng(seed)
    truth_defect = rng.random(n_parts) < true_defect_rate

    appraiser_defect = np.empty(n_parts, dtype=bool)
    for i in range(n_parts):
        if truth_defect[i]:
            appraiser_defect[i] = rng.random() < sensitivity
        else:
            appraiser_defect[i] = rng.random() < (1.0 - specificity)

    return pd.DataFrame({
        "part": [f"P{i:04d}" for i in range(n_parts)],
        "truth_defect": truth_defect,
        "appraiser_defect": appraiser_defect,
    })


def cohens_kappa(truth: np.ndarray, appraiser: np.ndarray) -> float:
    """Cohen's kappa between two boolean arrays of the same length,
    computed directly from the 2x2 marginal proportions (not from a
    library), which is the same closed-form identity a manual by-hand
    calculation from a confusion matrix would use."""
    truth = np.asarray(truth, dtype=bool)
    appraiser = np.asarray(appraiser, dtype=bool)
    n = len(truth)
    p_observed = np.mean(truth == appraiser)
    p_truth_defect = np.mean(truth)
    p_appraiser_defect = np.mean(appraiser)
    p_expected = (p_truth_defect * p_appraiser_defect
                  + (1.0 - p_truth_defect) * (1.0 - p_appraiser_defect))
    if p_expected >= 1.0:
        # Degenerate case: every call is identical to every truth value
        # (e.g. a population with 0% and 100% defect rate simultaneously,
        # which cannot happen with n>0, but guarded for completeness).
        return 1.0 if p_observed >= 1.0 else 0.0
    return (p_observed - p_expected) / (1.0 - p_expected)


@dataclass(frozen=True)
class AttributeAgreementResult:
    n_parts: int
    n_defect_truth: int
    n_defect_calls: int
    percent_agreement: float
    kappa: float
    passed: bool
    reasons: tuple


def evaluate_attribute_study(n_parts: int, true_defect_rate: float,
                              sensitivity: float, specificity: float, seed: int) -> AttributeAgreementResult:
    df = simulate_attribute_study(n_parts, true_defect_rate, sensitivity, specificity, seed)
    kappa = cohens_kappa(df["truth_defect"].to_numpy(), df["appraiser_defect"].to_numpy())
    percent_agreement = float(np.mean(df["truth_defect"] == df["appraiser_defect"]))

    reasons = []
    if not (kappa >= KAPPA_THRESHOLD):  # also catches NaN
        reasons.append(f"kappa={kappa:.4f} below threshold {KAPPA_THRESHOLD}")
    passed = len(reasons) == 0

    return AttributeAgreementResult(
        n_parts=n_parts,
        n_defect_truth=int(df["truth_defect"].sum()),
        n_defect_calls=int(df["appraiser_defect"].sum()),
        percent_agreement=percent_agreement,
        kappa=kappa,
        passed=passed,
        reasons=tuple(reasons),
    )
