"""Variables (normal-distribution) sample size via one-sided tolerance
intervals.

A variables (measured, continuous) sampling plan can demonstrate the same
reliability/confidence claim as an attribute (pass/fail) plan with far
fewer units, by using the actual measured values instead of collapsing
them to pass/fail, provided the underlying distribution is (approximately)
normal. The tool for this is the one-sided normal tolerance interval:
given n measurements with sample mean x_bar and sample standard deviation
s, the interval [x_bar - k*s, infinity) contains at least a proportion R
of the population with confidence C, where k = k(n, C, R) is the one-sided
tolerance factor computed here.

k-factor, exact (noncentral t) method
--------------------------------------
    k(n, C, R) = t'_{C}(n - 1, delta) / sqrt(n)
    delta = z_R * sqrt(n)

where z_R = Phi^-1(R) is the standard normal quantile for proportion R,
and t'_{C}(df, delta) is the C-quantile of the noncentral t distribution
with df = n-1 degrees of freedom and noncentrality delta. This is the
textbook exact construction (see e.g. Hahn & Meeker, "Statistical
Intervals: A Guide for Practitioners", section 4.2, or the NIST/SEMATECH
e-Handbook of Statistical Methods, section 7.2.6.3) implemented directly
with `scipy.stats.nct.ppf`, not approximated.

Why noncentral-t rather than a large-sample normal approximation: the
whole point of small design-verification sample sizes (n well under 30)
is that the sampling distribution of s is not tight enough to ignore, and
a naive k = z_R + z_C/sqrt(n) style formula understates k (and therefore
overstates the demonstrated reliability) exactly in the small-n regime
this project cares about. The noncentral-t construction accounts for that
extra uncertainty exactly.

Reference values validated against (see README and
docs/sample_size_reference_check.txt for the actual measured comparison):
one-sided tolerance factors k for confidence gamma = 0.95, tabulated in
Natrella, "Experimental Statistics" (NBS Handbook 91, 1963), Table A-6,
and reproduced in the NIST/SEMATECH e-Handbook of Statistical Methods,
section 7.2.6.3:
    n=2,  P=0.90 -> k=20.581
    n=5,  P=0.90 -> k=3.407
    n=10, P=0.90 -> k=2.355
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import nct, norm


def one_sided_k_factor(n: int, confidence: float, reliability: float) -> float:
    """Exact one-sided normal tolerance factor k(n, C, R) via the
    noncentral t distribution."""
    if n < 2:
        raise ValueError("n must be >= 2 (need at least 1 degree of freedom)")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    if not (0.0 < reliability < 1.0):
        raise ValueError("reliability must be in (0, 1)")

    z_r = norm.ppf(reliability)
    delta = z_r * np.sqrt(n)
    t_quantile = nct.ppf(confidence, df=n - 1, nc=delta)
    return float(t_quantile / np.sqrt(n))


def howe_approximation_k_factor(n: int, confidence: float, reliability: float) -> float:
    """Howe (1969) closed-form approximation to the one-sided tolerance
    factor, used here purely as an independent second formula (not
    derived from the noncentral-t machinery above) to cross-check
    `one_sided_k_factor` on a value that is not one of the cited table
    entries. See NIST/SEMATECH e-Handbook of Statistical Methods, section
    7.2.6.3, "Approximate method", for the same formula.

        a = 1 - z_C^2 / (2*(n-1))
        b = z_R^2 - z_C^2 / n
        k = (z_R + sqrt(z_R^2 - a*b)) / a
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    z_c = norm.ppf(confidence)
    z_r = norm.ppf(reliability)
    a = 1.0 - (z_c ** 2) / (2.0 * (n - 1))
    b = (z_r ** 2) - (z_c ** 2) / n
    radicand = z_r ** 2 - a * b
    if radicand < 0 or a <= 0:
        raise ValueError(f"Howe approximation not valid at n={n}, C={confidence}, R={reliability}")
    return float((z_r + np.sqrt(radicand)) / a)


@dataclass(frozen=True)
class VariablesSampleSizeResult:
    confidence: float
    reliability: float
    capability_margin_sigma: float
    n: int
    k_at_n: float


def variables_sample_size(confidence: float, reliability: float, capability_margin_sigma: float,
                           n_min: int = 2, n_max: int = 2000) -> VariablesSampleSizeResult:
    """Smallest n such that the one-sided tolerance factor k(n, C, R) is
    at most `capability_margin_sigma`, the process-capability margin
    (distance from the sample mean to the nearest specification limit,
    expressed in sample standard deviations, e.g. margin=1.5 means the
    spec limit sits 1.5 sample-sigma away from x_bar). Demonstrating
    k(n,C,R) <= margin means: with confidence C, at least proportion R of
    the population lies on the safe side of that spec limit.

    k(n, C, R) is monotonically decreasing in n (more data narrows the
    tolerance interval), so a forward linear search from n_min is exact
    and terminates at the first n meeting the margin.
    """
    if capability_margin_sigma <= 0:
        raise ValueError("capability_margin_sigma must be > 0")
    for n in range(n_min, n_max + 1):
        k = one_sided_k_factor(n, confidence, reliability)
        if k <= capability_margin_sigma:
            return VariablesSampleSizeResult(confidence, reliability, capability_margin_sigma, n, k)
    raise RuntimeError(f"no n <= {n_max} achieves k <= {capability_margin_sigma} at C={confidence}, R={reliability}")
