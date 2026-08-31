"""Attribute (binomial pass/fail) reliability demonstration sample size.

Answers: "how many units must I test, with c or fewer allowed failures, to
demonstrate at confidence C that the true reliability is at least R?"

Zero-failure case (c = 0): the standard "success-run" closed form

    n = ceil( ln(1 - C) / ln(R) )

comes directly from requiring the one-sided Clopper-Pearson upper
confidence bound on the failure probability, after observing zero
failures in n trials, to be no greater than 1 - R:

    P(zero failures in n trials | true failure prob = 1-R) = R**n <= 1 - C
    => n >= ln(1-C) / ln(R)          (ln(R) < 0, so the inequality flips)

Generalized case (c > 0 allowed failures): there is no single named closed
form as compact as the zero-failure one, but it reduces to the same
Clopper-Pearson idea via the classical binomial/incomplete-beta identity

    P(X <= c | X ~ Binomial(n, p)) = I_{1-p}(n - c, c + 1) = 1 - I_p(c + 1, n - c)

so the (1 - alpha) upper confidence bound on the failure probability p,
after observing c failures in n trials, is the C-quantile of a
Beta(c + 1, n - c) distribution (this is exactly `scipy.stats.beta.ppf`,
not a re-implementation of the incomplete beta function). We search for
the smallest n for which that upper bound is at most 1 - R. Reference:
NIST/SEMATECH e-Handbook of Statistical Methods section 8.2.1.2("Choice
of confidence intervals for a binomial proportion") and the classical
"success-run theorem" for reliability demonstration tests (e.g.
Krishnamoorthi, "Reliability Methods for Engineers", or MIL-HDBK-108
zero/c-failure sampling plans). At c = 0 this generalized search must
reproduce the closed-form n exactly; that reproduction is checked in
tests/test_attribute_sampling.py and in scripts/run_sample_size_reference_check.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import beta


@dataclass(frozen=True)
class AttributeSampleSizeResult:
    confidence: float
    reliability: float
    allowed_failures: int
    n: int
    method: str


def zero_failure_sample_size(confidence: float, reliability: float) -> int:
    """Closed-form zero-failure reliability demonstration sample size.

    n = ceil( ln(1 - C) / ln(R) )
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    if not (0.0 < reliability < 1.0):
        raise ValueError("reliability must be in (0, 1)")
    n = math.log(1.0 - confidence) / math.log(reliability)
    return math.ceil(n)


def c_failure_sample_size(confidence: float, reliability: float, allowed_failures: int,
                           n_max: int = 100_000) -> int:
    """Smallest n such that, if exactly `allowed_failures` failures are
    observed in n trials, the Clopper-Pearson upper confidence bound (at
    confidence `confidence`) on the true failure probability is <=
    1 - reliability.

    Implemented as a direct search using scipy.stats.beta.ppf, which is
    the reference computation for the Clopper-Pearson bound (not a
    re-derivation of the incomplete beta function by this project).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    if not (0.0 < reliability < 1.0):
        raise ValueError("reliability must be in (0, 1)")
    if allowed_failures < 0:
        raise ValueError("allowed_failures must be >= 0")

    c = allowed_failures
    max_allowed_failure_prob = 1.0 - reliability
    n = c + 1
    while n <= n_max:
        p_upper = beta.ppf(confidence, c + 1, n - c)
        if p_upper <= max_allowed_failure_prob:
            return n
        n += 1
    raise RuntimeError(f"no sample size <= {n_max} satisfies C={confidence}, R={reliability}, c={c}")


def attribute_sample_size(confidence: float, reliability: float, allowed_failures: int = 0) -> AttributeSampleSizeResult:
    """Public entry point: dispatches to the closed form at c=0 and the
    generalized incomplete-beta search at c>0."""
    if allowed_failures == 0:
        n = zero_failure_sample_size(confidence, reliability)
        method = "zero-failure closed form: ceil(ln(1-C)/ln(R))"
    else:
        n = c_failure_sample_size(confidence, reliability, allowed_failures)
        method = "generalized c-failure search: smallest n with Beta(c+1, n-c).ppf(C) <= 1-R"
    return AttributeSampleSizeResult(confidence, reliability, allowed_failures, n, method)
