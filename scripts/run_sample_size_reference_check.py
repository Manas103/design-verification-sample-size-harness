"""Computes the attribute and variables sample sizes at 95% confidence /
95% reliability, and checks both against published closed-form / table
reference values. Prints everything to stdout; redirect to
docs/sample_size_reference_check.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dv_harness.attribute_sampling import attribute_sample_size, zero_failure_sample_size, c_failure_sample_size
from dv_harness.variables_sampling import howe_approximation_k_factor, one_sided_k_factor, variables_sample_size

PUBLISHED_K = {
    # (n, confidence, proportion) -> published k
    (2, 0.95, 0.90): 20.581,
    (5, 0.95, 0.90): 3.407,
    (10, 0.95, 0.90): 2.355,
    (20, 0.95, 0.90): 1.926,
}


def line(s=""):
    print(s)


def main():
    line("=== Attribute (binomial) sample size, zero-failure, 95% confidence / 95% reliability ===")
    n = zero_failure_sample_size(0.95, 0.95)
    line(f"zero_failure_sample_size(C=0.95, R=0.95) = {n}")
    line("published textbook value: n = 59 (standard success-run zero-failure reliability demonstration test)")
    line(f"match: {n == 59}")
    line()

    line("=== Attribute sample size, second (C,R) pair, zero-failure, 90% / 90% ===")
    n2 = zero_failure_sample_size(0.90, 0.90)
    line(f"zero_failure_sample_size(C=0.90, R=0.90) = {n2}")
    line("published reference value: n = 22 (Krishnamoorthi, Reliability Methods for Engineers)")
    line(f"match: {n2 == 22}")
    line()

    line("=== Generalized c-allowed-failure attribute sample size, 95%/95%, cross-check vs closed form at c=0 ===")
    for c in range(4):
        r = attribute_sample_size(0.95, 0.95, c)
        line(f"c={c}: n={r.n}  (method: {r.method})")
    n_c0 = c_failure_sample_size(0.95, 0.95, 0)
    line(f"c=0 generalized search reproduces closed form exactly: {n_c0} == {n} -> {n_c0 == n}")
    line()

    line("=== Variables (normal tolerance interval) k-factor vs published table values ===")
    line("Source: Natrella, 'Experimental Statistics' (NBS Handbook 91, 1963), Table A-6,")
    line("one-sided tolerance factors, reproduced in NIST/SEMATECH e-Handbook of Statistical")
    line("Methods, section 7.2.6.3.")
    max_abs_diff = 0.0
    for (nn, C, P), published in PUBLISHED_K.items():
        computed = one_sided_k_factor(nn, C, P)
        diff = abs(computed - published)
        max_abs_diff = max(max_abs_diff, diff)
        line(f"n={nn:4d} C={C} P={P}: computed k={computed:.4f}  published k={published:.4f}  |diff|={diff:.4f}")
    line(f"max absolute discrepancy across {len(PUBLISHED_K)} published reference points: {max_abs_diff:.4f}")
    line()

    line("=== Independent second formula cross-check: Howe (1969) approximation vs exact noncentral-t ===")
    for (nn, C, P) in [(10, 0.95, 0.90), (30, 0.95, 0.95), (50, 0.95, 0.90)]:
        exact = one_sided_k_factor(nn, C, P)
        approx = howe_approximation_k_factor(nn, C, P)
        rel = abs(exact - approx) / exact
        line(f"n={nn:4d} C={C} P={P}: exact(nct)={exact:.4f}  howe_approx={approx:.4f}  rel_diff={rel:.5f}")
    line()

    line("=== Variables sample size at 95%/95%, process-capability margin search ===")
    result = variables_sample_size(0.95, 0.95, capability_margin_sigma=2.5)
    line(f"variables_sample_size(C=0.95, R=0.95, margin=2.5 sigma) -> n={result.n} (k at n = {result.k_at_n:.4f})")
    line(f"compare: attribute zero-failure sample size at the same C,R is n={n} (variables needs far fewer units)")
    line()

    line("=== Summary ===")
    line(f"attribute n (95%/95%, zero-failure) = {n}")
    line(f"variables n (95%/95%, margin=2.5 sigma) = {result.n}")
    line(f"max |computed - published| k-factor discrepancy = {max_abs_diff:.4f}")


if __name__ == "__main__":
    main()
