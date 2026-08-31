import math

import pytest

from dv_harness.variables_sampling import (
    howe_approximation_k_factor,
    one_sided_k_factor,
    variables_sample_size,
)

# Published one-sided normal tolerance factors k(n, confidence=0.95,
# proportion=0.90), from Natrella, "Experimental Statistics" (NBS
# Handbook 91, 1963), Table A-6, reproduced in the NIST/SEMATECH
# e-Handbook of Statistical Methods, section 7.2.6.3.
PUBLISHED_K_95_90 = {
    2: 20.581,
    5: 3.407,
    10: 2.355,
}

# A third published reference point at proportion=0.90, confidence=0.95.
PUBLISHED_K_95_90_N20 = 1.926


def test_k_factor_matches_published_table_n2():
    k = one_sided_k_factor(2, 0.95, 0.90)
    assert k == pytest.approx(PUBLISHED_K_95_90[2], abs=0.01)


def test_k_factor_matches_published_table_n5():
    k = one_sided_k_factor(5, 0.95, 0.90)
    assert k == pytest.approx(PUBLISHED_K_95_90[5], abs=0.005)


def test_k_factor_matches_published_table_n10():
    k = one_sided_k_factor(10, 0.95, 0.90)
    assert k == pytest.approx(PUBLISHED_K_95_90[10], abs=0.005)


def test_k_factor_matches_published_table_n20():
    k = one_sided_k_factor(20, 0.95, 0.90)
    assert k == pytest.approx(PUBLISHED_K_95_90_N20, abs=0.005)


def test_k_factor_decreases_with_n():
    ks = [one_sided_k_factor(n, 0.95, 0.95) for n in [5, 10, 20, 50, 100]]
    assert ks == sorted(ks, reverse=True)


def test_k_factor_approaches_z_reliability_as_n_grows():
    # k(n,C,R) -> z_R as n -> infinity, but the convergence is slow
    # (the leading correction term is O(z_C/sqrt(n))), so this needs a
    # genuinely large n, not just n=5000, to land within a tight bound.
    from scipy.stats import norm
    k_large = one_sided_k_factor(200_000, 0.95, 0.90)
    assert k_large == pytest.approx(norm.ppf(0.90), abs=0.005)


def test_howe_approximation_cross_checks_exact_noncentral_t():
    # Independent second formula (not derived from nct machinery) used
    # as a genuine reference oracle, not a self-check of the same code.
    # The Howe approximation is known to be less accurate at very small n
    # (n=10 lands at ~1.4% relative difference here, still checked, but
    # with a wider tolerance than the larger-n cases; see README
    # Limitations for this boundary case).
    for n, C, R in [(30, 0.95, 0.95), (50, 0.95, 0.90)]:
        exact = one_sided_k_factor(n, C, R)
        approx = howe_approximation_k_factor(n, C, R)
        assert exact == pytest.approx(approx, rel=0.01)

    exact_n10 = one_sided_k_factor(10, 0.95, 0.90)
    approx_n10 = howe_approximation_k_factor(10, 0.95, 0.90)
    assert exact_n10 == pytest.approx(approx_n10, rel=0.02)


def test_variables_sample_size_search_is_self_consistent():
    result = variables_sample_size(0.95, 0.95, capability_margin_sigma=2.5)
    assert result.k_at_n <= 2.5
    # n-1 must NOT meet the margin, otherwise the search did not find the minimum.
    k_prev = one_sided_k_factor(result.n - 1, 0.95, 0.95)
    assert k_prev > 2.5


def test_variables_sample_size_needs_far_fewer_than_attribute_zero_failure():
    from dv_harness.attribute_sampling import zero_failure_sample_size
    variables_n = variables_sample_size(0.95, 0.95, capability_margin_sigma=2.5).n
    attribute_n = zero_failure_sample_size(0.95, 0.95)
    assert variables_n < attribute_n


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        one_sided_k_factor(1, 0.95, 0.95)
    with pytest.raises(ValueError):
        variables_sample_size(0.95, 0.95, capability_margin_sigma=-1.0)
