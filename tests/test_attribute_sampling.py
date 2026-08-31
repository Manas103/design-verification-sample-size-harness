import math

import pytest

from dv_harness.attribute_sampling import (
    attribute_sample_size,
    c_failure_sample_size,
    zero_failure_sample_size,
)


def test_zero_failure_textbook_95_95():
    # The canonical textbook zero-failure reliability demonstration value.
    assert zero_failure_sample_size(0.95, 0.95) == 59


def test_zero_failure_closed_form_matches_direct_derivation():
    # Independently re-derive from the definition (probability of zero
    # failures in n trials at true reliability R must be <= 1-C) rather
    # than trusting the ceil(ln/ln) shortcut blindly.
    C, R = 0.95, 0.95
    n = zero_failure_sample_size(C, R)
    assert R ** n <= 1 - C
    assert R ** (n - 1) > 1 - C  # n-1 must NOT satisfy the requirement


def test_zero_failure_second_published_pair_90_90():
    # A second, independently checkable (C, R) pair: 90% confidence /
    # 90% reliability zero-failure demonstration test, commonly tabulated
    # as n=22 (e.g. Krishnamoorthi, "Reliability Methods for Engineers").
    assert zero_failure_sample_size(0.90, 0.90) == 22


def test_generalized_c_failure_matches_closed_form_at_c_zero():
    # The generalized incomplete-beta search must exactly reproduce the
    # closed-form zero-failure result: this is the reference-oracle cross
    # check between the two independently implemented code paths.
    for C, R in [(0.95, 0.95), (0.90, 0.90), (0.99, 0.90)]:
        assert c_failure_sample_size(C, R, 0) == zero_failure_sample_size(C, R)


def test_generalized_c_failure_increases_with_allowed_failures():
    # Allowing more failures always costs more samples at fixed C, R.
    ns = [c_failure_sample_size(0.95, 0.95, c) for c in range(4)]
    assert ns == sorted(ns)
    assert ns[0] < ns[1] < ns[2] < ns[3]


def test_public_entry_point_dispatches_correctly():
    r0 = attribute_sample_size(0.95, 0.95, 0)
    assert r0.n == 59
    assert r0.allowed_failures == 0
    r1 = attribute_sample_size(0.95, 0.95, 1)
    assert r1.n > r0.n


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        zero_failure_sample_size(1.5, 0.95)
    with pytest.raises(ValueError):
        zero_failure_sample_size(0.95, 0.0)
    with pytest.raises(ValueError):
        c_failure_sample_size(0.95, 0.95, -1)
