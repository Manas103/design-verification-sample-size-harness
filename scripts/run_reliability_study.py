"""Runs the full reliability study: simulates right-censored life-test
data for all 6 seeded failure modes, fits each with the censored Weibull
MLE, checks parameter recovery against the injected true values, computes
B10 life with a 90% confidence interval per mode, and ranks the
failure-mode Pareto by ascending B10 life. Prints everything to stdout;
redirect to docs/reliability_output.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dv_harness.reliability import (
    FAILURE_MODES,
    RECOVERY_TOLERANCE,
    fit_weibull_censored_mle,
    fit_weibull_uncensored_naive,
    run_reliability_study,
    simulate_life_test,
)


def main():
    total_units = sum(spec.n_units for spec in FAILURE_MODES)
    print(f"Failure modes: {len(FAILURE_MODES)}, total units on test: {total_units}")
    print(f"Recovery claim: every injected shape and scale recovered within {RECOVERY_TOLERANCE * 100:.0f}%")
    print()

    result = run_reliability_study()

    print("=== Right-censored Weibull MLE parameter recovery ===")
    header = f"{'mode':28s} {'n_units':>8s} {'n_fail':>7s} {'beta_true':>10s} {'beta_hat':>9s} {'beta_err%':>10s} {'eta_true':>9s} {'eta_hat':>8s} {'eta_err%':>9s}"
    print(header)
    print("-" * len(header))
    for spec, recovery in zip(FAILURE_MODES, result.recoveries):
        data = simulate_life_test(spec)
        print(f"{spec.name:28s} {data.n_units:8d} {data.n_failures:7d} "
              f"{recovery.true_shape:10.3f} {recovery.fit_shape:9.4f} {recovery.shape_rel_error * 100:10.2f} "
              f"{recovery.true_scale:9.3f} {recovery.fit_scale:8.4f} {recovery.scale_rel_error * 100:9.2f}")
    print()
    print(f"Max relative error across 6 modes x 2 parameters (12 numbers): {result.max_rel_error * 100:.2f}%")
    print(f"Meets claim (<= {RECOVERY_TOLERANCE * 100:.0f}%): {result.meets_claim}")
    print()

    print("=== Naive uncensored-fit bug, for comparison (see README Findings) ===")
    heavy_censor_spec = next(s for s in FAILURE_MODES if s.name == "electronics_random_failure")
    data = simulate_life_test(heavy_censor_spec)
    naive_beta, naive_eta = fit_weibull_uncensored_naive(data.times, data.event)
    censored_beta, censored_eta = fit_weibull_censored_mle(data.times, data.event)
    print(f"mode: {heavy_censor_spec.name} ({data.n_failures}/{data.n_units} units observed to fail before censor_time={heavy_censor_spec.censor_time})")
    print(f"true:      beta={heavy_censor_spec.true_shape:.4f}  eta={heavy_censor_spec.true_scale:.4f}")
    print(f"naive fit (failures only, censored units discarded): beta={naive_beta:.4f}  eta={naive_eta:.4f}  "
          f"eta_rel_error={abs(naive_eta - heavy_censor_spec.true_scale) / heavy_censor_spec.true_scale * 100:.2f}%")
    print(f"censored MLE (correct):                              beta={censored_beta:.4f}  eta={censored_eta:.4f}  "
          f"eta_rel_error={abs(censored_eta - heavy_censor_spec.true_scale) / heavy_censor_spec.true_scale * 100:.2f}%")
    print()

    print("=== B10 life with 90% confidence bounds ===")
    header2 = f"{'mode':28s} {'B10':>10s} {'CI low (90%)':>13s} {'CI high (90%)':>14s}"
    print(header2)
    print("-" * len(header2))
    for b in result.b10_results:
        print(f"{b.mode:28s} {b.b10:10.4f} {b.ci_low:13.4f} {b.ci_high:14.4f}")
    print()

    print("=== Failure-mode Pareto (ranked by ascending B10 life: shortest B10 = highest priority) ===")
    by_mode = {b.mode: b for b in result.b10_results}
    print(f"{'rank':>4s}  {'mode':28s} {'B10':>10s}")
    for rank, name in enumerate(result.pareto_order, start=1):
        print(f"{rank:4d}  {name:28s} {by_mode[name].b10:10.4f}")
    print()

    print("=== Summary ===")
    print(f"max_param_recovery_error_pct = {result.max_rel_error * 100:.2f}")
    print(f"meets_claim (recovery within {RECOVERY_TOLERANCE * 100:.0f}%) = {result.meets_claim}")
    print(f"pareto top priority failure mode = {result.pareto_order[0]}")


if __name__ == "__main__":
    main()
