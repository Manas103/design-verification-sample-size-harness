"""Runs all 13 seeded gage R&R scenarios (12 engineered to fail, 1
engineered to pass), prints the ANOVA-method variance decomposition and
pass/fail classification for each, and reports the honest measured catch
rate. Prints everything to stdout; redirect to docs/gage_rr_output.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dv_harness.gage_rr import PCT_GRR_THRESHOLD, NDC_THRESHOLD, evaluate_scenario
from dv_harness.gage_rr_scenarios import N_OPERATORS, N_PARTS, N_TRIALS, SCENARIOS


def main():
    print(f"Study shape: {N_PARTS} parts x {N_OPERATORS} operators x {N_TRIALS} trials")
    print(f"Acceptance rule: PASS iff %GRR <= {PCT_GRR_THRESHOLD} AND NDC >= {NDC_THRESHOLD}")
    print()

    header = f"{'scenario':42s} {'engineered':11s} {'%GRR':>8s} {'NDC':>4s} {'classified':>11s} {'correct':>8s}"
    print(header)
    print("-" * len(header))

    n_fail_total = 0
    n_fail_caught = 0
    n_pass_total = 0
    n_pass_correct = 0

    for s in SCENARIOS:
        result = evaluate_scenario(N_PARTS, N_OPERATORS, N_TRIALS, s.sigma_part, s.sigma_operator,
                                    s.sigma_po, s.sigma_repeat, s.seed)
        engineered = "fail" if s.engineered_to_fail else "pass"
        classified = "fail" if not result.passed else "pass"
        correct = (classified == engineered)
        print(f"{s.name:42s} {engineered:11s} {result.pct_grr:8.2f} {result.ndc:4d} {classified:>11s} {str(correct):>8s}")
        if s.engineered_to_fail:
            n_fail_total += 1
            n_fail_caught += int(correct)
        else:
            n_pass_total += 1
            n_pass_correct += int(correct)

    print()
    print("Variance components (raw), per scenario:")
    for s in SCENARIOS:
        result = evaluate_scenario(N_PARTS, N_OPERATORS, N_TRIALS, s.sigma_part, s.sigma_operator,
                                    s.sigma_po, s.sigma_repeat, s.seed)
        print(f"  {s.name}: repeatability_var={result.repeatability_var:.4f} operator_var={result.operator_var:.4f} "
              f"interaction_var={result.interaction_var:.4f} part_var={result.part_var:.4f} "
              f"grr_var={result.grr_var:.4f} total_var={result.total_var:.4f} reasons={list(result.reasons)}")

    print()
    print(f"Seeded-failure catch rate: {n_fail_caught}/{n_fail_total}")
    print(f"Seeded-pass correctness: {n_pass_correct}/{n_pass_total}")
    print(f"Overall: {n_fail_caught + n_pass_correct}/{n_fail_total + n_pass_total} scenarios correctly classified")


if __name__ == "__main__":
    main()
