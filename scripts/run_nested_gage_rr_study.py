"""Runs all 7 seeded nested (destructive-test) gage R&R scenarios (6
engineered to fail, 1 engineered to pass), prints the hand-computed
nested ANOVA variance decomposition, the operator-term F-test pooling
decision, and the pass/fail classification for each. Prints everything to
stdout; redirect to docs/nested_gage_rr_output.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dv_harness.nested_gage_rr import ALPHA_POOL, PCT_GRR_THRESHOLD, NDC_THRESHOLD, evaluate_nested_scenario
from dv_harness.nested_gage_rr_scenarios import N_OPERATORS, N_PARTS_PER_OPERATOR, N_TRIALS, NESTED_SCENARIOS


def main():
    print(f"Study shape: {N_OPERATORS} operators x {N_PARTS_PER_OPERATOR} destructive parts each "
          f"x {N_TRIALS} replicate specimens per part ({N_OPERATORS * N_PARTS_PER_OPERATOR * N_TRIALS} measurements)")
    print(f"Acceptance rule: PASS iff %GRR <= {PCT_GRR_THRESHOLD} AND NDC >= {NDC_THRESHOLD}")
    print(f"Operator-term pooling: operator variance is set to 0 unless its F-test vs "
          f"part(operator) is significant at alpha={ALPHA_POOL}")
    print()

    header = f"{'scenario':38s} {'engineered':11s} {'%GRR':>8s} {'NDC':>4s} {'pooled':>7s} {'classified':>11s} {'correct':>8s}"
    print(header)
    print("-" * len(header))

    n_fail_total = 0
    n_fail_caught = 0
    n_pass_total = 0
    n_pass_correct = 0

    for s in NESTED_SCENARIOS:
        result = evaluate_nested_scenario(N_OPERATORS, N_PARTS_PER_OPERATOR, N_TRIALS,
                                           s.sigma_part, s.sigma_operator, s.sigma_repeat, s.seed)
        engineered = "fail" if s.engineered_to_fail else "pass"
        classified = "fail" if not result.passed else "pass"
        correct = (classified == engineered)
        print(f"{s.name:38s} {engineered:11s} {result.pct_grr:8.2f} {result.ndc:4d} "
              f"{str(result.operator_term_pooled):>7s} {classified:>11s} {str(correct):>8s}")
        if s.engineered_to_fail:
            n_fail_total += 1
            n_fail_caught += int(correct)
        else:
            n_pass_total += 1
            n_pass_correct += int(correct)

    print()
    print("Variance components (raw), per scenario:")
    for s in NESTED_SCENARIOS:
        result = evaluate_nested_scenario(N_OPERATORS, N_PARTS_PER_OPERATOR, N_TRIALS,
                                           s.sigma_part, s.sigma_operator, s.sigma_repeat, s.seed)
        print(f"  {s.name}: repeatability_var={result.repeatability_var:.4f} operator_var={result.operator_var:.4f} "
              f"part_var={result.part_var:.4f} grr_var={result.grr_var:.4f} total_var={result.total_var:.4f} "
              f"operator_f_pvalue={result.operator_f_pvalue:.4f} reasons={list(result.reasons)}")

    print()
    print(f"Seeded-failure catch rate: {n_fail_caught}/{n_fail_total}")
    print(f"Seeded-pass correctness: {n_pass_correct}/{n_pass_total}")
    print(f"Overall: {n_fail_caught + n_pass_correct}/{n_fail_total + n_pass_total} scenarios correctly classified")

    print()
    print("Note: unlike the crossed design, no part*operator interaction failure mode is")
    print("representable here (see nested_gage_rr.py docstring); a real interaction problem")
    print("in a destructive-test measurement system would be structurally invisible to this")
    print("study design, not just to this harness.")


if __name__ == "__main__":
    main()
