"""Runs all 7 seeded attribute agreement scenarios (6 engineered to fail,
1 engineered to pass), prints observed percent agreement, Cohen's kappa,
and the pass/fail classification for each. Prints everything to stdout;
redirect to docs/attribute_agreement_output.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dv_harness.attribute_agreement import KAPPA_THRESHOLD, evaluate_attribute_study
from dv_harness.attribute_agreement_scenarios import ATTRIBUTE_SCENARIOS


def main():
    print(f"Acceptance rule: PASS iff kappa >= {KAPPA_THRESHOLD}")
    print()

    header = f"{'scenario':30s} {'engineered':11s} {'n_defect':>9s} {'%agree':>7s} {'kappa':>8s} {'classified':>11s} {'correct':>8s}"
    print(header)
    print("-" * len(header))

    n_fail_total = 0
    n_fail_caught = 0
    n_pass_total = 0
    n_pass_correct = 0

    for s in ATTRIBUTE_SCENARIOS:
        result = evaluate_attribute_study(s.n_parts, s.true_defect_rate, s.sensitivity, s.specificity, s.seed)
        engineered = "fail" if s.engineered_to_fail else "pass"
        classified = "fail" if not result.passed else "pass"
        correct = (classified == engineered)
        print(f"{s.name:30s} {engineered:11s} {result.n_defect_truth:9d} "
              f"{result.percent_agreement * 100:6.1f}% {result.kappa:8.4f} {classified:>11s} {str(correct):>8s}")
        if s.engineered_to_fail:
            n_fail_total += 1
            n_fail_caught += int(correct)
        else:
            n_pass_total += 1
            n_pass_correct += int(correct)

    print()
    print(f"Seeded-failure catch rate: {n_fail_caught}/{n_fail_total}")
    print(f"Seeded-pass correctness: {n_pass_correct}/{n_pass_total}")
    print(f"Overall: {n_fail_caught + n_pass_correct}/{n_fail_total + n_pass_total} scenarios correctly classified")


if __name__ == "__main__":
    main()
