"""Runs the combined 24-fault / 30-healthy-study sweep across all three
test methods (crossed gage R&R, nested/destructive gage R&R, attribute
agreement), and the %GRR-of-study-variation vs %GRR-of-tolerance
disagreement check on the crossed scenario set. Prints to stdout;
redirect to docs/measurement_system_fault_sweep_output.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dv_harness.gage_rr import DEFAULT_TOLERANCE, PCT_GRR_THRESHOLD, evaluate_scenario
from dv_harness.gage_rr_scenarios import N_OPERATORS, N_PARTS, N_TRIALS, SCENARIOS
from dv_harness.measurement_system_fault_sweep import run_fault_sweep


def main():
    result = run_fault_sweep()
    print(f"Seeded measurement-system faults caught: {result.n_faults_caught}/{result.n_faults_total}")
    print(f"  (12 crossed gage R&R + 6 nested/destructive gage R&R + 6 attribute agreement)")
    print()
    print(f"False flags over {result.n_healthy_total} healthy studies (10 crossed + 10 nested + 10 attribute): "
          f"{result.n_false_flags}")
    for line in result.false_flag_detail:
        print(f"  {line}")
    print()

    print(f"%GRR of study variation vs %GRR of tolerance (tolerance={DEFAULT_TOLERANCE}), crossed scenario set:")
    header = f"{'scenario':42s} {'%GRR study var':>15s} {'%GRR tolerance':>15s} {'agree':>7s}"
    print(header)
    print("-" * len(header))
    n_disagree = 0
    for s in SCENARIOS:
        r = evaluate_scenario(N_PARTS, N_OPERATORS, N_TRIALS, s.sigma_part, s.sigma_operator,
                               s.sigma_po, s.sigma_repeat, s.seed)
        study_side = r.pct_grr > PCT_GRR_THRESHOLD
        tol_side = r.pct_grr_tolerance > PCT_GRR_THRESHOLD
        agree = study_side == tol_side
        n_disagree += int(not agree)
        print(f"{s.name:42s} {r.pct_grr:15.2f} {r.pct_grr_tolerance:15.2f} {str(agree):>7s}")
    print()
    print(f"Disagreements: {n_disagree}/{len(SCENARIOS)} scenarios classify differently depending on which "
          f"%GRR denominator is used against the same 30% bright line (see README Findings).")


if __name__ == "__main__":
    main()
