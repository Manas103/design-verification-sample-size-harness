"""Builds real evidence from the attribute, variables, and gage R&R
modules, attaches it to the demo requirement set, and renders the
per-requirement VERIFIED / NOT VERIFIED report. REQ-005 is deliberately
left without evidence to prove the refusal path fires for real. Prints
to stdout; redirect to docs/requirement_report_output.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dv_harness.demo_requirements import REQUIREMENTS, build_evidence
from dv_harness.requirements_report import build_report, render_report_text


def main():
    evidence = build_evidence()
    report = build_report(REQUIREMENTS, evidence)
    print(render_report_text(report))
    print()
    req_005 = next(s for s in report if s.requirement.id == "REQ-005")
    print(f"Refusal proof: REQ-005 has 0 attached evidence items and is reported as "
          f"'{req_005.status_label}' (verified={req_005.verified}), reason: {req_005.reason}")


if __name__ == "__main__":
    main()
