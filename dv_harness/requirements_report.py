"""Per-requirement verification report.

A requirement is marked VERIFIED if and only if it has at least one piece
of evidence attached and every piece of attached evidence passed. Any
other state (no evidence attached at all, or at least one attached piece
of evidence that failed) is reported as NOT VERIFIED with an explicit
reason. There is no silent-pass path: a requirement with zero evidence is
never marked VERIFIED just because nothing contradicted it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Requirement:
    id: str
    description: str
    acceptance_criterion: str


@dataclass(frozen=True)
class Evidence:
    requirement_id: str
    passed: bool
    detail: str
    source: str


@dataclass(frozen=True)
class RequirementStatus:
    requirement: Requirement
    verified: bool
    status_label: str  # "VERIFIED" or "NOT VERIFIED"
    reason: str
    evidence: tuple


def evaluate_requirement(requirement: Requirement, evidence_list) -> RequirementStatus:
    evidence_list = tuple(evidence_list)
    if len(evidence_list) == 0:
        return RequirementStatus(
            requirement=requirement, verified=False, status_label="NOT VERIFIED",
            reason="no evidence attached", evidence=evidence_list,
        )
    failing = [e for e in evidence_list if not e.passed]
    if failing:
        detail = "; ".join(f"{e.source}: {e.detail}" for e in failing)
        return RequirementStatus(
            requirement=requirement, verified=False, status_label="NOT VERIFIED",
            reason=f"failing evidence present ({detail})", evidence=evidence_list,
        )
    return RequirementStatus(
        requirement=requirement, verified=True, status_label="VERIFIED",
        reason="all attached evidence passed", evidence=evidence_list,
    )


def build_report(requirements, evidence_by_requirement_id) -> list:
    """requirements: list[Requirement]. evidence_by_requirement_id:
    dict[str, list[Evidence]] (a missing key is treated as an empty list,
    i.e. no evidence attached)."""
    report = []
    for req in requirements:
        evidence = evidence_by_requirement_id.get(req.id, [])
        report.append(evaluate_requirement(req, evidence))
    return report


def render_report_text(report) -> str:
    lines = []
    for status in report:
        r = status.requirement
        lines.append(f"[{status.status_label}] {r.id}: {r.description}")
        lines.append(f"    acceptance criterion: {r.acceptance_criterion}")
        lines.append(f"    reason: {status.reason}")
        if status.evidence:
            for e in status.evidence:
                mark = "PASS" if e.passed else "FAIL"
                lines.append(f"    evidence [{mark}] ({e.source}): {e.detail}")
        else:
            lines.append("    evidence: (none attached)")
        lines.append("")
    n_verified = sum(1 for s in report if s.verified)
    lines.append(f"summary: {n_verified}/{len(report)} requirements VERIFIED")
    return "\n".join(lines)
