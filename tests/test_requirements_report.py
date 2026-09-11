import pytest

from dv_harness.demo_requirements import REQUIREMENTS, build_evidence
from dv_harness.requirements_report import Evidence, Requirement, build_report, evaluate_requirement


def _req(rid="REQ-X"):
    return Requirement(id=rid, description="demo requirement", acceptance_criterion="demo criterion")


def test_refuses_when_no_evidence_attached():
    status = evaluate_requirement(_req(), [])
    assert status.verified is False
    assert status.status_label == "NOT VERIFIED"
    assert "no evidence attached" in status.reason


def test_refuses_when_evidence_fails():
    ev = [Evidence("REQ-X", passed=False, detail="measured value out of tolerance", source="test")]
    status = evaluate_requirement(_req(), ev)
    assert status.verified is False
    assert status.status_label == "NOT VERIFIED"
    assert "failing evidence present" in status.reason


def test_refuses_when_any_evidence_fails_even_if_others_pass():
    ev = [
        Evidence("REQ-X", passed=True, detail="ok", source="test"),
        Evidence("REQ-X", passed=False, detail="not ok", source="test"),
    ]
    status = evaluate_requirement(_req(), ev)
    assert status.verified is False


def test_verifies_when_all_evidence_passes():
    ev = [Evidence("REQ-X", passed=True, detail="matches reference", source="test")]
    status = evaluate_requirement(_req(), ev)
    assert status.verified is True
    assert status.status_label == "VERIFIED"


def test_demo_requirement_set_has_exactly_one_requirement_without_evidence():
    evidence = build_evidence()
    missing = [r.id for r in REQUIREMENTS if r.id not in evidence or len(evidence.get(r.id, [])) == 0]
    assert missing == ["REQ-005"]


def test_demo_report_actually_refuses_req_005():
    evidence = build_evidence()
    report = build_report(REQUIREMENTS, evidence)
    by_id = {s.requirement.id: s for s in report}
    assert by_id["REQ-005"].verified is False
    assert by_id["REQ-005"].status_label == "NOT VERIFIED"
    assert by_id["REQ-005"].reason == "no evidence attached"


def test_demo_report_verifies_the_others_that_have_passing_evidence():
    evidence = build_evidence()
    report = build_report(REQUIREMENTS, evidence)
    by_id = {s.requirement.id: s for s in report}
    for rid in ["REQ-001", "REQ-002", "REQ-003", "REQ-004", "REQ-007"]:
        assert by_id[rid].verified is True, f"{rid} expected VERIFIED, got {by_id[rid].reason}"


def test_demo_report_req_006_is_not_verified_for_a_real_measured_reason_not_missing_evidence():
    # REQ-006's evidence is genuinely attached (unlike REQ-005), it just
    # honestly fails: the real measured reliability parameter-recovery
    # error exceeds the 6% target (see README Findings). This is a
    # different refusal path than REQ-005's "no evidence attached" and
    # must not be confused with it.
    evidence = build_evidence()
    report = build_report(REQUIREMENTS, evidence)
    by_id = {s.requirement.id: s for s in report}
    assert len(evidence["REQ-006"]) == 1
    assert by_id["REQ-006"].verified is False
    assert "failing evidence present" in by_id["REQ-006"].reason
    assert by_id["REQ-006"].reason != "no evidence attached"


def test_missing_key_in_evidence_dict_treated_as_no_evidence():
    report = build_report([_req("REQ-Y")], {})
    assert report[0].verified is False
    assert report[0].reason == "no evidence attached"
