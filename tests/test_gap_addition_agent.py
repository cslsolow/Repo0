import sys
from pathlib import Path

import pytest

pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.analysis.gap_addition import (  # noqa: E402
    GapAdditionCandidate,
    GapAdditionDecision,
    GapAdditionJudge,
    fuse_gap_confidence,
    normalize_gap_judge_response,
    propose_gap_candidate_for_parent,
    should_accept_gap_decision,
)


def _parent_architecture():
    return {
        "parent_task": "ReqA",
        "architecture": {
            "requirement": {"name": "ReqA"},
            "sub_requirements": [
                {
                    "name": "proxy environment configuration",
                    "description": "Support proxy environment configuration and runtime resolution.",
                    "order": 0,
                    "status": "active",
                },
                {
                    "name": "response handling",
                    "description": "Handle response parsing and normalization.",
                    "order": 1,
                    "status": "active",
                },
            ],
            "components": [
                {
                    "name": "ResponseHandling",
                    "responsibilities": ["Handle response parsing and normalization."],
                    "serves_subrequirements": ["response handling"],
                    "exports": ["ResponseHandling"],
                }
            ],
        },
    }


def test_propose_gap_candidate_for_parent_emits_missing_subrequirement():
    candidate = propose_gap_candidate_for_parent(
        parent_entry=_parent_architecture(),
        input_text="Support proxy environment configuration and runtime resolution.",
        requirements_payload={"project_summary": "", "requirements": []},
        generated_entries=[],
        realization_report={},
        proposal_threshold=0.55,
        max_candidates_per_parent=1,
    )

    assert isinstance(candidate, GapAdditionCandidate)
    assert candidate.candidate_type == "missing_subrequirement"
    assert candidate.parent_requirement == "ReqA"
    assert candidate.target_subrequirement == "proxy environment configuration"
    assert candidate.proposed_component == "ProxyEnvironmentSupport"


def test_propose_gap_candidate_for_parent_rejects_requirement_add_without_direct_evidence():
    candidate = propose_gap_candidate_for_parent(
        parent_entry={
            "parent_task": "ReqA",
            "architecture": {
                "requirement": {"name": "ReqA"},
                "sub_requirements": [],
                "components": [],
            },
        },
        input_text="General request utilities.",
        requirements_payload={"project_summary": "", "requirements": []},
        generated_entries=[],
        realization_report={},
        proposal_threshold=0.55,
        max_candidates_per_parent=1,
    )

    assert candidate is None


def test_propose_gap_candidate_for_parent_can_emit_missing_requirement_under_parent():
    candidate = propose_gap_candidate_for_parent(
        parent_entry={
            "parent_task": "ReqA",
            "architecture": {
                "requirement": {"name": "ReqA"},
                "sub_requirements": [],
                "components": [],
            },
        },
        input_text="Support proxy environment configuration and runtime resolution.",
        requirements_payload={"project_summary": "", "requirements": []},
        generated_entries=[],
        realization_report={},
        proposal_threshold=0.55,
        max_candidates_per_parent=1,
    )

    assert candidate is not None
    assert candidate.candidate_type == "missing_requirement_under_parent"
    assert candidate.proposed_requirement == "proxy environment configuration"


def test_normalize_gap_judge_response_rejects_requirement_add_without_evidence():
    candidate = GapAdditionCandidate(
        parent_requirement="ReqA",
        candidate_type="missing_subrequirement",
        target_subrequirement="proxy environment configuration",
        proposed_requirement="",
        proposed_component="ProxyEnvironmentSupport",
        heuristic_gap_score=0.8,
        evidence_spans=[],
        reason="gap",
    )

    decision = normalize_gap_judge_response(
        candidate=candidate,
        payload={
            "decision": "accept",
            "action": "add_requirement_and_component",
            "confidence": 0.95,
            "new_requirement": "proxy environment configuration",
            "new_component": "ProxyEnvironmentSupport",
            "served_subrequirements": ["proxy environment configuration"],
        },
    )

    assert decision.decision == "reject"
    assert decision.action == "none"


def test_should_accept_gap_decision_uses_stricter_requirement_threshold():
    candidate = GapAdditionCandidate(
        parent_requirement="ReqA",
        candidate_type="missing_requirement_under_parent",
        target_subrequirement="",
        proposed_requirement="proxy environment configuration",
        proposed_component="ProxyEnvironmentSupport",
        heuristic_gap_score=0.8,
        evidence_spans=["Support proxy environment configuration and runtime resolution."],
        reason="gap",
    )
    decision = GapAdditionDecision(
        decision="accept",
        action="add_requirement_and_component",
        confidence=0.82,
        final_confidence=fuse_gap_confidence(0.8, 0.82),
        parent_requirement="ReqA",
        target_subrequirement="",
        new_requirement="proxy environment configuration",
        new_component="ProxyEnvironmentSupport",
        served_subrequirements=["proxy environment configuration"],
        evidence_spans=candidate.evidence_spans,
        reason="accepted",
    )

    assert should_accept_gap_decision(
        decision,
        add_component_accept_threshold=0.74,
        add_requirement_and_component_accept_threshold=0.82,
    ) is False

    accepted = GapAdditionDecision(
        decision="accept",
        action="add_requirement_and_component",
        confidence=0.9,
        final_confidence=fuse_gap_confidence(0.8, 0.9),
        parent_requirement="ReqA",
        target_subrequirement="",
        new_requirement="proxy environment configuration",
        new_component="ProxyEnvironmentSupport",
        served_subrequirements=["proxy environment configuration"],
        evidence_spans=candidate.evidence_spans,
        reason="accepted",
    )

    assert should_accept_gap_decision(
        accepted,
        add_component_accept_threshold=0.74,
        add_requirement_and_component_accept_threshold=0.82,
    ) is True


def test_gap_addition_judge_prefers_add_component_when_requirement_node_already_exists():
    captured = {}

    class _FakeLLMClient:
        def call_json(self, messages, temperature=0.0, max_tokens=0, **kwargs):
            captured["prompt"] = messages[-1]["content"]
            return {
                "decision": "accept",
                "action": "add_component",
                "confidence": 0.86,
                "new_requirement": "",
                "new_component": "ProxyEnvironmentSupport",
                "served_subrequirements": ["proxy environment configuration"],
                "reason": "existing subrequirement is uncovered",
            }

    judge = GapAdditionJudge(api_config={"api_key": "test"}, output_dir="/tmp")
    judge.llm_client = _FakeLLMClient()
    candidate = GapAdditionCandidate(
        parent_requirement="ReqA",
        candidate_type="missing_subrequirement",
        target_subrequirement="proxy environment configuration",
        proposed_requirement="",
        proposed_component="ProxyEnvironmentSupport",
        heuristic_gap_score=0.8,
        evidence_spans=["Support proxy environment configuration and runtime resolution."],
        reason="gap",
    )

    decision = judge.judge_candidate(
        candidate=candidate,
        parent_entry=_parent_architecture(),
        generated_entries=[],
        realization_report={},
    )

    assert decision.action == "add_component"
    assert "Do not create a new parent requirement" in captured["prompt"]
