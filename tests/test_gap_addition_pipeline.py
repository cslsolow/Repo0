import sys
from pathlib import Path

import pytest

pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.analysis.gap_addition import (  # noqa: E402
    GapAdditionCandidate,
    GapAdditionDecision,
    apply_gap_addition_decision,
    run_local_gap_cleanup,
)
from run_agents import run_gap_addition_stage  # noqa: E402


def _parent_entry():
    return {
        "parent_task": "ReqA",
        "architecture": {
            "requirement": {"name": "ReqA"},
            "sub_requirements": [
                {
                    "name": "response handling",
                    "description": "Handle response parsing and normalization.",
                    "order": 0,
                    "status": "active",
                }
            ],
            "components": [
                {
                    "name": "ResponseHandling",
                    "responsibilities": ["Handle response parsing and normalization."],
                    "serves_subrequirements": ["response handling"],
                }
            ],
        },
    }


def test_apply_gap_addition_decision_appends_component_without_new_parent():
    updated = apply_gap_addition_decision(
        parent_entry=_parent_entry(),
        decision=GapAdditionDecision(
            decision="accept",
            action="add_component",
            confidence=0.8,
            final_confidence=0.79,
            parent_requirement="ReqA",
            target_subrequirement="proxy environment configuration",
            new_requirement="",
            new_component="ProxyEnvironmentSupport",
            served_subrequirements=["proxy environment configuration"],
            evidence_spans=["Support proxy environment configuration and runtime resolution."],
            reason="accepted",
        ),
    )

    names = [component["name"] for component in updated["architecture"]["components"]]
    assert names == ["ResponseHandling", "ProxyEnvironmentSupport"]
    assert updated["parent_task"] == "ReqA"


def test_apply_gap_addition_decision_can_add_subrequirement_under_existing_parent():
    updated = apply_gap_addition_decision(
        parent_entry=_parent_entry(),
        decision=GapAdditionDecision(
            decision="accept",
            action="add_requirement_and_component",
            confidence=0.9,
            final_confidence=0.86,
            parent_requirement="ReqA",
            target_subrequirement="",
            new_requirement="proxy environment configuration",
            new_component="ProxyEnvironmentSupport",
            served_subrequirements=["proxy environment configuration"],
            evidence_spans=["Support proxy environment configuration and runtime resolution."],
            reason="accepted",
        ),
    )

    subreq_names = [item["name"] for item in updated["architecture"]["sub_requirements"]]
    assert "proxy environment configuration" in subreq_names


def test_run_local_gap_cleanup_scopes_refinement_to_single_parent():
    calls = []

    def _fake_metric_actions(architectures, actions, **kwargs):
        calls.append(("metric", [item["parent_task"] for item in architectures]))
        return actions, {"stats": {"split_upgrades": 0, "merge_candidates": 0, "merge_upgrades": 0}}

    def _fake_refinement(architectures, component_merge_agent, component_split_agent):
        calls.append(("refine", [item["parent_task"] for item in architectures]))
        return architectures, {"stats": {"components_after": 2, "merge_group_count": 0, "split_group_count": 0}}

    cleaned, report = run_local_gap_cleanup(
        parent_entry=_parent_entry(),
        component_merge_agent=None,
        component_split_agent=None,
        augment_actions_with_component_metrics_fn=_fake_metric_actions,
        apply_action_guided_structure_refinement_fn=_fake_refinement,
        split_cohesion_threshold=2.0 / 3.0,
        split_min_subrequirements=4,
        merge_max_small_subrequirements=1,
    )

    assert [component["name"] for component in cleaned["architecture"]["components"]] == ["ResponseHandling"]
    assert calls == [("metric", ["ReqA"]), ("refine", ["ReqA"])]
    assert report["parent_task"] == "ReqA"


def test_run_gap_addition_stage_returns_original_architectures_when_disabled(tmp_path):
    architectures = [_parent_entry()]

    class _Args:
        enable_gap_add_actions = False
        gap_add_proposal_threshold = 0.55
        gap_add_component_threshold = 0.74
        gap_add_requirement_threshold = 0.82
        component_metric_split_cohesion_threshold = 2.0 / 3.0
        component_metric_split_min_subrequirements = 3
        component_metric_merge_max_small_subrequirements = 1

    updated, report = run_gap_addition_stage(
        architectures=architectures,
        args=_Args(),
        output_dir=tmp_path,
        input_text="Support proxy environment configuration and runtime resolution.",
        requirements_payload={"project_summary": "", "requirements": []},
        generated_entries=[],
        realization_report={},
        component_merge_agent=None,
        component_split_agent=None,
    )

    assert updated == architectures
    assert report["enabled"] is False


def test_run_gap_addition_stage_accepts_component_add_and_writes_report(tmp_path):
    architectures = [_parent_entry()]

    class _Args:
        enable_gap_add_actions = True
        gap_add_proposal_threshold = 0.55
        gap_add_component_threshold = 0.74
        gap_add_requirement_threshold = 0.82
        component_metric_split_cohesion_threshold = 2.0 / 3.0
        component_metric_split_min_subrequirements = 3
        component_metric_merge_max_small_subrequirements = 1

    def _fake_proposer(**kwargs):
        return GapAdditionCandidate(
            parent_requirement="ReqA",
            candidate_type="missing_subrequirement",
            target_subrequirement="proxy environment configuration",
            proposed_requirement="",
            proposed_component="ProxyEnvironmentSupport",
            heuristic_gap_score=0.8,
            evidence_spans=["Support proxy environment configuration and runtime resolution."],
            reason="gap",
        )

    def _fake_judge(**kwargs):
        return GapAdditionDecision(
            decision="accept",
            action="add_component",
            confidence=0.82,
            final_confidence=0.811,
            parent_requirement="ReqA",
            target_subrequirement="proxy environment configuration",
            new_requirement="",
            new_component="ProxyEnvironmentSupport",
            served_subrequirements=["proxy environment configuration"],
            evidence_spans=["Support proxy environment configuration and runtime resolution."],
            reason="accepted",
        )

    updated, report = run_gap_addition_stage(
        architectures=architectures,
        args=_Args(),
        output_dir=tmp_path,
        input_text="Support proxy environment configuration and runtime resolution.",
        requirements_payload={"project_summary": "", "requirements": []},
        generated_entries=[],
        realization_report={},
        component_merge_agent=None,
        component_split_agent=None,
        propose_gap_candidate_for_parent_fn=_fake_proposer,
        judge_gap_candidate_fn=_fake_judge,
        run_local_gap_cleanup_fn=lambda **kwargs: (kwargs["parent_entry"], {"parent_task": "ReqA"}),
    )

    assert [component["name"] for component in updated[0]["architecture"]["components"]] == [
        "ResponseHandling",
        "ProxyEnvironmentSupport",
    ]
    assert (tmp_path / "gap_addition_report.json").exists()
    assert report["accepted_count"] == 1
