import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_refine_from_architecture.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_refine_from_architecture", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_from_existing_architecture_writes_refinement_outputs_without_rebuilding_inputs(tmp_path: Path):
    module = _load_script_module()

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    architectures = [
        {
            "parent_task": "ReqA",
            "task": "ReqA",
            "architecture": {
                "requirement": {"name": "ReqA"},
                "sub_requirements": [
                    {"name": "ReqA::core", "description": "core"},
                    {"name": "ReqA::api", "description": "api"},
                ],
                "components": [
                    {
                        "name": "Alpha",
                        "responsibilities": ["core + api"],
                        "serves_subrequirements": ["ReqA::core", "ReqA::api"],
                    }
                ],
            },
        }
    ]
    decomposed_dag = {
        "nodes": [
            {"name": "ReqA::core", "description": "core"},
            {"name": "ReqA::api", "description": "api"},
        ],
        "adjacency": {
            "ReqA::core": ["ReqA::api"],
            "ReqA::api": [],
        },
    }
    (input_dir / "architectures.json").write_text(json.dumps(architectures), encoding="utf-8")
    (input_dir / "decomposed_dag.json").write_text(json.dumps(decomposed_dag), encoding="utf-8")
    requirements_file = tmp_path / "requirements.json"
    requirements_file.write_text(json.dumps({"project_summary": "summary", "requirements": [{"name": "ReqA"}]}), encoding="utf-8")
    req_path = tmp_path / "README.req"
    req_path.write_text("dummy req\n", encoding="utf-8")

    args = SimpleNamespace(
        repo="requests",
        input_dir=input_dir,
        output_dir=output_dir,
        requirements_file=requirements_file,
        req_path=req_path,
        base_url="https://example.invalid/v1",
        api_key="",
        model="gpt-5-mini",
        reasoning_effort="medium",
        max_workers=2,
        action_refinement_rounds=2,
        action_refinement_stop_on_stable=True,
        action_refinement_save_stops_component=False,
        enable_component_metric_actions=False,
        enable_component_metric_merge_judge=False,
        component_metric_split_cohesion_threshold=2.0 / 3.0,
        component_metric_split_min_subrequirements=3,
        component_split_min_confidence=0.7,
        component_metric_merge_max_small_subrequirements=1,
        enable_gap_add_actions=False,
        gap_add_proposal_threshold=0.55,
        gap_add_component_threshold=0.74,
        gap_add_requirement_threshold=0.82,
    )

    def _fake_choose_actions(architectures, api_config, output_dir, max_workers):
        return [{"task": "ReqA", "actions": [{"component": "Alpha", "action": "split", "rationale": "too broad"}]}]

    def _fake_feedback_rounds(**kwargs):
        refined = [
            {
                "parent_task": "ReqA",
                "task": "ReqA",
                "architecture": {
                    "requirement": {"name": "ReqA"},
                    "sub_requirements": [
                        {"name": "ReqA::core", "description": "core"},
                        {"name": "ReqA::api", "description": "api"},
                    ],
                    "components": [
                        {"name": "AlphaCore", "serves_subrequirements": ["ReqA::core"]},
                        {"name": "AlphaAPI", "serves_subrequirements": ["ReqA::api"]},
                    ],
                },
            }
        ]
        final_actions = [{"task": "ReqA", "actions": [{"component": "Alpha", "action": "split", "rationale": "too broad"}]}]
        report = {"stats": {"components_after": 2, "merge_group_count": 0, "split_group_count": 1}}
        return refined, final_actions, report

    summary = module.run_from_existing_architecture(
        args=args,
        choose_actions_fn=_fake_choose_actions,
        feedback_rounds_fn=_fake_feedback_rounds,
        gap_add_stage_fn=lambda **kwargs: (kwargs["architectures"], {"enabled": False, "accepted_count": 0, "parents": []}),
    )

    assert summary["refined_parent_count"] == 1
    assert (output_dir / "actions.json").exists()
    assert (output_dir / "action_refinement_report.json").exists()
    assert (output_dir / "architectures.json").exists()
    assert (output_dir / "architectures_flattened.json").exists()
    assert not (output_dir / "requirements_merge_result.json").exists()
    assert not (output_dir / "requirements_for_dag.json").exists()


def test_run_from_existing_architecture_uses_metric_actions_as_initial_structural_source(tmp_path: Path):
    module = _load_script_module()

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    architectures = [
        {
            "parent_task": "ReqA",
            "task": "ReqA",
            "architecture": {
                "requirement": {"name": "ReqA"},
                "sub_requirements": [
                    {"name": "ReqA::core", "description": "core"},
                    {"name": "ReqA::api", "description": "api"},
                    {"name": "ReqA::ops", "description": "ops"},
                ],
                "components": [
                    {
                        "name": "Wide",
                        "responsibilities": ["core + api + ops"],
                        "serves_subrequirements": ["ReqA::core", "ReqA::api", "ReqA::ops"],
                    }
                ],
            },
        }
    ]
    decomposed_dag = {
        "nodes": [
            {"name": "ReqA::core", "description": "core"},
            {"name": "ReqA::api", "description": "api"},
            {"name": "ReqA::ops", "description": "ops"},
        ],
        "adjacency": {
            "ReqA::core": ["ReqA::api"],
            "ReqA::api": ["ReqA::ops"],
            "ReqA::ops": [],
        },
    }
    (input_dir / "architectures.json").write_text(json.dumps(architectures), encoding="utf-8")
    (input_dir / "decomposed_dag.json").write_text(json.dumps(decomposed_dag), encoding="utf-8")
    requirements_file = tmp_path / "requirements.json"
    requirements_file.write_text(json.dumps({"project_summary": "summary", "requirements": [{"name": "ReqA"}]}), encoding="utf-8")
    req_path = tmp_path / "README.req"
    req_path.write_text("dummy req\n", encoding="utf-8")

    args = SimpleNamespace(
        repo="requests",
        input_dir=input_dir,
        output_dir=output_dir,
        requirements_file=requirements_file,
        req_path=req_path,
        base_url="https://example.invalid/v1",
        api_key="",
        model="gpt-5-mini",
        reasoning_effort="medium",
        max_workers=2,
        action_refinement_rounds=2,
        action_refinement_stop_on_stable=True,
        action_refinement_save_stops_component=False,
        enable_component_metric_actions=True,
        enable_component_metric_merge_judge=False,
        component_metric_split_cohesion_threshold=2.0 / 3.0,
        component_metric_split_min_subrequirements=3,
        component_split_min_confidence=0.7,
        component_metric_merge_max_small_subrequirements=1,
        enable_gap_add_actions=False,
        gap_add_proposal_threshold=0.55,
        gap_add_component_threshold=0.74,
        gap_add_requirement_threshold=0.82,
    )

    choose_called = {"value": False}

    def _fake_choose_actions(*args, **kwargs):
        choose_called["value"] = True
        return [{"task": "ReqA", "actions": [{"component": "Wide", "action": "revise"}]}]

    def _fake_feedback_rounds(**kwargs):
        initial_actions = kwargs["initial_actions"]
        assert initial_actions[0]["task"] == "ReqA"
        assert initial_actions[0]["actions"][0]["component"] == "Wide"
        assert initial_actions[0]["actions"][0]["action"] == "split"
        assert initial_actions[0]["actions"][0]["action_origin"] == "metric_split"
        assert "Metric split trigger:" in initial_actions[0]["actions"][0]["rationale"]
        assert "3 served subrequirements" in initial_actions[0]["actions"][0]["rationale"]
        return architectures, initial_actions, {"stats": {"components_after": 1, "merge_group_count": 0, "split_group_count": 0}}

    summary = module.run_from_existing_architecture(
        args=args,
        choose_actions_fn=_fake_choose_actions,
        feedback_rounds_fn=_fake_feedback_rounds,
        gap_add_stage_fn=lambda **kwargs: (kwargs["architectures"], {"enabled": False, "accepted_count": 0, "parents": []}),
    )

    assert choose_called["value"] is False
    assert (output_dir / "component_metric_action_report.json").exists()
    assert summary["refined_parent_count"] == 1


def test_action_refine_run_directory_pattern_uses_timestamp_isolation():
    root = "/artifact/repo0"
    repo = "requests"
    ts = "20260608_190500"
    run_dir = f"{root}/tmp/action_refine_runs/{repo}/{ts}"
    out_dir = f"{run_dir}/agents_output"

    assert run_dir.endswith("/tmp/action_refine_runs/requests/20260608_190500")
    assert out_dir.endswith("/tmp/action_refine_runs/requests/20260608_190500/agents_output")
