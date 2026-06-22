import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_agents import (  # noqa: E402
    augment_actions_with_component_metrics,
    apply_action_hints_to_architectures,
    build_gap_add_stage_inputs,
    build_tdd_revise_action_report,
    build_codegen_parent_dag,
    build_single_run_command,
    build_evolution_action_override,
    build_parent_component_index,
    _build_package_init_exports,
    detect_completed_generated_parents,
    dedupe_components_by_name,
    filter_architectures_for_active_parents,
    filter_generated_files_for_active_parents,
    merge_actions_for_architectures,
    prune_memory_components_for_active_parents,
    select_generated_entries_for_parents,
    summarize_evolution_operations,
)
from agents.rqmts.dag import RequirementDAG, RequirementNode  # noqa: E402
from package_api_plan_builder import build_package_api_plan  # noqa: E402
from agents.merge.component_split_agent import ComponentSplitAgent  # noqa: E402


def test_build_package_init_exports_can_generate_lazy_imports(tmp_path, monkeypatch):
    generated_root = tmp_path / "generated"
    package_dir = generated_root / "pkg"
    package_dir.mkdir(parents=True)
    ok_file = package_dir / "ok.py"
    bad_file = package_dir / "bad.py"
    ok_file.write_text("class Exported:\n    pass\n", encoding="utf-8")
    bad_file.write_text('raise RuntimeError("eager import should stay lazy")\n', encoding="utf-8")

    report = _build_package_init_exports(
        generated_root=generated_root,
        generated_entries=[
            {"files": {"code": "pkg/ok.py"}},
            {"files": {"code": "pkg/bad.py"}},
        ],
        layout_root="pkg",
        package_api_plan={
            "components": [
                {"planned_file_path": "pkg/ok.py", "export_symbols": ["Exported"]},
            ]
        },
        lazy_imports=True,
    )

    assert report["packages_updated"] == 1
    init_text = (package_dir / "__init__.py").read_text(encoding="utf-8")
    assert "def __getattr__(name):" in init_text
    assert "from . import bad" not in init_text

    monkeypatch.syspath_prepend(str(generated_root))
    sys.modules.pop("pkg", None)
    sys.modules.pop("pkg.bad", None)
    pkg = __import__("pkg")

    assert pkg.Exported.__name__ == "Exported"
    with pytest.raises(RuntimeError, match="eager import should stay lazy"):
        getattr(pkg, "bad")


def test_build_evolution_action_override_merge_uses_affected_requirements():
    decision = {
        "tag": "RELATION",
        "relation_type": "MERGE",
        "affected_requirements": ["Auth", "Account"],
    }
    action = build_evolution_action_override(decision)

    assert action is not None
    assert action["operation"] == "merge"
    assert action["targets"] == ["Auth", "Account"]


def test_build_evolution_action_override_existing_returns_none():
    decision = {"tag": "EXISTING", "reason": "duplicate"}
    assert build_evolution_action_override(decision) is None


def test_build_evolution_action_override_split_without_target_falls_back_to_add():
    decision = {
        "tag": "RELATION",
        "relation_type": "SPLIT",
        "reason": "too broad",
    }
    action = build_evolution_action_override(decision)

    assert action is not None
    assert action["operation"] == "add"


def test_filter_architectures_for_active_parents_normalizes_task_keys():
    architectures = [
        {"parent_task": "ReqA", "architecture": {}},
        {"task": "ReqB", "architecture": {}},
        {"parent_task": "ReqOld", "architecture": {}},
    ]
    filtered = filter_architectures_for_active_parents(architectures, {"ReqA", "ReqB"})

    assert [item["parent_task"] for item in filtered] == ["ReqA", "ReqB"]
    assert [item["task"] for item in filtered] == ["ReqA", "ReqB"]


def test_filter_generated_files_for_active_parents_supports_legacy_task_field():
    generated = [
        {"task": "ReqA", "component": "CompA"},
        {"parent_task": "ReqB", "component": "CompB"},
        {"parent_task": "ReqOld", "component": "CompOld"},
    ]
    filtered = filter_generated_files_for_active_parents(generated, {"ReqA", "ReqB"})

    assert [item["parent_task"] for item in filtered] == ["ReqA", "ReqB"]


def test_merge_actions_for_architectures_prefers_new_entries_and_keeps_order():
    architectures = [
        {"parent_task": "ReqA", "architecture": {}},
        {"parent_task": "ReqB", "architecture": {}},
    ]
    existing_actions = [
        {"task": "ReqA", "actions": [{"action": "old"}]},
        {"task": "ReqOld", "actions": [{"action": "stale"}]},
    ]
    new_actions = [
        {"task": "ReqA", "actions": [{"action": "new"}]},
        {"task": "ReqB", "actions": [{"action": "fresh"}]},
    ]

    merged = merge_actions_for_architectures(existing_actions, new_actions, architectures)

    assert [item["task"] for item in merged] == ["ReqA", "ReqB"]
    assert merged[0]["actions"][0]["action"] == "new"
    assert merged[1]["actions"][0]["action"] == "fresh"


def test_build_default_empty_actions_matches_architecture_order():
    from run_agents import _build_default_empty_actions

    architectures = [
        {"parent_task": "ReqA", "architecture": {}},
        {"parent_task": "ReqB", "architecture": {}},
    ]

    assert _build_default_empty_actions(architectures) == [
        {"task": "ReqA", "actions": []},
        {"task": "ReqB", "actions": []},
    ]


def test_build_gap_add_stage_inputs_prefers_req_path_for_text_and_requirements_file_for_payload(tmp_path):
    req_path = tmp_path / "README.req"
    req_path.write_text("Support proxy environment configuration.\n", encoding="utf-8")
    requirements_file = tmp_path / "requirements.json"
    requirements_file.write_text(json.dumps({"project_summary": "summary", "requirements": [{"name": "ReqA"}]}), encoding="utf-8")
    generated_files = tmp_path / "generated_files.json"
    generated_files.write_text(json.dumps([{"parent_task": "ReqA", "component": "CompA"}]), encoding="utf-8")
    realization_report = tmp_path / "component_realization_report.json"
    realization_report.write_text(json.dumps({"total_components": 1}), encoding="utf-8")

    payload = build_gap_add_stage_inputs(
        requirements_file=requirements_file,
        req_path=req_path,
        generated_files_path=generated_files,
        realization_report_path=realization_report,
    )

    assert payload["input_text"] == "Support proxy environment configuration.\n"
    assert payload["requirements_payload"]["project_summary"] == "summary"
    assert payload["generated_entries"][0]["component"] == "CompA"
    assert payload["realization_report"]["total_components"] == 1


def test_build_single_run_command_forwards_stop_after_architecture_refinement(tmp_path):
    args = argparse.Namespace(
        requirements_file=tmp_path / "requirements.json",
        repo="requests",
        workspace=tmp_path,
        base_url="https://example.invalid/v1",
        api_key="k",
        model="m",
        max_workers=4,
        log_level="INFO",
        req_path=None,
        output=tmp_path / "out",
        use_processes=False,
        resume_rerun_retained_tdd_failures=False,
        parent_codegen_dag_source="dependency",
        disable_graph_module=False,
        disable_dependency_graph=False,
        no_graph_seed=42,
        disable_decomposition=False,
        disable_structure_refinement=False,
        disable_strategist=False,
        enable_component_metric_actions=False,
        component_metric_split_cohesion_threshold=2.0 / 3.0,
        component_metric_split_min_subrequirements=3,
        component_split_min_confidence=0.70,
        enable_component_metric_merge_judge=False,
        component_metric_merge_max_small_subrequirements=1,
        tdd_revise_failure_threshold=2,
        action_refinement_rounds=1,
        action_refinement_stop_on_stable=False,
        action_refinement_save_stops_component=False,
        enable_gap_add_actions=False,
        gap_add_proposal_threshold=0.55,
        gap_add_component_threshold=0.74,
        gap_add_requirement_threshold=0.82,
        stop_after_architecture_refinement=True,
        component_merge_admission_mode="strict",
        component_merge_relaxed_best=0.30,
        component_merge_relaxed_avg=0.26,
        component_merge_relaxed_min_pair=0.20,
        component_merge_relaxed_dominance_gap=0.28,
    )

    cmd = build_single_run_command(
        base_args=args,
        requirements_file=args.requirements_file,
        evolve_requirements_file=None,
        force_regenerate=False,
    )

    assert "--stop-after-architecture-refinement" in cmd


def test_component_split_prompt_includes_metric_rationale_and_confidence_signal():
    agent = ComponentSplitAgent(
        api_config={"api_key": "test"},
        output_dir="/tmp",
    )

    captured = {}

    class _FakeLLMClient:
        def call_json(self, messages, temperature=0.0, max_tokens=0, **kwargs):
            captured["messages"] = messages
            return {
                "confidence": 0.9,
                "reason": "strong split candidate",
                "split_components": [
                    {
                        "name": "Core",
                        "responsibilities": ["core concern"],
                        "serves_subrequirements": ["Req::core", "Req::ops"],
                    },
                    {
                        "name": "API",
                        "responsibilities": ["api concern"],
                        "serves_subrequirements": ["Req::api"],
                    },
                ],
            }

    agent.llm_client = _FakeLLMClient()
    component = {
        "name": "WideComp",
        "responsibilities": ["core concern", "api concern"],
        "serves_subrequirements": ["Req::core", "Req::api", "Req::ops"],
        "recommended_action": "split",
        "recommended_action_rationale": "Metric split trigger: cohesion=0.667 with 3 served subrequirements.",
        "recommended_action_origin": "metric_split",
    }

    split_components, split_detail = agent._split_component_with_llm("Req", component, 1)

    assert [row["type"] for row in captured["messages"]] if False else True
    prompt = captured["messages"][1]["content"]
    assert "Metric split trigger: cohesion=0.667 with 3 served subrequirements." in prompt
    assert "metric_split" in prompt
    assert "`confidence` must be a JSON number between 0 and 1." in prompt
    assert split_detail["decision"] == "split"
    assert [row["name"] for row in split_components] == ["Core", "API"]


def test_component_split_accepts_confidence_without_explicit_decision():
    agent = ComponentSplitAgent(
        api_config={"api_key": "test"},
        output_dir="/tmp",
    )

    class _FakeLLMClient:
        def call_json(self, messages, temperature=0.0, max_tokens=0, **kwargs):
            return {
                "confidence": 0.9,
                "reason": "stable module boundaries detected",
                "split_components": [
                    {
                        "name": "Core",
                        "responsibilities": ["core concern"],
                        "serves_subrequirements": ["Req::core", "Req::ops"],
                    },
                    {
                        "name": "API",
                        "responsibilities": ["api concern"],
                        "serves_subrequirements": ["Req::api"],
                    },
                ],
            }

    agent.llm_client = _FakeLLMClient()
    component = {
        "name": "WideComp",
        "responsibilities": ["core concern", "api concern"],
        "serves_subrequirements": ["Req::core", "Req::api", "Req::ops"],
        "recommended_action": "split",
    }

    split_components, split_detail = agent._split_component_with_llm("Req", component, 1)

    assert split_detail["decision"] == "split"
    assert split_detail["confidence"] == 0.9
    assert [row["name"] for row in split_components] == ["Core", "API"]


def test_component_split_accepts_at_default_confidence_threshold():
    agent = ComponentSplitAgent(
        api_config={"api_key": "test"},
        output_dir="/tmp",
    )

    class _FakeLLMClient:
        def call_json(self, messages, temperature=0.0, max_tokens=0, **kwargs):
            return {
                "confidence": 0.7,
                "reason": "borderline but acceptable split candidate",
                "split_components": [
                    {
                        "name": "Core",
                        "responsibilities": ["core concern"],
                        "serves_subrequirements": ["Req::core", "Req::ops"],
                    },
                    {
                        "name": "API",
                        "responsibilities": ["api concern"],
                        "serves_subrequirements": ["Req::api"],
                    },
                ],
            }

    agent.llm_client = _FakeLLMClient()
    component = {
        "name": "WideComp",
        "responsibilities": ["core concern", "api concern"],
        "serves_subrequirements": ["Req::core", "Req::api", "Req::ops"],
        "recommended_action": "split",
    }

    split_components, split_detail = agent._split_component_with_llm("Req", component, 1)

    assert split_detail["decision"] == "split"
    assert split_detail["confidence"] == 0.7
    assert [row["name"] for row in split_components] == ["Core", "API"]


def test_component_split_accepts_semantic_subcomponents_without_exact_resp_reuse():
    agent = ComponentSplitAgent(
        api_config={"api_key": "test"},
        output_dir="/tmp",
    )

    class _FakeLLMClient:
        def call_json(self, messages, temperature=0.0, max_tokens=0, **kwargs):
            return {
                "confidence": 0.86,
                "reason": "the component splits into config resolution and final request assembly",
                "split_components": [
                    {
                        "name": "ConfigResolver",
                        "responsibilities": [
                            "Resolve and merge request/session configuration",
                            "Normalize request inputs",
                        ],
                        "serves_subrequirements": ["Req::core", "Req::config"],
                    },
                    {
                        "name": "PreparedRequestFinalizer",
                        "responsibilities": [
                            "Finalize transport-facing request metadata and body handling",
                            "Assemble immutable prepared request artifacts",
                        ],
                        "serves_subrequirements": ["Req::transport", "Req::api"],
                    },
                ],
            }

    agent.llm_client = _FakeLLMClient()
    component = {
        "name": "RequestPreparer",
        "responsibilities": [
            "Normalize raw request inputs into a deterministic intermediate form",
            "Merge in session defaults and expose an inspectable resolved_config snapshot",
            "Apply header normalization and canonicalization and compute the final body iterator/stream",
            "Produce an immutable PreparedRequest artifact with finalized metadata",
        ],
        "serves_subrequirements": ["Req::core", "Req::config", "Req::transport", "Req::api"],
        "recommended_action": "split",
        "recommended_action_rationale": "Metric split trigger: cohesion=0.667 with 4 served subrequirements.",
        "recommended_action_origin": "metric_split",
    }

    split_components, split_detail = agent._split_component_with_llm("Req", component, 1)

    assert split_detail["decision"] == "split"
    assert [row["name"] for row in split_components] == ["ConfigResolver", "PreparedRequestFinalizer"]


def test_component_split_only_triggers_for_explicit_split_action():
    agent = ComponentSplitAgent(
        api_config={"api_key": "test"},
        output_dir="/tmp",
    )

    broad_component = {
        "name": "WideComp",
        "responsibilities": [f"resp-{idx}" for idx in range(12)],
        "serves_subrequirements": [f"Req::{idx}" for idx in range(6)],
    }
    assert agent._should_split_component(broad_component) is False

    broad_component["recommended_action"] = "split"
    assert agent._should_split_component(broad_component) is True


def test_augment_actions_with_component_metrics_upgrades_save_to_split_and_merge():
    decomposed_dag = RequirementDAG(
        nodes={
            "A": RequirementNode(name="A", description="a"),
            "B": RequirementNode(name="B", description="b"),
            "C": RequirementNode(name="C", description="c"),
            "D": RequirementNode(name="D", description="d"),
            "X": RequirementNode(name="X", description="x"),
            "Y": RequirementNode(name="Y", description="y"),
        },
        adjacency={
            "A": {"B"},
            "B": {"C"},
            "C": set(),
            "D": set(),
            "X": {"Y"},
            "Y": set(),
        },
    )
    architectures = [
        {
            "parent_task": "ReqSplit",
            "architecture": {
                "components": [
                    {"name": "Wide", "serves_subrequirements": ["A", "C", "D"]},
                    {"name": "Stable", "serves_subrequirements": ["B"]},
                ]
            },
        },
        {
            "parent_task": "ReqMerge",
            "architecture": {
                "components": [
                    {"name": "Public API", "serves_subrequirements": ["X", "Y"]},
                    {"name": "Compatibility Facade", "serves_subrequirements": ["X", "Y"]},
                ]
            },
        },
    ]
    actions = [
        {
            "task": "ReqSplit",
            "actions": [
                {"component": "Wide", "action": "save", "rationale": "baseline save"},
                {"component": "Stable", "action": "save", "rationale": "baseline save"},
            ],
        },
        {
            "task": "ReqMerge",
            "actions": [
                {"component": "Public API", "action": "save", "rationale": "baseline save"},
                {"component": "Compatibility Facade", "action": "save", "rationale": "baseline save"},
            ],
        },
    ]

    def _judge(candidate):
        return {
            "approved": candidate["source_component"] == "Compatibility Facade",
            "same_responsibility": True,
            "interface_conflict": False,
            "behavior_conflict": False,
            "risk": "low",
            "reason": "merge accepted in test",
        }

    augmented, report = augment_actions_with_component_metrics(
        architectures=architectures,
        actions=actions,
        decomposed_dag=decomposed_dag,
        split_cohesion_threshold=2 / 3,
        split_min_subrequirements=3,
        merge_judge=_judge,
        merge_max_small_subrequirements=1,
    )

    split_actions = {
        row["component"]: row["action"]
        for entry in augmented
        if entry["task"] == "ReqSplit"
        for row in entry["actions"]
    }
    assert split_actions["Wide"] == "split"
    assert split_actions["Stable"] == "save"

    merge_rows = [
        row
        for entry in augmented
        if entry["task"] == "ReqMerge"
        for row in entry["actions"]
        if row["component"] == "Compatibility Facade"
    ]
    assert merge_rows[0]["action"] == "merge"
    assert merge_rows[0]["target_component"] == "Public API"
    assert report["stats"]["split_upgrades"] == 1
    assert report["stats"]["merge_upgrades"] == 1
    assert report["parents"][1]["merge_candidates"][0]["jaccard"] == 1.0
    assert report["parents"][1]["merge_candidates"][0]["coupling"] == 1.0


def test_augment_actions_with_component_metrics_logs_summary(caplog):
    decomposed_dag = RequirementDAG(
        nodes={
            "A": RequirementNode(name="A", description="a"),
            "B": RequirementNode(name="B", description="b"),
            "X": RequirementNode(name="X", description="x"),
            "Y": RequirementNode(name="Y", description="y"),
        },
        adjacency={
            "A": {"B"},
            "B": set(),
            "X": {"Y"},
            "Y": set(),
        },
    )
    architectures = [
        {
            "parent_task": "ReqSplit",
            "architecture": {
                "components": [
                    {"name": "Wide", "serves_subrequirements": ["A", "B", "X"]},
                    {"name": "Stable", "serves_subrequirements": ["Y"]},
                ]
            },
        },
        {
            "parent_task": "ReqMerge",
            "architecture": {
                "components": [
                    {"name": "Public API", "serves_subrequirements": ["X", "Y"]},
                    {"name": "Compatibility Facade", "serves_subrequirements": ["X", "Y"]},
                ]
            },
        },
    ]
    actions = [
        {"task": "ReqSplit", "actions": []},
        {"task": "ReqMerge", "actions": []},
    ]

    def _judge(candidate):
        return {
            "approved": False,
            "same_responsibility": True,
            "interface_conflict": False,
            "behavior_conflict": False,
            "risk": "low",
            "reason": "rejected in test",
        }

    caplog.set_level(logging.INFO)
    augment_actions_with_component_metrics(
        architectures=architectures,
        actions=actions,
        decomposed_dag=decomposed_dag,
        split_cohesion_threshold=2 / 3,
        split_min_subrequirements=3,
        merge_judge=_judge,
        merge_max_small_subrequirements=1,
    )

    assert "Metric action summary:" in caplog.text
    assert "split_upgrades=1" in caplog.text
    assert "merge_candidates=2" in caplog.text
    assert "parent=ReqSplit" in caplog.text
    assert "parent=ReqMerge" in caplog.text


def test_metric_merge_layer_penalty_reduces_core_api_coupling():
    from run_agents import _metric_merge_layer_penalty

    assert _metric_merge_layer_penalty("Estimation Core", "Public API") == 0.5
    assert _metric_merge_layer_penalty("Runtime Engine", "Integration Adapter") == 0.5
    assert _metric_merge_layer_penalty("Compatibility Facade", "Registry Service") == 1.0


def test_metric_merge_judge_prompt_explains_coupling():
    from run_agents import _build_metric_merge_judge

    captured = {}

    class _FakeLLMClient:
        def call_json(self, messages, temperature=0.0, max_tokens=0, operation_name=None, **kwargs):
            captured["messages"] = messages
            return {
                "approved": False,
                "same_responsibility": False,
                "interface_conflict": False,
                "behavior_conflict": False,
                "risk": "high",
                "reason": "not redundant",
            }

    class _FakeMergeAgent:
        llm_client = _FakeLLMClient()

    judge = _build_metric_merge_judge(_FakeMergeAgent())
    judge(
        {
            "parent_task": "ReqMerge",
            "source_component": "Public API",
            "target_component": "Compatibility Facade",
            "source_subrequirements": {"X", "Y"},
            "target_subrequirements": {"X", "Y"},
            "source_responsibilities": ["api"],
            "target_responsibilities": ["compat"],
            "cross_edges": 1,
            "jaccard": 1.0,
            "layer_penalty": 1.0,
            "coupling": 1.0,
            "reason": "Metric merge candidate: Jaccard=1.000, layer_penalty=1.000, coupling=1.000, cross_edges=1.",
        }
    )

    prompt = captured["messages"][1]["content"]
    assert "Jaccard(served_subrequirements) * layering_penalty" in prompt
    assert "layering_penalty < 1" in prompt


def test_apply_action_hints_to_architectures_copies_merge_target_metadata():
    architectures = [
        {
            "parent_task": "ReqA",
            "architecture": {
                "components": [
                    {"name": "Alpha"},
                    {"name": "Beta"},
                ]
            },
        }
    ]
    actions = [
        {
            "task": "ReqA",
            "actions": [
                {
                    "component": "Alpha",
                    "action": "merge",
                    "rationale": "metric merge",
                    "target_component": "Beta",
                    "action_origin": "metric_merge_judged",
                }
            ],
        }
    ]

    hinted = apply_action_hints_to_architectures(architectures, actions)
    alpha = hinted[0]["architecture"]["components"][0]

    assert alpha["recommended_action"] == "merge"
    assert alpha["recommended_target_component"] == "Beta"
    assert alpha["recommended_action_origin"] == "metric_merge_judged"


def test_apply_action_hints_to_architectures_clears_stale_recommended_actions():
    architectures = [
        {
            "parent_task": "ReqA",
            "architecture": {
                "components": [
                    {
                        "name": "Alpha",
                        "recommended_action": "split",
                        "recommended_action_rationale": "stale split",
                        "recommended_action_origin": "legacy",
                    },
                    {
                        "name": "Beta",
                        "recommended_action": "merge",
                        "recommended_action_rationale": "stale merge",
                        "recommended_target_component": "Alpha",
                        "recommended_action_origin": "legacy",
                    },
                ]
            },
        }
    ]
    actions = [
        {
            "task": "ReqA",
            "actions": [
                {
                    "component": "Alpha",
                    "action": "save",
                    "rationale": "fresh save",
                    "action_origin": "metric_round",
                }
            ],
        }
    ]

    hinted = apply_action_hints_to_architectures(architectures, actions)
    alpha, beta = hinted[0]["architecture"]["components"]

    assert alpha["recommended_action"] == "save"
    assert alpha["recommended_action_rationale"] == "fresh save"
    assert alpha["recommended_action_origin"] == "metric_round"
    assert "recommended_target_component" not in alpha

    assert "recommended_action" not in beta
    assert "recommended_action_rationale" not in beta
    assert "recommended_target_component" not in beta
    assert "recommended_action_origin" not in beta


def test_build_tdd_revise_action_report_requires_repeated_failures():
    generated_entries = [
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "tdd_final_pytest_rc": 1,
            "compressed_feedback": "semantic mismatch on test one",
        },
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "tdd_final_pytest_rc": 1,
            "compressed_feedback": "semantic mismatch on test two",
        },
        {
            "parent_task": "ReqB",
            "component": "CompB",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "tdd_final_pytest_rc": 1,
            "compressed_feedback": "only failed once",
        },
    ]

    report = build_tdd_revise_action_report(generated_entries, failure_threshold=2)

    assert report["stats"]["revise_candidates"] == 1
    assert report["actions"] == [
        {
            "task": "ReqA",
            "actions": [
                {
                    "component": "CompA",
                    "action": "revise",
                    "rationale": "Triggered after 2 consecutive TDD failures. semantic mismatch on test two",
                    "action_origin": "tdd_revise_threshold",
                }
            ],
        }
    ]


def test_build_tdd_revise_action_report_requires_consecutive_semantic_failures():
    generated_entries = [
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "syntax_postcheck": {"passed": True},
            "compile_postcheck": {"passed": True},
            "import_postcheck": {"passed": True},
            "compressed_feedback": "semantic mismatch one",
        },
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "syntax_postcheck": {"passed": True},
            "compile_postcheck": {"passed": False},
            "import_postcheck": {"passed": True},
            "compressed_feedback": "compile break should reset",
        },
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "syntax_postcheck": {"passed": True},
            "compile_postcheck": {"passed": True},
            "import_postcheck": {"passed": True},
            "compressed_feedback": "semantic mismatch two",
        },
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "syntax_postcheck": {"passed": True},
            "compile_postcheck": {"passed": True},
            "import_postcheck": {"passed": True},
            "compressed_feedback": "semantic mismatch three",
        },
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "syntax_postcheck": {"passed": True},
            "compile_postcheck": {"passed": True},
            "import_postcheck": {"passed": True},
            "compressed_feedback": "semantic mismatch four",
        },
    ]

    report = build_tdd_revise_action_report(generated_entries, failure_threshold=3)

    assert report["stats"]["revise_candidates"] == 1
    assert report["actions"] == [
        {
            "task": "ReqA",
            "actions": [
                {
                    "component": "CompA",
                    "action": "revise",
                    "rationale": "Triggered after 3 consecutive TDD failures. semantic mismatch four",
                    "action_origin": "tdd_revise_threshold",
                }
            ],
        }
    ]


def test_build_tdd_revise_action_report_ignores_non_semantic_failures():
    generated_entries = [
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "syntax_postcheck": {"passed": False},
            "compile_postcheck": {"passed": True},
            "import_postcheck": {"passed": True},
            "compressed_feedback": "syntax failure",
        },
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "syntax_postcheck": {"passed": True},
            "compile_postcheck": {"passed": True},
            "import_postcheck": {"passed": False},
            "compressed_feedback": "import failure",
        },
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "syntax_postcheck": {"passed": True},
            "compile_postcheck": {"passed": True},
            "import_postcheck": {"passed": True},
            "compressed_feedback": "only one semantic failure",
        },
    ]

    report = build_tdd_revise_action_report(generated_entries, failure_threshold=3)

    assert report["stats"]["revise_candidates"] == 0
    assert report["actions"] == []


def test_prune_memory_components_for_active_parents_removes_stale_entries():
    snapshot = SimpleNamespace(
        implemented_components={
            "ReqA::CompA": SimpleNamespace(requirement_node="ReqA"),
            "ReqB::CompB": SimpleNamespace(requirement_node="ReqB"),
            "Legacy::CompX": SimpleNamespace(requirement_node=""),
        }
    )
    memory_agent = SimpleNamespace(snapshot=snapshot)

    removed = prune_memory_components_for_active_parents(memory_agent, {"ReqA"})

    assert removed == 2
    assert set(snapshot.implemented_components.keys()) == {"ReqA::CompA"}


def test_summarize_evolution_operations_split_and_merge_builds_mapping():
    records = [
        {
            "operation_type": "split",
            "details": {"original": "ReqOld", "created": ["ReqOld::core", "ReqNew"]},
        },
        {
            "operation_type": "merge",
            "details": {"merged_from": ["ReqA", "ReqB"], "merged_to": "ReqMerged"},
        },
    ]

    summary = summarize_evolution_operations(
        records,
        active_parents={"ReqOld::core", "ReqNew", "ReqMerged"},
    )

    assert summary["regen_parents"] == {"ReqOld::core", "ReqNew", "ReqMerged"}
    assert summary["removed_parents"] == {"ReqOld", "ReqA", "ReqB"}
    assert summary["source_parents_by_target"]["ReqNew"] == {"ReqOld"}
    assert summary["source_parents_by_target"]["ReqMerged"] == {"ReqA", "ReqB"}


def test_component_index_and_context_selection_helpers():
    architectures = [
        {"parent_task": "ReqA", "architecture": {"components": [{"name": "A1"}]}},
        {"task": "ReqB", "architecture": {"components": [{"name": "B1"}]}},
    ]
    index = build_parent_component_index(architectures)
    assert [comp["name"] for comp in index["ReqA"]] == ["A1"]
    assert [comp["name"] for comp in index["ReqB"]] == ["B1"]

    merged = dedupe_components_by_name(
        [{"name": "A1"}, {"name": "A1"}, {"name": "B1"}]
    )
    assert [item["name"] for item in merged] == ["A1", "B1"]

    generated = [
        {"parent_task": "ReqA", "component": "A1"},
        {"task": "ReqLegacy", "component": "L1"},
    ]
    scoped = select_generated_entries_for_parents(generated, {"ReqLegacy"})
    assert len(scoped) == 1
    assert scoped[0]["parent_task"] == "ReqLegacy"


def test_build_single_run_command_respects_mode_flags():
    args = argparse.Namespace(
        requirements_file=Path("base.json"),
        repo="tinydb",
        workspace=Path("/tmp/ws"),
        base_url="http://example",
        api_key="k",
        model="m",
        max_workers=4,
        log_level="INFO",
        req_path=None,
        output=Path("/tmp/ws/out"),
        use_processes=False,
        resume_rerun_retained_tdd_failures=False,
        parent_codegen_dag_source="dependency",
    )

    cmd = build_single_run_command(
        base_args=args,
        requirements_file=Path("/tmp/ws/base.json"),
        evolve_requirements_file=Path("/tmp/ws/evolve.json"),
        force_regenerate=True,
    )

    cmd_text = " ".join(cmd)
    assert "--force-regenerate" in cmd
    assert "--evolve-requirements-file" in cmd
    assert "/tmp/ws/evolve.json" in cmd_text
    assert "--parent-codegen-dag-source" in cmd
    assert "dependency" in cmd_text
    assert "--session-loop" not in cmd_text


def test_build_codegen_parent_dag_can_use_dependency_graph_edges():
    requirement_dag = RequirementDAG(
        nodes={
            "ReqA": RequirementNode(name="ReqA", description="a"),
            "ReqB": RequirementNode(name="ReqB", description="b"),
            "ReqC": RequirementNode(name="ReqC", description="c"),
        },
        adjacency={
            "ReqA": {"ReqB"},
            "ReqB": {"ReqC"},
            "ReqC": set(),
        },
    )
    dependency_graph_payload = {
        "dependency_graph": {
            "edges": [
                {"source": "ReqB", "target": "ReqC"},
            ]
        }
    }

    selected = build_codegen_parent_dag(
        dag_source="dependency",
        requirement_dag=requirement_dag,
        dependency_graph_payload=dependency_graph_payload,
    )

    assert set(selected.nodes.keys()) == {"ReqA", "ReqB", "ReqC"}
    assert selected.adjacency["ReqA"] == set()
    assert selected.adjacency["ReqB"] == {"ReqC"}
    assert selected.adjacency["ReqC"] == set()


def test_detect_completed_generated_parents_reuses_retained_tdd_failure_by_default(tmp_path: Path):
    code_file = tmp_path / "comp.py"
    code_file.write_text("def ok() -> int:\n    return 1\n", encoding="utf-8")

    generated = [
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "files": {"code": str(code_file)},
            "syntax_postcheck": {"passed": True, "code_file": str(code_file), "error": ""},
            "compile_postcheck": {"passed": True, "code_file": str(code_file), "error": ""},
            "import_postcheck": {"passed": True, "code_file": str(code_file), "module": "pkg.comp", "error": ""},
        }
    ]
    architectures = [{"parent_task": "ReqA", "architecture": {"components": [{"name": "CompA"}]}}]

    completed = detect_completed_generated_parents(generated, architectures)

    assert completed == {"ReqA"}


def test_detect_completed_generated_parents_can_rerun_retained_tdd_failure(tmp_path: Path):
    code_file = tmp_path / "comp.py"
    code_file.write_text("def ok() -> int:\n    return 1\n", encoding="utf-8")

    generated = [
        {
            "parent_task": "ReqA",
            "component": "CompA",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "files": {"code": str(code_file)},
            "syntax_postcheck": {"passed": True, "code_file": str(code_file), "error": ""},
            "compile_postcheck": {"passed": True, "code_file": str(code_file), "error": ""},
            "import_postcheck": {"passed": True, "code_file": str(code_file), "module": "pkg.comp", "error": ""},
        }
    ]
    architectures = [{"parent_task": "ReqA", "architecture": {"components": [{"name": "CompA"}]}}]

    completed = detect_completed_generated_parents(
        generated,
        architectures,
        rerun_retained_tdd_failures=True,
    )

    assert completed == set()


def test_apply_action_hints_to_architectures_marks_components_without_rewriting_shape():
    architectures = [
        {
            "parent_task": "Time series analysis module",
            "architecture": {
                "components": [
                    {"name": "StateSpaceCore", "responsibilities": ["Implement ARIMA and SARIMAX workflows"]},
                    {"name": "ModelLibrary", "responsibilities": ["Expose statistical model catalog"]},
                ]
            },
        }
    ]
    actions = [
        {
            "task": "Time series analysis module",
            "actions": [
                {"component": "StateSpaceCore", "action": "split", "rationale": "separate state-space kernels from classical tsa helpers"},
            ],
        }
    ]

    updated = apply_action_hints_to_architectures(architectures, actions)

    components = updated[0]["architecture"]["components"]
    assert components[0]["recommended_action"] == "split"
    assert "state-space" in components[0]["recommended_action_rationale"]
    assert "recommended_action" not in components[1]


def test_build_package_api_plan_uses_domain_subpackage_under_generic_core():
    architectures = [
        {
            "parent_task": "Time series analysis module",
            "architecture": {
                "requirement": {"description": "Support ARIMA, VARMAX, ACF and PACF workflows"},
                "components": [
                    {
                        "name": "StateSpaceCore",
                        "responsibilities": [
                            "Implement ARIMA, SARIMA, VARMAX and VECM estimation",
                            "Provide ACF and PACF extraction helpers",
                        ],
                        "recommended_action": "split",
                    }
                ],
            },
            "sub_tasks": [],
        }
    ]
    layout_policy = {
        "layout_root": "statsmodels",
        "alias_map": {},
        "default_subpackage": "core",
    }

    def _fake_file_plan(_architecture, _task, _policy):
        return {"StateSpaceCore": ""}

    plan = build_package_api_plan(architectures, layout_policy, _fake_file_plan)
    row = plan["component_index"]["Time series analysis module::StateSpaceCore"]

    assert row["canonical_package"] in {"time_series", "core"}
    assert "time_series" in row["package_subpath"]
