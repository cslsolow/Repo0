import argparse
import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.rqmts.dag import RequirementDAG, RequirementNode  # noqa: E402
from agents.cognitive.architect import ArchitectAgent  # noqa: E402
from agents.cognitive.generator import GenerationAgent  # noqa: E402
from agents import ComponentMergeAgent  # noqa: E402
from run_agents import (  # noqa: E402
    apply_action_guided_structure_refinement,
    build_codegen_parent_dag,
    build_codegen_parent_order,
    build_no_graph_parent_groups,
    build_no_graph_plan,
    build_single_run_command,
    process_architecture_task,
    run_action_feedback_rounds,
)


def test_build_single_run_command_includes_graph_module_controls():
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
        parent_codegen_dag_source="requirement",
        disable_graph_module=True,
        disable_dependency_graph=True,
        no_graph_seed=17,
        disable_decomposition=True,
        disable_structure_refinement=True,
        disable_strategist=True,
    )

    cmd = build_single_run_command(
        base_args=args,
        requirements_file=Path("/tmp/ws/base.json"),
        evolve_requirements_file=Path("/tmp/ws/evolve.json"),
        force_regenerate=True,
    )

    cmd_text = " ".join(cmd)
    assert "--parent-codegen-dag-source" in cmd
    assert "requirement" in cmd_text
    assert "--disable-graph-module" in cmd
    assert "--disable-dependency-graph" in cmd
    assert "--no-graph-seed" in cmd
    assert "17" in cmd_text
    assert "--disable-decomposition" in cmd
    assert "--disable-structure-refinement" in cmd
    assert "--disable-strategist" in cmd


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


def test_build_codegen_parent_dag_can_disable_graph_edges_entirely():
    requirement_dag = RequirementDAG(
        nodes={
            "ReqA": RequirementNode(name="ReqA", description="a"),
            "ReqB": RequirementNode(name="ReqB", description="b"),
        },
        adjacency={
            "ReqA": {"ReqB"},
            "ReqB": set(),
        },
    )

    selected = build_codegen_parent_dag(
        dag_source="none",
        requirement_dag=requirement_dag,
        dependency_graph_payload=None,
    )

    assert set(selected.nodes.keys()) == {"ReqA", "ReqB"}
    assert selected.adjacency["ReqA"] == set()
    assert selected.adjacency["ReqB"] == set()


def test_disable_dependency_graph_switches_dependency_mode_to_requirement_mode():
    raw_source = "dependency"
    graph_module_disabled = False
    dependency_graph_disabled = True

    if graph_module_disabled:
        effective = "none"
    elif dependency_graph_disabled and raw_source == "dependency":
        effective = "requirement"
    else:
        effective = raw_source

    assert effective == "requirement"


def test_action_guided_structure_refinement_runs_merge_then_split_when_hinted():
    architectures = [
        {
            "parent_task": "ReqA",
            "architecture": {
                "components": [
                    {"name": "Alpha", "recommended_action": "merge"},
                    {"name": "Beta"},
                ]
            },
        }
    ]
    calls = []

    class _MergeAgent:
        def __init__(self):
            self.call_count = 0

        def merge_architecture_components(self, parent_task, architecture, **kwargs):
            self.call_count += 1
            calls.append(
                (
                    "merge",
                    parent_task,
                    len(architecture.get("components", [])),
                    bool(kwargs.get("require_split_origin")),
                )
            )
            if self.call_count == 1:
                merged = dict(architecture)
                merged["components"] = [{"name": "MergedAlphaBeta"}]
                return merged, {"stats": {"merged_component_count": 1}, "merge_groups": [{"source_ids": ["C1", "C2"]}]}
            return architecture, {"stats": {"merged_component_count": 0}, "merge_groups": []}

    class _SplitAgent:
        def split_architecture_components(self, parent_task, architecture):
            calls.append(("split", parent_task, len(architecture.get("components", []))))
            split = dict(architecture)
            split["components"] = [
                {"name": "MergedAlpha", "split_from_name": "MergedAlphaBeta"},
                {"name": "MergedBeta", "split_from_name": "MergedAlphaBeta"},
            ]
            return split, {"stats": {"split_group_count": 1}}

    refined, report = apply_action_guided_structure_refinement(
        architectures=architectures,
        component_merge_agent=_MergeAgent(),
        component_split_agent=_SplitAgent(),
    )

    assert calls == [
        ("merge", "ReqA", 2, False),
        ("split", "ReqA", 1),
        ("merge", "ReqA", 2, True),
    ]
    assert [c["name"] for c in refined[0]["architecture"]["components"]] == ["MergedAlpha", "MergedBeta"]
    assert report["stats"]["merge_group_count"] == 1
    assert report["stats"]["split_group_count"] == 1


def test_action_guided_structure_refinement_skips_merge_without_merge_hint():
    architectures = [
        {
            "parent_task": "ReqA",
            "architecture": {
                "components": [
                    {"name": "Alpha", "recommended_action": "save"},
                    {"name": "Beta"},
                ]
            },
        }
    ]
    calls = []

    class _MergeAgent:
        def merge_architecture_components(self, parent_task, architecture, **kwargs):
            calls.append(("merge", parent_task, len(architecture.get("components", [])), kwargs))
            return architecture, {"stats": {"merged_component_count": 0}, "merge_groups": []}

    class _SplitAgent:
        def split_architecture_components(self, parent_task, architecture):
            calls.append(("split", parent_task, len(architecture.get("components", []))))
            return architecture, {"stats": {"split_group_count": 0}}

    refined, report = apply_action_guided_structure_refinement(
        architectures=architectures,
        component_merge_agent=_MergeAgent(),
        component_split_agent=_SplitAgent(),
    )

    assert calls == [("split", "ReqA", 2)]
    assert [c["name"] for c in refined[0]["architecture"]["components"]] == ["Alpha", "Beta"]
    assert report["stats"]["merge_group_count"] == 0


def test_action_guided_structure_refinement_runs_post_split_merge_without_pre_merge_hint():
    architectures = [
        {
            "parent_task": "ReqA",
            "architecture": {
                "components": [
                    {"name": "Alpha", "recommended_action": "save"},
                    {"name": "Beta"},
                ]
            },
        }
    ]
    calls = []

    class _MergeAgent:
        def merge_architecture_components(self, parent_task, architecture, **kwargs):
            calls.append(
                (
                    "merge",
                    parent_task,
                    [item["name"] for item in architecture.get("components", [])],
                    bool(kwargs.get("require_split_origin")),
                    bool(kwargs.get("include_rule_candidates")),
                )
            )
            return architecture, {"stats": {"merged_component_count": 0}, "merge_groups": []}

    class _SplitAgent:
        def split_architecture_components(self, parent_task, architecture):
            calls.append(("split", parent_task, [item["name"] for item in architecture.get("components", [])]))
            split = dict(architecture)
            split["components"] = [
                {"name": "AlphaCore", "split_from_name": "Alpha"},
                {"name": "AlphaIO", "split_from_name": "Alpha"},
                {"name": "Beta"},
            ]
            return split, {"stats": {"split_group_count": 1}}

    refined, report = apply_action_guided_structure_refinement(
        architectures=architectures,
        component_merge_agent=_MergeAgent(),
        component_split_agent=_SplitAgent(),
    )

    assert calls == [
        ("split", "ReqA", ["Alpha", "Beta"]),
        ("merge", "ReqA", ["AlphaCore", "AlphaIO", "Beta"], True, True),
    ]
    assert [c["name"] for c in refined[0]["architecture"]["components"]] == ["AlphaCore", "AlphaIO", "Beta"]
    assert report["stats"]["merge_group_count"] == 0
    assert report["stats"]["split_group_count"] == 1


def test_run_action_feedback_rounds_uses_refined_components_in_next_round(tmp_path):
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
    initial_actions = [
        {
            "task": "ReqA",
            "actions": [
                {"component": "Alpha", "action": "split", "rationale": "too broad"},
                {"component": "Beta", "action": "save", "rationale": "cohesive"},
            ],
        }
    ]
    seen_rounds = []

    class _SplitAgent:
        def split_architecture_components(self, parent_task, architecture):
            names = [item["name"] for item in architecture.get("components", [])]
            split_requested = any(
                item.get("name") == "Alpha" and item.get("recommended_action") == "split"
                for item in architecture.get("components", [])
            )
            seen_rounds.append((parent_task, tuple(names), split_requested))
            if split_requested:
                return {
                    "components": [
                        {"name": "AlphaCore"},
                        {"name": "AlphaIO"},
                        {"name": "Beta"},
                    ]
                }, {"stats": {"split_group_count": 1}}
            return architecture, {"stats": {"split_group_count": 0}}

    def _action_selector(round_architectures, round_idx):
        assert round_idx == 2
        names = [
            item["name"]
            for item in round_architectures[0]["architecture"]["components"]
        ]
        assert names == ["AlphaCore", "AlphaIO", "Beta"]
        return [
            {
                "task": "ReqA",
                "actions": [
                    {"component": name, "action": "save", "rationale": "stable"}
                    for name in names
                ],
            }
        ]

    refined, final_actions, report = run_action_feedback_rounds(
        architectures=architectures,
        initial_actions=initial_actions,
        component_merge_agent=None,
        component_split_agent=_SplitAgent(),
        rounds=2,
        api_config={},
        output_dir=tmp_path,
        max_workers=1,
        action_selector=_action_selector,
    )

    assert [c["name"] for c in refined[0]["architecture"]["components"]] == ["AlphaCore", "AlphaIO", "Beta"]
    assert [row["component"] for row in final_actions[0]["actions"]] == ["AlphaCore", "AlphaIO", "Beta"]
    assert report["stats"]["rounds_completed"] == 2
    assert report["stats"]["split_group_count"] == 1
    assert (tmp_path / "actions_round_1.json").exists()
    assert (tmp_path / "action_refinement_round_2.json").exists()
    assert seen_rounds == [
        ("ReqA", ("Alpha", "Beta"), True),
        ("ReqA", ("AlphaCore", "AlphaIO", "Beta"), False),
    ]


def test_run_action_feedback_rounds_logs_proposed_and_accepted_actions(tmp_path, caplog):
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
    initial_actions = [
        {
            "task": "ReqA",
            "actions": [
                {
                    "component": "Alpha",
                    "action": "split",
                    "rationale": "too broad",
                    "action_origin": "metric_split",
                },
                {
                    "component": "Beta",
                    "action": "save",
                    "rationale": "stable",
                },
            ],
        }
    ]

    class _SplitAgent:
        def split_architecture_components(self, parent_task, architecture):
            return {
                "components": [
                    {"name": "AlphaCore"},
                    {"name": "AlphaAPI"},
                    {"name": "Beta"},
                ]
            }, {
                "split_groups": [
                    {
                        "component_name": "Alpha",
                        "split_into": ["AlphaCore", "AlphaAPI"],
                        "confidence": 0.88,
                        "reason": "clear separation",
                    }
                ],
                "stats": {"split_group_count": 1},
            }

    caplog.set_level(logging.INFO)
    run_action_feedback_rounds(
        architectures=architectures,
        initial_actions=initial_actions,
        component_merge_agent=None,
        component_split_agent=_SplitAgent(),
        rounds=1,
        api_config={},
        output_dir=tmp_path,
        max_workers=1,
    )

    assert "Round 1 proposed actions for parent 'ReqA'" in caplog.text
    assert "Alpha -> split [metric_split]" in caplog.text
    assert "Beta -> save" in caplog.text
    assert "Round 1 accepted split for parent 'ReqA'" in caplog.text
    assert "Alpha -> ['AlphaCore', 'AlphaAPI']" in caplog.text


def test_run_action_feedback_rounds_regenerates_metric_actions_after_refinement(tmp_path):
    decomposed_dag = RequirementDAG(
        nodes={
            "A": RequirementNode(name="A", description="a"),
            "B": RequirementNode(name="B", description="b"),
            "C": RequirementNode(name="C", description="c"),
        },
        adjacency={
            "A": {"B"},
            "B": {"C"},
            "C": set(),
        },
    )
    architectures = [
        {
            "parent_task": "ReqA",
            "architecture": {
                "components": [
                    {
                        "name": "Wide",
                        "serves_subrequirements": ["A", "B", "C"],
                    }
                ]
            },
        }
    ]
    initial_actions = [
        {
            "task": "ReqA",
            "actions": [
                {"component": "Wide", "action": "split", "rationale": "metric round 1"},
            ],
        }
    ]
    seen_round_actions = []

    class _SplitAgent:
        def split_architecture_components(self, parent_task, architecture):
            names = [item["name"] for item in architecture.get("components", [])]
            seen_round_actions.append(
                [
                    (item["name"], item.get("recommended_action"))
                    for item in architecture.get("components", [])
                ]
            )
            if names == ["Wide"]:
                return {
                    "components": [
                        {"name": "WideCore", "serves_subrequirements": ["A", "B", "C"]},
                        {"name": "WideIO", "serves_subrequirements": ["C"]},
                    ]
                }, {"stats": {"split_group_count": 1}}
            return architecture, {"stats": {"split_group_count": 0}}

    refined, final_actions, report = run_action_feedback_rounds(
        architectures=architectures,
        initial_actions=initial_actions,
        component_merge_agent=None,
        component_split_agent=_SplitAgent(),
        rounds=2,
        api_config={},
        output_dir=tmp_path,
        max_workers=1,
        decomposed_dag=decomposed_dag,
        enable_metric_actions=True,
        metric_split_cohesion_threshold=2 / 3,
        metric_split_min_subrequirements=3,
    )

    assert [c["name"] for c in refined[0]["architecture"]["components"]] == ["WideCore", "WideIO"]
    assert final_actions == [{"task": "ReqA", "actions": []}]
    assert seen_round_actions == [
        [("Wide", "split")],
        [("WideCore", None), ("WideIO", None)],
    ]
    assert report["stats"]["rounds_completed"] == 2


def test_run_action_feedback_rounds_adapts_split_subrequirement_threshold_by_round(tmp_path):
    architectures = [
        {
            "parent_task": "ReqA",
            "architecture": {
                "components": [
                    {"name": "Wide", "serves_subrequirements": ["A", "B", "C"]},
                ]
            },
        }
    ]
    decomposed_dag = {
        "nodes": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
        "edges": [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
    }
    initial_actions = [
        {
            "task": "ReqA",
            "actions": [
                {"component": "Wide", "action": "split", "rationale": "metric round 1"},
            ],
        }
    ]

    class _SplitAgent:
        def split_architecture_components(self, parent_task, architecture):
            names = [item["name"] for item in architecture.get("components", [])]
            if names == ["Wide"]:
                return {
                    "components": [
                        {"name": "WideCore", "serves_subrequirements": ["A", "B", "C"]},
                        {"name": "WideIO", "serves_subrequirements": ["C"]},
                    ]
                }, {"stats": {"split_group_count": 1}}
            return architecture, {"stats": {"split_group_count": 0}}

    refined, final_actions, report = run_action_feedback_rounds(
        architectures=architectures,
        initial_actions=initial_actions,
        component_merge_agent=None,
        component_split_agent=_SplitAgent(),
        rounds=2,
        api_config={},
        output_dir=tmp_path,
        max_workers=1,
        decomposed_dag=decomposed_dag,
        enable_metric_actions=True,
        metric_split_cohesion_threshold=2 / 3,
        metric_split_min_subrequirements=3,
    )

    assert [c["name"] for c in refined[0]["architecture"]["components"]] == ["WideCore", "WideIO"]
    assert final_actions == [{"task": "ReqA", "actions": []}]
    assert report["stats"]["rounds_completed"] == 2


def test_run_action_feedback_rounds_removes_saved_components_from_next_round(tmp_path):
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
    initial_actions = [
        {
            "task": "ReqA",
            "actions": [
                {"component": "Alpha", "action": "save", "rationale": "stable"},
                {"component": "Beta", "action": "revise", "rationale": "needs work"},
            ],
        }
    ]

    def _action_selector(round_architectures, round_idx):
        assert round_idx == 2
        names = [
            item["name"]
            for item in round_architectures[0]["architecture"]["components"]
        ]
        assert names == ["Beta"]
        return [
            {
                "task": "ReqA",
                "actions": [
                    {"component": "Beta", "action": "save", "rationale": "stable now"},
                ],
            }
        ]

    refined, final_actions, report = run_action_feedback_rounds(
        architectures=architectures,
        initial_actions=initial_actions,
        component_merge_agent=None,
        component_split_agent=None,
        rounds=2,
        api_config={},
        output_dir=tmp_path,
        max_workers=1,
        save_stops_component=True,
        action_selector=_action_selector,
    )

    assert [c["name"] for c in refined[0]["architecture"]["components"]] == ["Alpha", "Beta"]
    assert [row["component"] for row in final_actions[0]["actions"]] == ["Beta", "Alpha"]
    assert report["stats"]["stopped_component_count"] == 2
    assert report["stats"]["stopped_components"] == {"ReqA": ["Alpha", "Beta"]}


def test_run_action_feedback_rounds_runs_revise_only_fallback_once_after_stable(tmp_path, monkeypatch):
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
    initial_actions = [
        {
            "task": "ReqA",
            "actions": [],
        }
    ]

    fallback_calls = []

    class _FakeStrategist:
        def __init__(self, api_config=None, output_dir="."):
            self.api_config = api_config or {}
            self.output_dir = output_dir

        def choose_actions(self, architecture):
            fallback_calls.append([c["name"] for c in architecture.get("components", [])])
            return [
                {"component": "Alpha", "action": "revise", "rationale": "revise alpha"},
                {"component": "Beta", "action": "save", "rationale": "save beta"},
                {"component": "Alpha", "action": "split", "rationale": "should be discarded"},
                {"component": "Beta", "action": "merge", "rationale": "should be discarded"},
            ]

    monkeypatch.setattr("agents.StrategistAgent", _FakeStrategist)

    refined, final_actions, report = run_action_feedback_rounds(
        architectures=architectures,
        initial_actions=initial_actions,
        component_merge_agent=None,
        component_split_agent=None,
        rounds=3,
        api_config={},
        output_dir=tmp_path,
        max_workers=1,
        stop_on_stable=True,
    )

    assert [c["name"] for c in refined[0]["architecture"]["components"]] == ["Alpha", "Beta"]
    assert fallback_calls == [["Alpha", "Beta"]]
    assert final_actions == [
        {
            "task": "ReqA",
            "actions": [
                {
                    "component": "Alpha",
                    "action": "revise",
                    "rationale": "revise alpha",
                }
            ],
        }
    ]
    assert report["stats"]["rounds_completed"] == 1


def test_run_action_feedback_rounds_combines_revise_fallback_with_tdd_revise(tmp_path, monkeypatch):
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
    initial_actions = [{"task": "ReqA", "actions": []}]

    class _FakeStrategist:
        def __init__(self, api_config=None, output_dir="."):
            pass

        def choose_actions(self, architecture):
            return [
                {"component": "Alpha", "action": "revise", "rationale": "revise alpha"},
            ]

    monkeypatch.setattr("agents.StrategistAgent", _FakeStrategist)

    generated_entries = [
        {
            "parent_task": "ReqA",
            "component": "Beta",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "syntax_postcheck": {"passed": True},
            "compile_postcheck": {"passed": True},
            "import_postcheck": {"passed": True},
            "compressed_feedback": "semantic mismatch one",
        },
        {
            "parent_task": "ReqA",
            "component": "Beta",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "syntax_postcheck": {"passed": True},
            "compile_postcheck": {"passed": True},
            "import_postcheck": {"passed": True},
            "compressed_feedback": "semantic mismatch two",
        },
        {
            "parent_task": "ReqA",
            "component": "Beta",
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "syntax_postcheck": {"passed": True},
            "compile_postcheck": {"passed": True},
            "import_postcheck": {"passed": True},
            "compressed_feedback": "semantic mismatch three",
        },
    ]

    refined, final_actions, report = run_action_feedback_rounds(
        architectures=architectures,
        initial_actions=initial_actions,
        component_merge_agent=None,
        component_split_agent=None,
        rounds=2,
        api_config={},
        output_dir=tmp_path,
        max_workers=1,
        stop_on_stable=True,
        existing_generated_entries=generated_entries,
    )

    assert [c["name"] for c in refined[0]["architecture"]["components"]] == ["Alpha", "Beta"]
    assert final_actions == [
        {
            "task": "ReqA",
            "actions": [
                {
                    "component": "Alpha",
                    "action": "revise",
                    "rationale": "revise alpha",
                },
                {
                    "component": "Beta",
                    "action": "revise",
                    "rationale": "Triggered after 3 consecutive TDD failures. semantic mismatch three",
                    "action_origin": "tdd_revise_threshold",
                },
            ],
        }
    ]
    assert report["stats"]["rounds_completed"] == 1


def test_build_codegen_parent_order_uses_seeded_shuffle_for_none_mode():
    ordered = ["ReqA", "ReqB", "ReqC", "ReqD"]

    shuffled_once = build_codegen_parent_order(
        dag_source="none",
        ordered_parents=ordered,
        no_graph_seed=17,
    )
    shuffled_twice = build_codegen_parent_order(
        dag_source="none",
        ordered_parents=ordered,
        no_graph_seed=17,
    )
    requirement_order = build_codegen_parent_order(
        dag_source="requirement",
        ordered_parents=ordered,
        no_graph_seed=17,
    )

    assert shuffled_once == shuffled_twice
    assert shuffled_once != ordered
    assert sorted(shuffled_once) == sorted(ordered)
    assert requirement_order == ordered


def test_architect_agent_exposes_parent_architecture_wrapper(monkeypatch):
    calls = {}

    def _fake_generate_parent_architecture(self, parent_requirement, sub_requirements, environment_feedback, existing_modules=None, dag_summary=None):
        calls["parent_requirement"] = parent_requirement
        calls["sub_requirements"] = sub_requirements
        calls["environment_feedback"] = environment_feedback
        calls["existing_modules"] = existing_modules
        calls["dag_summary"] = dag_summary
        return {"components": [{"name": "CompA"}], "component_count": 1}

    monkeypatch.setattr(GenerationAgent, "generate_parent_architecture", _fake_generate_parent_architecture)

    architect = ArchitectAgent(api_config={}, output_dir="/tmp")
    result = architect.generate_parent_architecture(
        parent_requirement={"name": "ParentReq"},
        sub_requirements=[{"name": "SubReq"}],
        environment_feedback="ctx",
        existing_modules=[{"name": "Existing"}],
        dag_summary={"node_count": 1},
    )

    assert result["component_count"] == 1
    assert calls["parent_requirement"]["name"] == "ParentReq"
    assert calls["sub_requirements"][0]["name"] == "SubReq"
    assert calls["environment_feedback"] == "ctx"


def test_process_architecture_task_keeps_parent_compatibility_fields():
    _, result = process_architecture_task(
        (0, {"name": "ReqA", "description": "Implement ReqA"}),
        api_config={},
        output_dir="/tmp",
        memory_desc="",
        dag_summary={},
    )

    assert result["parent_task"] == "ReqA"
    assert result["task"] == "ReqA"
    assert result["sub_tasks"] == []
    assert result["parent_node"]["name"] == "ReqA"


def test_build_no_graph_plan_preserves_requirement_order_without_dag():
    requirement_items = [
        {"name": "ReqA", "description": "alpha"},
        {"name": "ReqB", "description": "beta"},
    ]

    plan = build_no_graph_plan(requirement_items)

    assert [task["name"] for task in plan] == ["ReqA", "ReqB"]
    assert [task["order"] for task in plan] == [0, 1]
    assert plan[0]["description"] == "alpha"


def test_build_no_graph_parent_groups_use_empty_subtasks():
    plan = [
        {"name": "ReqB", "description": "beta", "order": 0},
        {"name": "ReqA", "description": "alpha", "order": 1},
    ]

    parent_groups = build_no_graph_parent_groups(plan)

    assert [(parent["name"], subtasks) for parent, subtasks in parent_groups] == [
        ("ReqB", []),
        ("ReqA", []),
    ]


def test_component_merge_agent_honors_explicit_target_component_hint():
    agent = ComponentMergeAgent(api_config={})
    architecture = {
        "components": [
            {
                "name": "Small",
                "responsibilities": ["helper"],
                "serves_subrequirements": ["Req::a"],
                "recommended_action": "merge",
                "recommended_target_component": "Owner",
                "recommended_action_rationale": "metric accepted",
            },
            {
                "name": "Owner",
                "responsibilities": ["helper", "owner"],
                "serves_subrequirements": ["Req::a", "Req::b"],
                "recommended_action": "save",
            },
        ]
    }

    merged, report = agent.merge_architecture_components("Req", architecture)

    assert report["stats"]["accepted_group_count"] == 1
    assert merged["component_count"] == 1
    assert merged["components"][0]["name"] == "Owner"
    assert merged["components"][0]["merged_from_ids"] == ["C1", "C2"]
