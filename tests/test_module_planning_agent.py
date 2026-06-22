import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.cognitive.module_assignment import ModuleAssignmentAgent  # noqa: E402
from agents.cognitive.module_planner import ModulePlanningAgent  # noqa: E402


def test_module_planner_proposes_candidate_families_without_component_assignment():
    planner = ModulePlanningAgent(api_config=None, output_dir=".")
    architectures = [
        {
            "parent_task": "Time series analysis module",
            "architecture": {
                "requirement": {"description": "Time-series models, diagnostics, and forecasting."},
                "sub_requirements": [
                    {"name": "state-space-core", "description": "State space and Kalman routines"},
                    {"name": "classical-models", "description": "ARIMA and classical forecasting"},
                ],
                "components": [
                    {
                        "name": "StateSpaceCore",
                        "responsibilities": ["Implement SARIMAX and Kalman filtering"],
                        "serves_subrequirements": ["state-space-core"],
                    },
                    {
                        "name": "ModelLibrary",
                        "responsibilities": ["Implement ARIMA utilities and forecasting helpers"],
                        "serves_subrequirements": ["classical-models"],
                    },
                ],
            },
        }
    ]
    layout_policy = {
        "canonical_packages": ["core", "time_series", "diagnostics_statistical"],
        "default_subpackage": "core",
    }

    report = planner.plan_modules(architectures, [], layout_policy)

    assert report["module_families"]
    assert "component_package_path_index" not in report
    assert "plans" not in report
    assert any("time" in row["module_family"] or "state" in row["module_family"] for row in report["module_families"])


def test_module_assignment_maps_components_into_planned_subdirectories():
    planner = ModulePlanningAgent(api_config=None, output_dir=".")
    assigner = ModuleAssignmentAgent(api_config=None, output_dir=".")
    architectures = [
        {
            "parent_task": "Time series analysis module",
            "architecture": {
                "requirement": {"description": "Time-series models, diagnostics, and forecasting."},
                "sub_requirements": [
                    {"name": "state-space-core", "description": "State space and Kalman routines"},
                    {"name": "classical-models", "description": "ARIMA and classical forecasting"},
                ],
                "components": [
                    {
                        "name": "StateSpaceCore",
                        "responsibilities": ["Implement SARIMAX and Kalman filtering"],
                        "serves_subrequirements": ["state-space-core"],
                    },
                    {
                        "name": "ModelLibrary",
                        "responsibilities": ["Implement ARIMA utilities and forecasting helpers"],
                        "serves_subrequirements": ["classical-models"],
                    },
                ],
            },
        }
    ]
    layout_policy = {
        "canonical_packages": ["core", "time_series", "diagnostics_statistical"],
        "default_subpackage": "core",
    }

    module_plan = planner.plan_modules(architectures, [], layout_policy)
    assignment = assigner.assign_modules(architectures, [], layout_policy, module_plan)
    index = assignment["component_package_path_index"]

    assert index["Time series analysis module::StateSpaceCore"] == index["Time series analysis module::ModelLibrary"]
    assert "/" in index["Time series analysis module::StateSpaceCore"]
    assert "time" in index["Time series analysis module::StateSpaceCore"] or "state" in index["Time series analysis module::StateSpaceCore"]


def test_module_assignment_avoids_flat_generic_root_when_split_requested():
    planner = ModulePlanningAgent(api_config=None, output_dir=".")
    assigner = ModuleAssignmentAgent(api_config=None, output_dir=".")
    architectures = [
        {
            "parent_task": "Numerical optimization utilities",
            "architecture": {
                "requirement": {"description": "Optimization backends and solvers."},
                "sub_requirements": [
                    {"name": "solver-backends", "description": "Iterative and trust-region solvers"},
                ],
                "components": [
                    {
                        "name": "OptimizationEngine",
                        "responsibilities": ["Implement constrained and unconstrained solver orchestration"],
                        "serves_subrequirements": ["solver-backends"],
                    }
                ],
            },
        }
    ]
    actions = [
        {
            "task": "Numerical optimization utilities",
            "actions": [
                {"component": "OptimizationEngine", "action": "split", "rationale": "keep solver code isolated"}
            ],
        }
    ]
    layout_policy = {
        "canonical_packages": ["core", "numerical_estimation"],
        "default_subpackage": "core",
    }

    module_plan = planner.plan_modules(architectures, actions, layout_policy)
    assignment = assigner.assign_modules(architectures, actions, layout_policy, module_plan)
    planned = assignment["component_package_path_index"]["Numerical optimization utilities::OptimizationEngine"]

    assert planned != "core"
    assert "/" in planned


def test_module_context_sampling_uses_dynamic_repo_root():
    with tempfile.TemporaryDirectory() as tmp:
        generated = Path(tmp) / "generated_code" / "my_repo"
        (generated / "pkg_a" / "sub_one").mkdir(parents=True)
        (generated / "pkg_b" / "sub_two").mkdir(parents=True)

        planner = ModulePlanningAgent(api_config={"repo": "my-repo"}, output_dir=tmp)
        assigner = ModuleAssignmentAgent(api_config={"repo": "my-repo"}, output_dir=tmp)

        planner_ctx = planner._collect_existing_package_context()
        assigner_ctx = assigner._collect_existing_package_context()

        assert planner_ctx["top_level_packages"] == ["pkg_a", "pkg_b"]
        assert assigner_ctx["top_level_packages"] == ["pkg_a", "pkg_b"]
        assert "pkg_a/sub_one" in planner_ctx["sample_subpackages"]
        assert "pkg_b/sub_two" in assigner_ctx["sample_subpackages"]


def test_module_context_normalizes_hyphenated_repo_root():
    with tempfile.TemporaryDirectory() as tmp:
        generated = Path(tmp) / "generated_code" / "my_repo"
        (generated / "pkg_a").mkdir(parents=True)

        planner = ModulePlanningAgent(api_config={"repo": "my-repo"}, output_dir=tmp)
        assigner = ModuleAssignmentAgent(api_config={"repo": "my-repo"}, output_dir=tmp)

        assert planner._primary_generated_package_root() == generated
        assert assigner._primary_generated_package_root() == generated
