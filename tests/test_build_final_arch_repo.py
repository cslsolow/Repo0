from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "scripts" / "build_final_arch_repo.py"
SPEC = importlib.util.spec_from_file_location("build_final_arch_repo", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

GenerationConfig = MODULE.GenerationConfig
_stabilize_component_rows = MODULE._stabilize_component_rows
_render_smoke_test = MODULE._render_smoke_test
resolve_generation_config = MODULE.resolve_generation_config


def test_resolve_generation_config_uses_repo_specific_paths() -> None:
    config = resolve_generation_config(
        repo="django",
        run_root=ROOT / "tmp" / "action_refine_runs" / "django" / "20260609_153734",
        output_repo_root=ROOT / "tmp" / "django_final_arch_from_scratch_20260609" / "django",
    )

    assert isinstance(config, GenerationConfig)
    assert config.repo == "django"
    assert config.package_name == "django"
    assert config.repo_input_root == ROOT / "repo_input" / "django"
    assert config.final_architectures_path.name == "architectures.json"
    assert config.final_actions_path.name == "actions.json"
    assert config.output_repo_root == ROOT / "tmp" / "django_final_arch_from_scratch_20260609" / "django"


def test_resolve_generation_config_normalizes_hyphenated_repo_name() -> None:
    config = resolve_generation_config(
        repo="scikit-learn",
        run_root=ROOT / "tmp" / "action_refine_runs" / "scikit-learn" / "20260609_000000",
        output_repo_root=ROOT / "tmp" / "scikit_learn_final_arch_from_scratch_20260609" / "scikit-learn",
    )

    assert config.repo == "scikit-learn"
    assert config.package_name == "scikit_learn"


def test_render_smoke_test_is_dynamic_per_repo() -> None:
    rendered = _render_smoke_test(package_name="statsmodels", component_count=86)

    assert "from statsmodels.final_arch_registry import ARCHITECTURE_COMPONENT_INDEX" in rendered
    assert "assert len(ARCHITECTURE_COMPONENT_INDEX) == 86" in rendered


def test_stabilize_component_rows_avoids_duplicate_component_and_path_collisions() -> None:
    rows = [
        {
            "parent_task": "Syndication and Feed Generation",
            "component": "FeedSerializationEngine",
            "planned_file_path": "django/syndication_feed/feed_serialization_engine.py",
        },
        {
            "parent_task": "Syndication and Feed Generation",
            "component": "FeedSerializationEngine",
            "planned_file_path": "django/syndication_feed/feed_serialization_engine.py",
        },
    ]

    stabilized = _stabilize_component_rows(rows)

    assert stabilized[0]["component_key"] == "Syndication and Feed Generation::FeedSerializationEngine"
    assert stabilized[1]["component_key"] == "Syndication and Feed Generation::FeedSerializationEngine#2"
    assert stabilized[0]["planned_file_path"] == "django/syndication_feed/feed_serialization_engine.py"
    assert stabilized[1]["planned_file_path"] == "django/syndication_feed/feed_serialization_engine__2.py"
