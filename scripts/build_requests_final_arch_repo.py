#!/usr/bin/env python3
"""Build a fresh requests repository bundle from the final optimized architecture.

This script does not call external APIs. It follows the downstream artifact flow
used by the existing pipeline:

- architectures.json
- package_api_plan.json
- module_plan.json
- module_assignment.json
- generated_files.json
- generated_code/

The generated code is a deterministic architecture-aligned skeleton repo built
from the final optimized requests architecture snapshot.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FINAL_RUN_ROOT = ROOT / "tmp" / "action_refine_runs" / "requests" / "20260609_093005"
FINAL_AGENTS_OUTPUT = FINAL_RUN_ROOT / "agents_output"
FINAL_ARCHITECTURES_PATH = FINAL_AGENTS_OUTPUT / "architectures.json"
FINAL_ACTIONS_PATH = FINAL_AGENTS_OUTPUT / "actions.json"
REPO_INPUT_ROOT = ROOT / "repo_input" / "requests"
OUTPUT_REPO_ROOT = ROOT / "tmp" / "requests_final_arch_from_scratch_20260609" / "requests"


def _to_snake_case(text: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(text or "").strip())
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "component"


def _to_pascal_case(text: str) -> str:
    snake = _to_snake_case(text)
    parts = [part for part in snake.split("_") if part]
    return "".join(part.capitalize() for part in parts) or "Component"


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _relative_module_name_from_path(rel_path: str) -> str:
    normalized = str(rel_path or "").replace("\\", "/").strip("/")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    return normalized.replace("/", ".")


def _build_component_lookup(architectures: List[dict]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for arch in architectures:
        if not isinstance(arch, dict):
            continue
        parent_task = str(arch.get("parent_task") or "").strip()
        architecture = arch.get("architecture", {})
        if not parent_task or not isinstance(architecture, dict):
            continue
        components = architecture.get("components", [])
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict):
                continue
            component_name = str(component.get("name") or "").strip()
            if not component_name:
                continue
            lookup[f"{parent_task}::{component_name}"] = component
    return lookup


def _simple_build_component_file_plan(
    architecture: Dict[str, Any],
    requirement: Dict[str, Any] | str,
    policy: Dict[str, Any],
) -> Dict[str, str]:
    components = architecture.get("components", []) if isinstance(architecture, dict) else []
    if not isinstance(components, list):
        return {}

    layout_root = str(policy.get("layout_root") or "requests").strip("/").strip() or "requests"
    assignment_index = policy.get("component_package_index", {}) or {}
    default_subpackage = str(policy.get("default_subpackage") or "core").strip() or "core"

    if isinstance(requirement, dict):
        parent_name = str(requirement.get("name") or "").strip()
    else:
        parent_name = str(requirement or "").strip()

    plan: Dict[str, str] = {}
    for component in components:
        if not isinstance(component, dict):
            continue
        component_name = str(component.get("name") or "").strip()
        if not component_name:
            continue
        assignment = assignment_index.get(f"{parent_name}::{component_name}", {})
        canonical_package = default_subpackage
        if isinstance(assignment, dict):
            canonical_package = str(assignment.get("canonical_package") or default_subpackage).strip() or default_subpackage
        elif isinstance(assignment, str):
            canonical_package = assignment.strip() or default_subpackage
        plan[component_name] = f"{layout_root}/{canonical_package}/{_to_snake_case(component_name)}.py"
    return plan


def _build_package_api_plan(architectures: List[dict]) -> Dict[str, Any]:
    from package_api_plan_builder import build_package_api_plan

    layout_policy: Dict[str, Any] = {
        "enabled": True,
        "layout_root": "requests",
        "top_whitelist": ["requests", "docs", "tests", "tools", "examples"],
        "alias_map": {},
    }
    package_api_plan = build_package_api_plan(
        architectures=architectures,
        layout_policy=layout_policy,
        build_component_file_plan=_simple_build_component_file_plan,
    )

    package_modules: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    component_index: Dict[str, Dict[str, Any]] = {}
    curated_components: List[Dict[str, Any]] = []

    for row in package_api_plan.get("components", []):
        if not isinstance(row, dict):
            continue
        component_name = str(row.get("component") or "").strip()
        parent_task = str(row.get("parent_task") or "").strip()
        planned_file_path = str(row.get("planned_file_path") or "").strip()
        main_class = _to_pascal_case(component_name)
        snake_alias = _to_snake_case(component_name)
        describe_fn = f"describe_{snake_alias}"
        list_fn = f"get_{snake_alias}_responsibilities"
        exports = [main_class, snake_alias, describe_fn, list_fn]

        normalized_row = dict(row)
        normalized_row["export_symbols"] = exports
        curated_components.append(normalized_row)
        component_index[f"{parent_task}::{component_name}"] = normalized_row

        package_dir = str(normalized_row.get("package_dir") or "").strip()
        module_name = Path(planned_file_path).stem
        package_modules[package_dir].append(
            {
                "module_name": module_name,
                "component": component_name,
                "planned_file_path": planned_file_path,
                "canonical_package": normalized_row.get("canonical_package"),
                "export_symbols": exports,
            }
        )

    curated_packages: List[Dict[str, Any]] = []
    for package_dir, modules in sorted(package_modules.items(), key=lambda item: item[0]):
        planned_exports: List[str] = []
        for module in modules:
            planned_exports.extend(module.get("export_symbols", []))
        curated_packages.append(
            {
                "package_dir": package_dir,
                "module_count": len(modules),
                "modules": modules,
                "planned_exports": list(dict.fromkeys(planned_exports)),
            }
        )

    package_api_plan["components"] = curated_components
    package_api_plan["packages"] = curated_packages
    package_api_plan["component_index"] = component_index
    package_api_plan["_meta"] = {
        "generated_by": "build_requests_final_arch_repo.py",
        "mode": "manual_no_api_from_final_architecture",
    }
    return package_api_plan


def _build_module_plan(architectures: List[dict], package_api_plan: Dict[str, Any]) -> Dict[str, Any]:
    component_lookup = _build_component_lookup(architectures)
    package_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in package_api_plan.get("components", []):
        if isinstance(row, dict):
            package_groups[str(row.get("package_subpath") or "").strip()].append(row)

    module_families: List[Dict[str, Any]] = []
    for package_subpath, rows in sorted(package_groups.items(), key=lambda item: item[0]):
        component_names = [str(row.get("component") or "").strip() for row in rows]
        parent_names = sorted({str(row.get("parent_task") or "").strip() for row in rows if str(row.get("parent_task") or "").strip()})
        covered_features: List[str] = []
        for row in rows:
            key = f"{row.get('parent_task','')}::{row.get('component','')}"
            component = component_lookup.get(key, {})
            serves = component.get("serves_subrequirements", [])
            if isinstance(serves, list):
                for item in serves:
                    token = str(item or "").strip()
                    if token and token not in covered_features:
                        covered_features.append(token)
        module_families.append(
            {
                "parent_task": parent_names[0] if len(parent_names) == 1 else "mixed",
                "module_family": package_subpath.replace("/", "_") or "core",
                "covers_features": covered_features,
                "components": component_names,
                "package_subpath": package_subpath,
                "rationale": "Deterministic local grouping derived from canonical package planning for the final optimized architecture.",
                "source": "manual_no_api",
            }
        )

    return {
        "default_package": package_api_plan.get("default_package", "core"),
        "module_families": module_families,
        "stats": {
            "module_family_count": len(module_families),
            "assigned_components": len(package_api_plan.get("components", [])),
        },
        "_meta": {
            "generated_by": "build_requests_final_arch_repo.py",
            "mode": "manual_no_api_from_final_architecture",
        },
    }


def _build_module_assignment(package_api_plan: Dict[str, Any]) -> Dict[str, Any]:
    assignments: List[Dict[str, Any]] = []
    component_package_path_index: Dict[str, str] = {}

    for row in package_api_plan.get("components", []):
        if not isinstance(row, dict):
            continue
        parent_task = str(row.get("parent_task") or "").strip()
        component_name = str(row.get("component") or "").strip()
        key = f"{parent_task}::{component_name}"
        package_subpath = str(row.get("package_subpath") or "").strip()
        planned_file_path = str(row.get("planned_file_path") or "").strip()
        module_family = package_subpath.replace("/", "_") or "core"
        component_package_path_index[key] = package_subpath
        assignments.append(
            {
                "parent_task": parent_task,
                "component": component_name,
                "package_subpath": package_subpath,
                "planned_file_path": planned_file_path,
                "module_family": module_family,
            }
        )

    return {
        "component_package_path_index": component_package_path_index,
        "assignments": assignments,
        "stats": {
            "assigned_components": len(assignments),
            "non_generic_subpaths": len([item for item in component_package_path_index.values() if item]),
        },
        "_meta": {
            "generated_by": "build_requests_final_arch_repo.py",
            "mode": "manual_no_api_from_final_architecture",
        },
    }


def _render_component_module(row: Dict[str, Any], component: Dict[str, Any]) -> str:
    parent_task = str(row.get("parent_task") or "").strip()
    component_name = str(row.get("component") or "").strip()
    responsibilities = component.get("responsibilities", [])
    serves = component.get("serves_subrequirements", [])
    main_class = _to_pascal_case(component_name)
    snake_alias = _to_snake_case(component_name)
    describe_fn = f"describe_{snake_alias}"
    list_fn = f"get_{snake_alias}_responsibilities"

    body = f'''"""Architecture-derived component module for {component_name}."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

PARENT_TASK = {parent_task!r}
COMPONENT_NAME = {component_name!r}
RESPONSIBILITIES: List[str] = {json.dumps(responsibilities, ensure_ascii=False, indent=4)}
SERVES_SUBREQUIREMENTS: List[str] = {json.dumps(serves, ensure_ascii=False, indent=4)}


@dataclass
class {main_class}:
    """Deterministic local skeleton for the optimized architecture component."""

    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def describe_component(self) -> Dict[str, Any]:
        return {{
            "parent_task": PARENT_TASK,
            "component": COMPONENT_NAME,
            "responsibilities": list(RESPONSIBILITIES),
            "serves_subrequirements": list(SERVES_SUBREQUIREMENTS),
            "configuration": dict(self.configuration),
            "metadata": dict(self.metadata),
        }}

    def get_responsibilities(self) -> List[str]:
        return list(RESPONSIBILITIES)

    def supports(self, text: str) -> bool:
        probe = str(text or "").strip().lower()
        if not probe:
            return False
        return any(probe in item.lower() for item in RESPONSIBILITIES)


def {snake_alias}(**configuration: Any) -> {main_class}:
    """Factory alias matching the component snake-case export."""

    return {main_class}(configuration=dict(configuration))


def {describe_fn}() -> Dict[str, Any]:
    return {main_class}().describe_component()


def {list_fn}() -> List[str]:
    return list(RESPONSIBILITIES)


__all__ = [
    "{main_class}",
    "{snake_alias}",
    "{describe_fn}",
    "{list_fn}",
]
'''
    return body


def _render_registry_module(registry_rows: List[Dict[str, Any]]) -> str:
    module_names = sorted({str(row.get("module_name") or "").strip() for row in registry_rows if str(row.get("module_name") or "").strip()})
    return f'''"""Registry for the fresh requests repository built from the final optimized architecture."""

from __future__ import annotations

from typing import Dict, Iterable, List

ARCHITECTURE_COMPONENTS: List[dict] = {json.dumps(registry_rows, ensure_ascii=False, indent=4)}
ARCHITECTURE_COMPONENT_INDEX: Dict[str, dict] = {{
    f"{{row['parent_task']}}::{{row['component']}}": row for row in ARCHITECTURE_COMPONENTS
}}


def iter_component_modules() -> List[str]:
    return sorted({module_names!r})


def iter_component_rows() -> Iterable[dict]:
    return list(ARCHITECTURE_COMPONENTS)


__all__ = [
    "ARCHITECTURE_COMPONENTS",
    "ARCHITECTURE_COMPONENT_INDEX",
    "iter_component_modules",
    "iter_component_rows",
]
'''


def _write_package_inits(generated_code_root: Path, package_api_plan: Dict[str, Any]) -> None:
    package_to_modules: Dict[str, List[str]] = defaultdict(list)
    for row in package_api_plan.get("components", []):
        if not isinstance(row, dict):
            continue
        package_dir = str(row.get("package_dir") or "").strip()
        module_name = Path(str(row.get("planned_file_path") or "")).stem
        if package_dir and module_name:
            package_to_modules[package_dir].append(module_name)

    root_package_dir = generated_code_root / "requests"
    root_subpackages = sorted({Path(package_dir).relative_to("requests").parts[0] for package_dir in package_to_modules if package_dir.startswith("requests/") and len(Path(package_dir).parts) > 1})
    root_lines = [
        "# Auto-generated package exports for the fresh final-architecture requests repo.",
        "",
    ]
    for name in root_subpackages:
        root_lines.append(f"from . import {name}")
    root_lines.append("from . import final_arch_registry")
    root_lines.append("")
    root_lines.append("__all__ = [")
    for name in root_subpackages:
        root_lines.append(f'    "{name}",')
    root_lines.append('    "final_arch_registry",')
    root_lines.append("]")
    (root_package_dir / "__init__.py").write_text("\n".join(root_lines) + "\n", encoding="utf-8")

    for package_dir, module_names in sorted(package_to_modules.items(), key=lambda item: item[0]):
        package_path = generated_code_root / package_dir
        package_path.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Auto-generated package exports for {package_dir}.",
            "",
        ]
        for name in sorted(dict.fromkeys(module_names)):
            lines.append(f"from . import {name}")
        lines.append("")
        lines.append("__all__ = [")
        for name in sorted(dict.fromkeys(module_names)):
            lines.append(f'    "{name}",')
        lines.append("]")
        (package_path / "__init__.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_setup_py() -> str:
    return """from setuptools import find_packages, setup

setup(
    name=\"requests_final_arch_generated\",
    version=\"0.1.0\",
    description=\"Fresh requests repository generated from the final optimized architecture.\",
    packages=find_packages(),
    include_package_data=True,
)
"""


def _render_smoke_test() -> str:
    return """from __future__ import annotations

import importlib

from requests.final_arch_registry import ARCHITECTURE_COMPONENT_INDEX, iter_component_modules


def test_final_architecture_component_count() -> None:
    assert len(ARCHITECTURE_COMPONENT_INDEX) == 60


def test_all_final_architecture_modules_import() -> None:
    for module_name in iter_component_modules():
        module = importlib.import_module(module_name)
        assert module is not None
"""


def build_bundle() -> Dict[str, Any]:
    if not FINAL_ARCHITECTURES_PATH.exists():
        raise FileNotFoundError(f"Missing final architectures snapshot: {FINAL_ARCHITECTURES_PATH}")

    if OUTPUT_REPO_ROOT.exists():
        shutil.rmtree(OUTPUT_REPO_ROOT)

    agents_output_dir = OUTPUT_REPO_ROOT / "agents_output"
    generated_code_dir = agents_output_dir / "generated_code"
    requests_package_dir = generated_code_dir / "requests"
    tests_dir = generated_code_dir / "tests"
    readme_output_dir = OUTPUT_REPO_ROOT / "readme_output"

    generated_code_dir.mkdir(parents=True, exist_ok=True)
    requests_package_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    readme_output_dir.mkdir(parents=True, exist_ok=True)

    architectures = json.loads(FINAL_ARCHITECTURES_PATH.read_text(encoding="utf-8"))
    actions = json.loads(FINAL_ACTIONS_PATH.read_text(encoding="utf-8")) if FINAL_ACTIONS_PATH.exists() else []
    package_api_plan = _build_package_api_plan(architectures)
    module_plan = _build_module_plan(architectures, package_api_plan)
    module_assignment = _build_module_assignment(package_api_plan)
    component_lookup = _build_component_lookup(architectures)

    _copy_if_exists(REPO_INPUT_ROOT / "README.req", OUTPUT_REPO_ROOT / "README.req")
    _copy_if_exists(REPO_INPUT_ROOT / "readme_output" / "requirements.json", readme_output_dir / "requirements.json")

    for name in [
        "architectures.json",
        "actions.json",
        "action_refinement_report.json",
        "component_metric_action_report.json",
        "revise_fallback_report.json",
        "requirements_for_dag.json",
        "requirement_dag.json",
        "decomposed_dag.json",
        "plan.json",
    ]:
        _copy_if_exists(FINAL_AGENTS_OUTPUT / name, agents_output_dir / name)

    _json_dump(package_api_plan, agents_output_dir / "package_api_plan.json")
    _json_dump(module_plan, agents_output_dir / "module_plan.json")
    _json_dump(module_assignment, agents_output_dir / "module_assignment.json")

    generated_entries: List[Dict[str, Any]] = []
    registry_rows: List[Dict[str, Any]] = []

    for row in package_api_plan.get("components", []):
        if not isinstance(row, dict):
            continue
        parent_task = str(row.get("parent_task") or "").strip()
        component_name = str(row.get("component") or "").strip()
        rel_path = str(row.get("planned_file_path") or "").strip()
        if not parent_task or not component_name or not rel_path:
            continue
        component = component_lookup.get(f"{parent_task}::{component_name}", {})
        abs_path = generated_code_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(_render_component_module(row, component), encoding="utf-8")

        module_name = _relative_module_name_from_path(rel_path)
        registry_row = {
            "parent_task": parent_task,
            "component": component_name,
            "module_name": module_name,
            "planned_file_path": rel_path,
            "package_dir": row.get("package_dir"),
            "package_subpath": row.get("package_subpath"),
            "canonical_package": row.get("canonical_package"),
            "responsibilities": component.get("responsibilities", []),
            "serves_subrequirements": component.get("serves_subrequirements", []),
        }
        registry_rows.append(registry_row)
        generated_entries.append(
            {
                "component": component_name,
                "parent_task": parent_task,
                "task": parent_task,
                "component_responsibilities": component.get("responsibilities", []),
                "component_export_symbols": row.get("export_symbols", []),
                "files": {
                    "code": rel_path,
                },
                "planned_file_path": rel_path,
                "module_path": module_name,
                "generation_status": "manual_no_api_generated",
                "tdd_passed": None,
                "tdd_final_pytest_rc": None,
            }
        )

    (requests_package_dir / "final_arch_registry.py").write_text(
        _render_registry_module(registry_rows),
        encoding="utf-8",
    )
    _json_dump(registry_rows, requests_package_dir / "final_arch_registry.json")
    _write_package_inits(generated_code_dir, package_api_plan)

    (generated_code_dir / "setup.py").write_text(_render_setup_py(), encoding="utf-8")
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_final_architecture_manifest.py").write_text(_render_smoke_test(), encoding="utf-8")

    _json_dump(generated_entries, agents_output_dir / "generated_files.json")

    summary = {
        "repo_root": str(OUTPUT_REPO_ROOT),
        "agents_output": str(agents_output_dir),
        "generated_code": str(generated_code_dir),
        "package_api_plan": str(agents_output_dir / "package_api_plan.json"),
        "module_plan": str(agents_output_dir / "module_plan.json"),
        "module_assignment": str(agents_output_dir / "module_assignment.json"),
        "generated_files": str(agents_output_dir / "generated_files.json"),
        "component_count": len(registry_rows),
        "package_count": package_api_plan.get("package_count", 0),
    }
    _json_dump(summary, agents_output_dir / "manual_generation_summary.json")
    return summary


def main() -> None:
    summary = build_bundle()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
