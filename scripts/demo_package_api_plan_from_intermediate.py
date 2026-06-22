#!/usr/bin/env python3
"""Build demo package_api_plan JSON from existing intermediate artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from package_api_plan_builder import build_package_api_plan  # noqa: E402
from run_agents import _build_component_file_plan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate demo package_api_plan files from existing architectures.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="agents_output directory containing architectures.json",
    )
    parser.add_argument(
        "--layout-root",
        type=str,
        default="",
        help="Layout root override (default: infer from existing package_api_plan.json or 'statsmodels')",
    )
    parser.add_argument(
        "--preview-packages",
        type=int,
        default=5,
        help="How many package rows to keep in preview JSON",
    )
    parser.add_argument(
        "--preview-components",
        type=int,
        default=8,
        help="How many component rows to keep in preview JSON",
    )
    return parser.parse_args()


def _infer_layout_root(output_dir: Path, override: str) -> str:
    text = str(override or "").strip()
    if text:
        return text
    existing = output_dir / "package_api_plan.json"
    if existing.exists():
        try:
            payload = json.loads(existing.read_text(encoding="utf-8"))
            root = str(payload.get("layout_root", "")).strip()
            if root:
                return root
        except Exception:
            pass
    return "statsmodels"


def _build_preview(full: Dict[str, Any], preview_packages: int, preview_components: int) -> Dict[str, Any]:
    packages = full.get("packages", []) if isinstance(full.get("packages", []), list) else []
    components = full.get("components", []) if isinstance(full.get("components", []), list) else []
    component_index = full.get("component_index", {}) if isinstance(full.get("component_index", {}), dict) else {}

    package_rows = []
    for row in packages[: max(0, int(preview_packages))]:
        if not isinstance(row, dict):
            continue
        package_rows.append(
            {
                "package_dir": row.get("package_dir"),
                "module_count": row.get("module_count"),
                "planned_exports_preview": (row.get("planned_exports", []) or [])[:10],
                "modules_preview": (row.get("modules", []) or [])[:3],
            }
        )

    component_rows = []
    for row in components[: max(0, int(preview_components))]:
        if not isinstance(row, dict):
            continue
        component_rows.append(
            {
                "parent_task": row.get("parent_task"),
                "component": row.get("component"),
                "planned_file_path": row.get("planned_file_path"),
                "export_symbols_preview": (row.get("export_symbols", []) or [])[:10],
            }
        )

    index_preview = {}
    for idx, key in enumerate(component_index.keys()):
        if idx >= max(0, int(preview_components)):
            break
        row = component_index.get(key, {})
        if isinstance(row, dict):
            index_preview[key] = {
                "planned_file_path": row.get("planned_file_path"),
                "export_symbols_preview": (row.get("export_symbols", []) or [])[:8],
            }

    return {
        "layout_root": full.get("layout_root"),
        "package_count": full.get("package_count"),
        "component_count": full.get("component_count"),
        "packages_preview": package_rows,
        "components_preview": component_rows,
        "component_index_preview": index_preview,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    architectures_path = output_dir / "architectures.json"
    if not architectures_path.exists():
        raise FileNotFoundError(f"architectures.json not found: {architectures_path}")

    architectures = json.loads(architectures_path.read_text(encoding="utf-8"))
    if isinstance(architectures, dict):
        architectures = architectures.get("architectures", [])
    if not isinstance(architectures, list):
        raise ValueError("architectures.json must be a list or {'architectures': [...]} object")

    layout_root = _infer_layout_root(output_dir, args.layout_root)
    layout_policy: Dict[str, Any] = {
        "enabled": True,
        "layout_root": layout_root,
        "top_whitelist": ["statsmodels", "docs", "tests", "tools", "examples"],
        "alias_map": {},
    }

    full = build_package_api_plan(
        architectures=architectures,
        layout_policy=layout_policy,
        build_component_file_plan=_build_component_file_plan,
    )
    preview = _build_preview(full, args.preview_packages, args.preview_components)

    full_path = output_dir / "package_api_plan.demo.full.json"
    preview_path = output_dir / "package_api_plan.demo.preview.json"
    full_path.write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    preview_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Demo package_api_plan generated ===")
    print(f"Input architectures: {architectures_path}")
    print(f"Full output: {full_path}")
    print(f"Preview output: {preview_path}")
    print(
        "Stats: layout_root={lr}, package_count={pc}, component_count={cc}".format(
            lr=full.get("layout_root", ""),
            pc=full.get("package_count", 0),
            cc=full.get("component_count", 0),
        )
    )


if __name__ == "__main__":
    main()

