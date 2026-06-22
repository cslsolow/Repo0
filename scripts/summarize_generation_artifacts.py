#!/usr/bin/env python3
"""Summarize generation artifacts: requirements, decomposed nodes, files, and APIs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_requirements(requirement_dag: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(requirement_dag, dict):
        return result

    for node in requirement_dag.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        name = str(node.get("name", "")).strip()
        if not name:
            continue
        result[name] = {
            "name": name,
            "description": str(node.get("description", "")),
            "metadata": node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {},
        }
    return result


def collect_decomposed_nodes(decomposed_dag: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    if not isinstance(decomposed_dag, dict):
        return by_parent

    for node in decomposed_dag.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
        parent = str(meta.get("parent", "")).strip()
        node_name = str(node.get("name", "")).strip()
        if not parent or not node_name:
            continue
        entry = {
            "name": node_name,
            "description": str(node.get("description", "")),
            "order": meta.get("order"),
            "metadata": meta,
        }
        by_parent.setdefault(parent, []).append(entry)

    for parent, nodes in by_parent.items():
        nodes.sort(key=lambda x: (x.get("order") is None, x.get("order", 10**9), x.get("name", "")))
    return by_parent


def extract_python_api(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists() or file_path.suffix != ".py":
        return {"module": None, "functions": [], "classes": []}

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return {"module": None, "functions": [], "classes": []}

    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                {
                    "name": node.name,
                    "args": [a.arg for a in node.args.args],
                    "lineno": getattr(node, "lineno", None),
                }
            )
        elif isinstance(node, ast.ClassDef):
            methods: List[Dict[str, Any]] = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(
                        {
                            "name": sub.name,
                            "args": [a.arg for a in sub.args.args],
                            "lineno": getattr(sub, "lineno", None),
                        }
                    )
            classes.append(
                {
                    "name": node.name,
                    "lineno": getattr(node, "lineno", None),
                    "methods": methods,
                }
            )

    return {
        "module": file_path.stem,
        "functions": functions,
        "classes": classes,
    }


def resolve_generated_file_path(raw_path: str, agents_output: Path, repo_root: Path) -> Optional[Path]:
    if not raw_path:
        return None

    candidate = Path(raw_path)

    normalized = raw_path.replace("\\", "/")
    candidate_markers = ["generated_code"]
    for marker in candidate_markers:
        marker_token = f"{marker}/"
        if marker_token in normalized:
            suffix = normalized.split(marker_token, 1)[1]
            mapped = agents_output / marker / suffix
            if mapped.exists():
                return mapped

    if candidate.exists():
        return candidate

    # Try relative to repo root
    rel_candidate = repo_root / raw_path
    if rel_candidate.exists():
        return rel_candidate

    # Last try: basename under expected generated code trees.
    basename = candidate.name
    for marker in candidate_markers:
        tree = agents_output / marker
        if not tree.exists():
            continue
        matches = list(tree.rglob(basename))
        if matches:
            return matches[0]

    return None


def collect_components_with_files_and_apis(
    generated_files_data: Any,
    agents_output: Path,
    repo_root: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    if not isinstance(generated_files_data, list):
        return by_parent

    for item in generated_files_data:
        if not isinstance(item, dict):
            continue

        parent = str(item.get("parent_task") or item.get("task") or "").strip()
        if not parent:
            continue

        component = str(item.get("component", "")).strip()
        sub_tasks = item.get("sub_tasks", []) if isinstance(item.get("sub_tasks"), list) else []
        files_obj = item.get("files", {}) if isinstance(item.get("files"), dict) else {}

        file_entries: List[Dict[str, Any]] = []
        for key, raw_path in files_obj.items():
            if not isinstance(raw_path, str):
                continue
            resolved = resolve_generated_file_path(raw_path, agents_output, repo_root)
            api = extract_python_api(resolved) if resolved else {"module": None, "functions": [], "classes": []}
            file_entries.append(
                {
                    "kind": key,
                    "raw_path": raw_path,
                    "resolved_path": str(resolved) if resolved else None,
                    "exists": bool(resolved and resolved.exists()),
                    "api": api,
                }
            )

        by_parent.setdefault(parent, []).append(
            {
                "component": component,
                "sub_tasks": sub_tasks,
                "files": file_entries,
            }
        )

    return by_parent


def build_summary(agents_output: Path, repo_root: Path) -> Dict[str, Any]:
    requirement_dag = load_json(agents_output / "requirement_dag.json")
    decomposed_dag = load_json(agents_output / "decomposed_dag.json")
    generated_files = load_json(agents_output / "generated_files.json")

    requirements = collect_requirements(requirement_dag)
    decomposed_by_parent = collect_decomposed_nodes(decomposed_dag)
    components_by_parent = collect_components_with_files_and_apis(generated_files, agents_output, repo_root)

    requirement_summaries: List[Dict[str, Any]] = []
    all_requirement_names = sorted(set(requirements.keys()) | set(decomposed_by_parent.keys()) | set(components_by_parent.keys()))

    for req_name in all_requirement_names:
        req_payload = requirements.get(req_name, {"name": req_name, "description": "", "metadata": {}})
        requirement_summaries.append(
            {
                "requirement": req_payload,
                "decomposed_nodes": decomposed_by_parent.get(req_name, []),
                "components": components_by_parent.get(req_name, []),
            }
        )

    return {
        "agents_output": str(agents_output),
        "repo_root": str(repo_root),
        "stats": {
            "requirement_count": len(requirement_summaries),
            "decomposed_node_count": sum(len(item["decomposed_nodes"]) for item in requirement_summaries),
            "component_count": sum(len(item["components"]) for item in requirement_summaries),
            "file_count": sum(len(comp.get("files", [])) for item in requirement_summaries for comp in item["components"]),
        },
        "requirements": requirement_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize requirement -> decomposed nodes -> files -> APIs from generation artifacts",
    )
    parser.add_argument("--agents-output", type=Path, required=True, help="Path to agents_output directory")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: parent of agents_output)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <agents-output>/generation_summary.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agents_output = args.agents_output.resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else agents_output.parent.resolve()

    summary = build_summary(agents_output, repo_root)

    output_path = args.output.resolve() if args.output else agents_output / "generation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Summary written to: {output_path}")
    print(
        "Stats: requirements={requirement_count}, decomposed_nodes={decomposed_node_count}, components={component_count}, files={file_count}".format(
            **summary["stats"]
        )
    )


if __name__ == "__main__":
    main()
