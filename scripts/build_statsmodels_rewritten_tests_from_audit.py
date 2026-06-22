#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_REPO = ROOT / "golden_repos" / "statsmodels"
DEFAULT_TASKS_JSONL = ROOT / "tmp" / "full_tasks" / "statsmodels.tasks.jsonl"


class TaskEntry:
    def __init__(self, task_id: str, file: str, module: str, function: str, status: str) -> None:
        self.task_id = task_id
        self.file = file
        self.module = module
        self.function = function
        self.status = status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a persisted rewritten functional test suite for statsmodels from prior audit results."
    )
    parser.add_argument(
        "--golden-repo",
        type=Path,
        default=DEFAULT_GOLDEN_REPO,
        help="Path to the original statsmodels repository with golden tests.",
    )
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        default=DEFAULT_TASKS_JSONL,
        help="Task manifest mapping files/modules/functions to audit nodeids.",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        action="append",
        required=True,
        help="One or more function-level audit JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where rewritten_tests/ and manifest files will be written.",
    )
    parser.add_argument(
        "--min-pass-ratio",
        type=float,
        default=1.0,
        help="Minimum share of passed/skipped tasks required to keep a file. Default keeps only fully passing files.",
    )
    return parser.parse_args()


def load_task_entries(tasks_jsonl: Path) -> Dict[str, List[TaskEntry]]:
    rows = [
        json.loads(line)
        for line in tasks_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_file: Dict[str, List[TaskEntry]] = defaultdict(list)
    for row in rows:
        file_path = str(row["file"])
        module = str(row["module"])
        for fn in row.get("functions", []) or []:
            by_file[file_path].append(
                TaskEntry(
                    task_id=str(row["id"]),
                    file=file_path,
                    module=module,
                    function=str(fn),
                    status="unknown",
                )
            )
    return by_file


def load_audit_entries(audit_paths: Iterable[Path]) -> Dict[str, List[TaskEntry]]:
    by_file: Dict[str, List[TaskEntry]] = defaultdict(list)
    for path in audit_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("results", []):
            by_file[str(row["file"])].append(
                TaskEntry(
                    task_id=str(row["task_id"]),
                    file=str(row["file"]),
                    module=str(row["module"]),
                    function=str(row["function"]),
                    status=str(row["status"]),
                )
            )
    return by_file


def should_keep_file(entries: List[TaskEntry], min_pass_ratio: float) -> bool:
    counts = Counter(entry.status for entry in entries)
    ok = counts.get("passed", 0) + counts.get("skipped", 0)
    return bool(entries) and (ok / len(entries)) >= min_pass_ratio


def selected_tasks(entries: List[TaskEntry]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for entry in entries:
        if entry.status in {"passed", "skipped"}:
            grouped[entry.module].append(entry.function)
    return {module: sorted(set(funcs)) for module, funcs in grouped.items()}


def build_rewritten_source(source_path: Path, keep_by_module: Dict[str, List[str]]) -> str:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    lines = source.splitlines()
    used_classes: set[str] = set()
    top_level_functions: set[str] = set()
    blocks: List[str] = []

    def node_start(node: ast.AST) -> int:
        decorators = getattr(node, "decorator_list", None) or []
        if decorators:
            return min(dec.lineno for dec in decorators)
        return node.lineno  # type: ignore[attr-defined]

    def node_text(node: ast.AST) -> str:
        start = node_start(node)
        end = node.end_lineno or node.lineno  # type: ignore[attr-defined]
        return "\n".join(lines[start - 1:end]).rstrip()

    for module_name, functions in keep_by_module.items():
        if module_name.startswith("class "):
            used_classes.add(module_name.split("class ", 1)[1].strip())
        else:
            top_level_functions.update(functions)

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            blocks.append(node_text(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            blocks.append(node_text(node))
        elif isinstance(node, ast.Expr):
            # Preserve module docstrings / simple constants.
            blocks.append(node_text(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in top_level_functions:
                blocks.append(node_text(node))
        elif isinstance(node, ast.ClassDef):
            if node.name in used_classes:
                class_keep = set(keep_by_module.get(f"class {node.name}", []))
                selected_children = [
                    child
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in class_keep
                ]
                if selected_children:
                    class_lines = lines[node_start(node) - 1: node.lineno]
                    for child in node.body:
                        if isinstance(child, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Pass)):
                            class_lines.extend(lines[child.lineno - 1: (child.end_lineno or child.lineno)])
                    for child in selected_children:
                        class_lines.extend(lines[node_start(child) - 1: (child.end_lineno or child.lineno)])
                    blocks.append("\n".join(class_lines).rstrip())

    return "\n\n".join(block for block in blocks if block.strip()).rstrip() + "\n"


def write_suite(
    *,
    golden_repo: Path,
    audit_entries: Dict[str, List[TaskEntry]],
    output_root: Path,
    min_pass_ratio: float,
) -> Dict[str, object]:
    rewritten_root = output_root / "rewritten_tests"
    rewritten_root.mkdir(parents=True, exist_ok=True)

    selected_files: List[Dict[str, object]] = []
    total_selected_tasks = 0
    for rel_path, entries in sorted(audit_entries.items()):
        if not should_keep_file(entries, min_pass_ratio):
            continue
        keep_by_module = selected_tasks(entries)
        if not keep_by_module:
            continue
        source_path = golden_repo / rel_path
        target_path = rewritten_root / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        rewritten = build_rewritten_source(source_path, keep_by_module)
        target_path.write_text(rewritten, encoding="utf-8")
        task_count = sum(len(v) for v in keep_by_module.values())
        total_selected_tasks += task_count
        selected_files.append(
            {
                "file": rel_path,
                "task_count": task_count,
                "modules": keep_by_module,
                "source": str(source_path),
                "target": str(target_path),
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "golden_repo_root": str(golden_repo.resolve()),
        "rewritten_tests_root": str(rewritten_root.resolve()),
        "min_pass_ratio": min_pass_ratio,
        "selected_file_count": len(selected_files),
        "selected_task_count": total_selected_tasks,
        "selected_files": selected_files,
    }
    manifest_path = output_root / "rewritten_tests_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    audit_entries = load_audit_entries(args.audit_json)
    manifest = write_suite(
        golden_repo=args.golden_repo.resolve(),
        audit_entries=audit_entries,
        output_root=args.output_root.resolve(),
        min_pass_ratio=float(args.min_pass_ratio),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
