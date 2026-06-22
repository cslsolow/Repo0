from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FAIL_PATTERNS = [
    r"Skeleton\+TDD codegen failed for component '([^']+)'",
    r"FixAgent could not repair syntax for component '([^']+)'",
    r"LLM code generation failed for component '([^']+)'",
    r"Component postcheck failed for '([^']+)'",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _collect_failed_components(log_text: str, output_dir: Path) -> set[str]:
    failed: set[str] = set()
    for pattern in FAIL_PATTERNS:
        failed.update(re.findall(pattern, log_text))

    realization_path = output_dir / "component_realization_report.json"
    if realization_path.exists():
        data = _load_json(realization_path)
        for item in data.get("components", []):
            if not item.get("passed", True):
                name = str(item.get("component", "")).strip()
                if name:
                    failed.add(name)

    import_report_path = output_dir / "component_import_postcheck_report.json"
    if import_report_path.exists():
        data = _load_json(import_report_path)
        for item in data.get("files", []):
            if not item.get("passed", True):
                name = str(item.get("component", "")).strip()
                if name:
                    failed.add(name)

    return failed


def _iter_entry_files(entry: dict[str, Any]) -> list[Path]:
    file_paths: list[Path] = []
    files = entry.get("files") or {}
    if isinstance(files, dict):
        for value in files.values():
            if value:
                file_paths.append(Path(str(value)))
    for value in entry.get("init_files") or []:
        if value:
            file_paths.append(Path(str(value)))
    return file_paths


def _prune_generated_files(output_dir: Path, failed: set[str]) -> tuple[list[str], int]:
    path = output_dir / "generated_files.json"
    if not path.exists():
        return [], 0
    entries = _load_json(path)
    if not isinstance(entries, list):
        return [], 0

    removed_files: list[str] = []
    kept_entries: list[dict[str, Any]] = []
    for entry in entries:
        component = str(entry.get("component", "")).strip()
        if component and component in failed:
            for file_path in _iter_entry_files(entry):
                removed_files.append(str(file_path))
            continue
        kept_entries.append(entry)

    _dump_json(path, kept_entries)
    return removed_files, len(entries) - len(kept_entries)


def _prune_realization_report(output_dir: Path, failed: set[str]) -> int:
    path = output_dir / "component_realization_report.json"
    if not path.exists():
        return 0
    data = _load_json(path)
    items = data.get("components", [])
    kept = [item for item in items if str(item.get("component", "")).strip() not in failed]
    data["components"] = kept
    data["total_components"] = len(kept)
    data["passed_components"] = sum(1 for item in kept if item.get("passed"))
    data["failed_components"] = sum(1 for item in kept if not item.get("passed"))
    _dump_json(path, data)
    return len(items) - len(kept)


def _prune_import_report(output_dir: Path, failed: set[str]) -> int:
    path = output_dir / "component_import_postcheck_report.json"
    if not path.exists():
        return 0
    data = _load_json(path)
    items = data.get("files", [])
    kept = [item for item in items if str(item.get("component", "")).strip() not in failed]
    data["files"] = kept
    data["total_files"] = len(kept)
    data["passed_files"] = sum(1 for item in kept if item.get("passed"))
    data["failed_files"] = sum(1 for item in kept if not item.get("passed"))
    _dump_json(path, data)
    return len(items) - len(kept)


def _prune_memory(output_dir: Path, failed: set[str]) -> int:
    path = output_dir / "memory.json"
    if not path.exists():
        return 0
    data = _load_json(path)
    implemented = data.get("implemented_components")
    if not isinstance(implemented, dict):
        return 0

    removed = 0
    for key in list(implemented.keys()):
        if "::" not in key:
            continue
        component = key.split("::", 1)[1].strip()
        if component in failed:
            implemented.pop(key, None)
            removed += 1
    data["implemented_components"] = implemented
    _dump_json(path, data)
    return removed


def _remove_paths(paths: list[str], repo_root: Path) -> list[str]:
    removed: list[str] = []
    for raw in sorted(set(paths)):
        path = Path(raw)
        try:
            resolved = path.resolve()
        except Exception:
            continue
        try:
            resolved.relative_to(repo_root.resolve())
        except Exception:
            continue
        if resolved.exists() and resolved.is_file():
            resolved.unlink()
            removed.append(str(resolved))
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    log_path = Path(args.log_file).resolve()
    log_text = log_path.read_text(encoding="utf-8")
    failed = _collect_failed_components(log_text, output_dir)

    removed_file_candidates, removed_entries = _prune_generated_files(output_dir, failed)
    removed_realization = _prune_realization_report(output_dir, failed)
    removed_import = _prune_import_report(output_dir, failed)
    removed_memory = _prune_memory(output_dir, failed)
    removed_files = _remove_paths(removed_file_candidates, output_dir)

    summary = {
        "failed_components": sorted(failed),
        "failed_component_count": len(failed),
        "generated_file_entries_removed": removed_entries,
        "realization_entries_removed": removed_realization,
        "import_report_entries_removed": removed_import,
        "memory_entries_removed": removed_memory,
        "deleted_files_count": len(removed_files),
        "deleted_files": removed_files,
    }
    _dump_json(output_dir / "failed_component_reset_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
