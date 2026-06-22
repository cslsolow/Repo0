#!/usr/bin/env python3
"""Batch repair invalid generated Python files using FixAgent."""

from __future__ import annotations

import argparse
import json
import py_compile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.coding.fix_agent import FixAgent


def _compile_error(path: Path) -> str:
    try:
        py_compile.compile(str(path), doraise=True)
        return ""
    except Exception as exc:
        return str(exc)


def _collect_failures(root: Path) -> List[Path]:
    failures: List[Path] = []
    for path in sorted(root.rglob("*.py")):
        if _compile_error(path):
            failures.append(path)
    return failures


def repair_tree(root: Path, *, in_place: bool, max_rounds: int) -> Dict[str, Any]:
    agent = FixAgent(max_rounds=max_rounds)
    failures = _collect_failures(root)
    report: Dict[str, Any] = {
        "root": str(root),
        "failed_before": len(failures),
        "failed_after": 0,
        "repaired": [],
        "unrepaired": [],
    }
    for path in failures:
        original = path.read_text()
        before = _compile_error(path)
        result = agent.fix_python_content(original)
        fixed = str(result.get("fixed_content", original))
        after = before
        repaired = False
        if result.get("fixed"):
            try:
                py_compile.compile(str(path), doraise=True)
            except Exception:
                pass
            # Validate candidate before writing.
            try:
                compile(fixed, str(path), "exec")
                after = ""
                repaired = True
                if in_place:
                    path.write_text(fixed)
            except Exception as exc:
                after = str(exc)
        entry = {
            "path": str(path),
            "error_before": before,
            "error_after": after,
            "rounds": result.get("rounds", 0),
        }
        if repaired:
            report["repaired"].append(entry)
        else:
            report["unrepaired"].append(entry)
    report["failed_after"] = len(report["unrepaired"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root directory containing generated Python files")
    parser.add_argument("--report", help="Optional JSON report path")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="Analyze and repair in memory only")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report = repair_tree(root, in_place=not args.dry_run, max_rounds=args.max_rounds)
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed_after"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
