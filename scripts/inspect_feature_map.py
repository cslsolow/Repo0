#!/usr/bin/env python3
"""Inspect feature mappings for interfaces in a Python file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.localization_pipeline_agent import view_file_interface_feature_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect feature mappings for a file")
    parser.add_argument("--repo", type=Path, required=True, help="Repository root")
    parser.add_argument("--file", type=str, required=True, help="Python file path (relative or absolute)")
    parser.add_argument(
        "--interface",
        type=str,
        default="",
        help="Interface name to filter (ClassName, ClassName.method, or function)",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, file_path: str) -> str:
    path = Path(file_path)
    if path.is_absolute():
        return str(path)
    return str(repo_root / file_path)


def main() -> None:
    args = parse_args()
    file_path = resolve_path(args.repo, args.file)
    results = view_file_interface_feature_map(file_path)
    if args.interface:
        results = [item for item in results if item.get("interface") == args.interface]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
