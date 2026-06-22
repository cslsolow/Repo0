#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_jsonl_files(data_dir: Path) -> Iterable[Path]:
    if data_dir.is_file() and data_dir.suffix == ".jsonl":
        yield data_dir
        return
    for path in sorted(data_dir.glob("*.jsonl")):
        if path.is_file():
            yield path


def normalize_repo(repo_input: str) -> tuple[str | None, str]:
    repo_input = repo_input.strip()
    if "/" in repo_input:
        owner, name = repo_input.split("/", 1)
        return owner.lower(), name.lower()
    return None, repo_input.lower()


def matches_repo(item_repo: str, owner: str | None, name: str) -> bool:
    if not item_repo:
        return False
    item_repo = item_repo.strip()
    if "/" in item_repo:
        item_owner, item_name = item_repo.split("/", 1)
    else:
        item_owner, item_name = "", item_repo
    item_owner = item_owner.lower()
    item_name = item_name.lower()
    if owner is None:
        return item_name == name
    return item_owner == owner and item_name == name


def extract_prnumbers(data_dir: Path, repo_input: str) -> list[int]:
    owner, name = normalize_repo(repo_input)
    prnumbers: list[int] = []
    for jsonl_path in iter_jsonl_files(data_dir):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if matches_repo(str(item.get("repo", "")), owner, name):
                    value = item.get("pull_number")
                    if isinstance(value, int):
                        prnumbers.append(value)
    return prnumbers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pull numbers for a repo from a JSONL dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing *.jsonl files or a single JSONL file.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Repository name (owner/name or name).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: <data-dir>/prnumbers_<repo>.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for JSON file (default: <data-dir>).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data path not found: {data_dir}")

    prnumbers = extract_prnumbers(data_dir, args.repo)
    if args.output is not None and args.output_dir is not None:
        raise ValueError("Use only one of --output or --output-dir.")

    if args.output is None:
        safe_repo = args.repo.replace("/", "_")
        if args.output_dir is not None:
            default_base = args.output_dir.expanduser().resolve()
        else:
            default_base = data_dir if data_dir.is_dir() else data_dir.parent
        output_path = default_base / f"prnumbers_{safe_repo}.json"
    else:
        output_path = args.output.expanduser().resolve()

    if not prnumbers:
        print(f"No PR numbers found for {args.repo}; skipping output.")
        return

    output_path.write_text(json.dumps(prnumbers, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {len(prnumbers)} PR numbers for {args.repo} to {output_path}")


if __name__ == "__main__":
    main()
