#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print PR numbers for each repo folder under repos_all."
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=Path("repo_input"),
        help="Directory containing repo folders (default: commit0/repos_all).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing JSONL dataset files.",
    )
    parser.add_argument(
        "--only-with-prs",
        action="store_true",
        help="Only print repos that have PR numbers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: <data-dir>/prnumbers_all.json).",
    )
    return parser.parse_args()


def iter_jsonl_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.glob("*.jsonl")):
        if path.is_file():
            yield path


def load_prnumbers_by_repo(data_dir: Path) -> dict[str, list[int]]:
    repo_map: dict[str, list[int]] = {}
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
                repo = str(item.get("repo", "")).strip()
                if not repo:
                    continue
                pr_number = item.get("pull_number")
                if not isinstance(pr_number, int):
                    continue
                repo_map.setdefault(repo, []).append(pr_number)
    return repo_map


def main() -> None:
    args = parse_args()
    repos_dir = args.repos_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()

    if not repos_dir.exists():
        raise FileNotFoundError(f"Repos dir not found: {repos_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    repo_map = load_prnumbers_by_repo(data_dir)
    repos_all = {p.name.lower() for p in repos_dir.iterdir() if p.is_dir()}
    result: dict[str, list[int]] = {}

    for repo_name, prnumbers in sorted(repo_map.items()):
        if repo_name.split("/")[-1].lower() not in repos_all:
            continue
        if args.only_with_prs and not prnumbers:
            continue
        result[repo_name] = prnumbers

    if args.output is None:
        output_path = data_dir / "prnumbers_all.json"
    else:
        output_path = args.output.expanduser().resolve()

    output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=4, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(result)} repos to {output_path}")


if __name__ == "__main__":
    main()
