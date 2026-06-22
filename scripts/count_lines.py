#!/usr/bin/env python3
"""Count lines of code for files under a root directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count lines of code under a directory.")
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=[],
        help="File extension to include (e.g., --ext py). Can be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Path substring to exclude (can be repeated).",
    )
    parser.add_argument(
        "--no-hidden",
        action="store_true",
        help="Skip hidden files and directories.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file counts.",
    )
    return parser.parse_args()


def should_skip(path: Path, excludes: List[str], no_hidden: bool) -> bool:
    path_str = str(path)
    if no_hidden and any(part.startswith(".") for part in path.parts):
        return True
    return any(excl in path_str for excl in excludes)


def iter_files(root: Path, exts: Iterable[str], excludes: List[str], no_hidden: bool) -> Iterable[Path]:
    normalized_exts = {ext.lstrip(".") for ext in exts if ext}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path, excludes, no_hidden):
            continue
        if normalized_exts:
            if path.suffix.lstrip(".") not in normalized_exts:
                continue
        yield path


def count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except OSError:
        return 0


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()

    total = 0
    files_scanned = 0
    per_file = []

    for path in iter_files(root, args.ext, args.exclude, args.no_hidden):
        lines = count_lines(path)
        files_scanned += 1
        total += lines
        if args.show_files:
            per_file.append((lines, path))

    if args.show_files:
        for lines, path in sorted(per_file, reverse=True):
            print(f"{lines:8d}  {path}")

    print(f"Files: {files_scanned}")
    print(f"Total lines: {total}")


if __name__ == "__main__":
    main()
