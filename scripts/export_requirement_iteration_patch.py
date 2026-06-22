#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Iterable, List, Sequence

TEXT_EXTS = {
    ".py",
    ".pyi",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".sh",
}
DEFAULT_EXCLUDES = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".git",
    "iteration_input",
    "fail2pass",
    "logs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export requirement-iteration changes as a unified diff patch.")
    parser.add_argument("--manifest", type=Path, required=True, help="iteration_manifest.json")
    parser.add_argument("--baseline-dir", type=Path, default=None, help="Override baseline repo dir from manifest")
    parser.add_argument("--target-dir", type=Path, default=None, help="Override target repo dir from manifest")
    parser.add_argument("--output-patch", type=Path, required=True, help="Where to write unified diff patch")
    parser.add_argument(
        "--include-root",
        action="append",
        default=[],
        help="Relative subtree to diff. Repeatable. Default: whole copied repo excluding transient dirs.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=list(DEFAULT_EXCLUDES),
        help="Substring path filters to skip. Repeatable.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def should_skip(relpath: str, excludes: Sequence[str]) -> bool:
    parts = relpath.split("/")
    for token in excludes:
        if token in parts or token in relpath:
            return True
    return False


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTS:
        return True
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if b"\x00" in data:
        return False
    return True


def collect_files(root: Path, include_roots: Sequence[str], excludes: Sequence[str]) -> List[str]:
    files: List[str] = []
    search_roots: Iterable[Path]
    if include_roots:
        search_roots = [(root / item).resolve() for item in include_roots]
    else:
        search_roots = [root]
    for base in search_roots:
        if not base.exists():
            continue
        if base.is_file():
            rel = base.relative_to(root).as_posix()
            if not should_skip(rel, excludes) and is_text_file(base):
                files.append(rel)
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if should_skip(rel, excludes):
                continue
            if not is_text_file(path):
                continue
            files.append(rel)
    return sorted(set(files))


def read_lines(path: Path) -> List[str]:
    if not path.exists() or not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)


def make_patch(baseline: Path, target: Path, relpaths: Sequence[str]) -> str:
    chunks: List[str] = []
    for rel in relpaths:
        old_path = baseline / rel
        new_path = target / rel
        old_lines = read_lines(old_path)
        new_lines = read_lines(new_path)
        if old_lines == new_lines:
            continue
        fromfile = f"a/{rel}"
        tofile = f"b/{rel}"
        diff = difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=tofile, lineterm="")
        diff_lines = list(diff)
        if diff_lines:
            chunks.append("\n".join(diff_lines) + "\n")
    return "".join(chunks)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest.resolve())
    baseline = (args.baseline_dir or Path(manifest["baseline_repo_dir"])).resolve()
    target = (args.target_dir or Path(manifest["target_repo_dir"])).resolve()

    baseline_files = collect_files(baseline, args.include_root, args.exclude)
    target_files = collect_files(target, args.include_root, args.exclude)
    relpaths = sorted(set(baseline_files) | set(target_files))
    patch_text = make_patch(baseline, target, relpaths)

    args.output_patch.parent.mkdir(parents=True, exist_ok=True)
    args.output_patch.write_text(patch_text, encoding="utf-8")
    summary = {
        "baseline_dir": str(baseline),
        "target_dir": str(target),
        "output_patch": str(args.output_patch.resolve()),
        "included_roots": args.include_root,
        "candidate_files": len(relpaths),
        "patch_bytes": len(patch_text.encode("utf-8")),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
