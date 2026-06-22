#!/usr/bin/env python3
"""Repair generated Python trees whose package paths contain hyphenated slugs."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_ROOT_PATH = ROOT / "agents" / "package_root.py"
_PACKAGE_ROOT_SPEC = importlib.util.spec_from_file_location("_repo0_package_root", _PACKAGE_ROOT_PATH)
if _PACKAGE_ROOT_SPEC is None or _PACKAGE_ROOT_SPEC.loader is None:
    raise RuntimeError(f"Cannot load package root helper from {_PACKAGE_ROOT_PATH}")
_PACKAGE_ROOT_MODULE = importlib.util.module_from_spec(_PACKAGE_ROOT_SPEC)
_PACKAGE_ROOT_SPEC.loader.exec_module(_PACKAGE_ROOT_MODULE)
normalize_python_package_root = _PACKAGE_ROOT_MODULE.normalize_python_package_root


SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _normalize_module_path(module_path: str) -> str:
    parts = [part for part in str(module_path or "").split(".") if part]
    return ".".join(normalize_python_package_root(part, default="") for part in parts)


def _target_name(path: Path) -> str:
    if path.is_file() and path.suffix == ".py":
        stem = normalize_python_package_root(path.stem, default="")
        return f"{stem}.py" if stem else path.name
    if path.is_dir() and not path.name.endswith(".egg-info"):
        return normalize_python_package_root(path.name, default="")
    return path.name


def _python_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def rewrite_hyphenated_imports(text: str) -> str:
    """Rewrite syntactically invalid absolute imports like ``from a-b.x import``."""
    if "-" not in text:
        return text

    text = re.sub(
        r"(?m)^(\s*from\s+)([A-Za-z0-9_.-]*-[A-Za-z0-9_.-]*)(\s+import\s+)",
        lambda m: f"{m.group(1)}{_normalize_module_path(m.group(2))}{m.group(3)}",
        text,
    )
    text = re.sub(
        r"(?m)^(\s*import\s+)([A-Za-z0-9_.-]*-[A-Za-z0-9_.-]*)(\s*(?:#.*)?)$",
        lambda m: f"{m.group(1)}{_normalize_module_path(m.group(2))}{m.group(3)}",
        text,
    )
    return text


def _rename_path(path: Path, dry_run: bool) -> bool:
    target_name = _target_name(path)
    if not target_name or target_name == path.name:
        return False

    target = path.with_name(target_name)
    if target.exists():
        raise FileExistsError(f"Cannot rename {path} -> {target}: target exists")
    if not dry_run:
        path.rename(target)
    return True


def _rename_invalid_paths(root: Path, dry_run: bool) -> int:
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_dir() and path.name.endswith(".egg-info"):
            continue
        if path.is_dir() or (path.is_file() and path.suffix == ".py"):
            candidates.append(path)

    renamed = 0
    for path in sorted(candidates, key=lambda p: len(p.parts), reverse=True):
        if not path.exists():
            continue
        if _rename_path(path, dry_run):
            renamed += 1
    return renamed


def _rewrite_files(root: Path, dry_run: bool) -> int:
    rewritten = 0
    for path in _python_paths(root):
        original = path.read_text(encoding="utf-8", errors="ignore")
        fixed = rewrite_hyphenated_imports(original)
        if fixed == original:
            continue
        rewritten += 1
        if not dry_run:
            path.write_text(fixed, encoding="utf-8")
    return rewritten


def repair_tree(root: Path | str, *, dry_run: bool = False) -> Dict[str, int]:
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(root_path)

    renamed = _rename_invalid_paths(root_path, dry_run=dry_run)
    rewritten = _rewrite_files(root_path, dry_run=dry_run)
    return {"renamed": renamed, "rewritten": rewritten}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize hyphenated generated Python package roots and imports."
    )
    parser.add_argument("roots", nargs="+", type=Path, help="Generated code/workspace roots to repair.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = parser.parse_args()

    total = {"renamed": 0, "rewritten": 0}
    for root in args.roots:
        report = repair_tree(root, dry_run=bool(args.dry_run))
        total["renamed"] += report["renamed"]
        total["rewritten"] += report["rewritten"]
        print(f"{root}: renamed={report['renamed']} rewritten={report['rewritten']}")
    print(f"TOTAL: renamed={total['renamed']} rewritten={total['rewritten']}")


if __name__ == "__main__":
    main()
