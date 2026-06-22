#!/usr/bin/env python3
"""Count Python compile errors under a directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count compile errors for Python files in a directory.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory to scan recursively.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON output file path, or an existing directory.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return exit code 1 when compile errors exist.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=200,
        help="Max number of error lines to print.",
    )
    return parser.parse_args()


def resolve_json_output_path(value: Path) -> Path:
    """
    Normalize JSON output target.

    - If `value` is an existing directory, write `compile_errors.json` inside it.
    - Otherwise treat `value` as the output file path.
    """
    candidate = value.resolve()
    if candidate.exists() and candidate.is_dir():
        return candidate / "compile_errors.json"
    return candidate


def check_python_file(path: Path) -> Dict[str, Any] | None:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "file": str(path),
            "line": 0,
            "column": 0,
            "error": f"ReadError: {exc}",
        }

    try:
        compile(content, str(path), "exec")
        return None
    except SyntaxError as exc:
        return {
            "file": str(path),
            "line": int(exc.lineno or 0),
            "column": int(exc.offset or 0),
            "error": str(exc.msg or exc),
        }
    except Exception as exc:
        return {
            "file": str(path),
            "line": 0,
            "column": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    py_files: List[Path] = sorted(path for path in input_dir.rglob("*.py") if path.is_file())
    errors: List[Dict[str, Any]] = []
    for path in py_files:
        err = check_python_file(path)
        if err is not None:
            errors.append(err)

    summary = {
        "input_dir": str(input_dir),
        "python_files": len(py_files),
        "compile_error_files": len(errors),
        "errors": errors,
    }

    print("=== Compile Error Summary ===")
    print(f"Input dir: {input_dir}")
    print(f"Python files scanned: {len(py_files)}")
    print(f"Files with compile errors: {len(errors)}")

    if errors:
        print("\nError details:")
        for idx, err in enumerate(errors[: max(0, int(args.max_print))], start=1):
            print(
                f"{idx}. {err['file']}:{err['line']}:{err['column']} - {err['error']}"
            )
        if len(errors) > args.max_print:
            print(f"... {len(errors) - args.max_print} more errors omitted")

    if args.json_output:
        json_path = resolve_json_output_path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON report: {json_path}")

    if args.fail_on_error and errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
