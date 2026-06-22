#!/usr/bin/env python3
"""Standalone CLI runner for LintFixAgent with optional multiprocessing."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import LintFixAgent  # noqa: E402


def _mask_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lint/fix for generated Python files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory to scan for .py files.")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output JSON report path (default: <input-dir>/lint_fix_report.json).",
    )
    parser.add_argument("--base-url", type=str, default="", help="LLM base URL.")
    parser.add_argument("--api-key", type=str, default="", help="LLM API key.")
    parser.add_argument("--model", type=str, default="", help="LLM model name.")
    parser.add_argument("--reasoning-effort", type=str, default="", help="Optional reasoning_effort to pass through to the LLM API.")
    parser.add_argument("--workers", type=int, default=1, help="Process count for parallel fixing.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Files per process chunk (default: auto by worker count).",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of Python files (0 means no limit).")
    parser.add_argument("--disable-llm-fix", action="store_true", help="Disable LLM-based fixing.")
    parser.add_argument("--lint-static-rounds", type=int, default=2, help="Max rounds for static syntax fix.")
    parser.add_argument("--lint-llm-rounds", type=int, default=5, help="Max LLM repair rounds per file.")
    parser.add_argument(
        "--lint-llm-max-files",
        type=int,
        default=0,
        help="Max files eligible for LLM fix (0 means unlimited).",
    )
    parser.add_argument(
        "--lint-llm-max-chars",
        type=int,
        default=120000,
        help="Skip LLM fix for files larger than this many characters.",
    )
    parser.add_argument(
        "--lint-fix-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for one LLM repair call.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def collect_python_files(input_dir: Path, max_files: int = 0) -> List[Path]:
    files = sorted(path.resolve() for path in input_dir.rglob("*.py") if path.is_file())
    if max_files > 0:
        files = files[:max_files]
    return files


def chunk_list(items: List[Path], chunk_size: int) -> List[List[Path]]:
    if chunk_size <= 0:
        return [items]
    chunks: List[List[Path]] = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i : i + chunk_size])
    return chunks


def merge_reports(reports: List[Dict[str, Any]], generated_root: Path) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "generated_root": str(generated_root),
        "ruff_available": any(r.get("ruff_available", False) for r in reports),
        "llm_enabled": any(r.get("llm_enabled", False) for r in reports),
        "checked_files": sum(int(r.get("checked_files", 0)) for r in reports),
        "files_with_issues": sum(int(r.get("files_with_issues", 0)) for r in reports),
        "fixed_by_static": sum(int(r.get("fixed_by_static", 0)) for r in reports),
        "fixed_by_llm": sum(int(r.get("fixed_by_llm", 0)) for r in reports),
        "unresolved": sum(int(r.get("unresolved", 0)) for r in reports),
        "files": [],
    }
    files: List[Dict[str, Any]] = []
    for report in reports:
        for item in report.get("files", []):
            if isinstance(item, dict):
                files.append(item)
    files.sort(key=lambda x: str(x.get("file", "")))
    merged["files"] = files
    return merged


def _run_chunk(
    file_paths: List[str],
    api_config: Dict[str, Any],
    output_dir: str,
    generated_root: str,
) -> Dict[str, Any]:
    agent = LintFixAgent(api_config=api_config, output_dir=output_dir)
    return agent.run_on_files(file_paths, generated_root=generated_root)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    report_path = args.report.resolve() if args.report else (input_dir / "lint_fix_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    llm_enabled = not args.disable_llm_fix and bool(api_key)

    workers = max(1, int(args.workers))
    llm_max_files = max(0, int(args.lint_llm_max_files))
    if workers > 1 and llm_max_files > 0:
        llm_max_files = max(1, math.ceil(llm_max_files / workers))

    api_config: Dict[str, Any] = {
        "base_url": args.base_url,
        "api_key": api_key,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "enable_lint_llm_fix": llm_enabled,
        "lint_static_rounds": int(args.lint_static_rounds),
        "lint_llm_rounds": int(args.lint_llm_rounds),
        "lint_llm_max_files": llm_max_files,
        "lint_llm_max_chars": int(args.lint_llm_max_chars),
        "lint_fix_timeout_seconds": float(args.lint_fix_timeout_seconds),
    }
    logging.info(
        "Run config: input_dir=%s report=%s workers=%d chunk_size=%d max_files=%d llm_enabled=%s",
        input_dir,
        report_path,
        workers,
        int(args.chunk_size),
        int(args.max_files),
        llm_enabled,
    )

    print("=== Lint Fix API Config ===")
    print(f"base_url: {api_config.get('base_url') or '<empty>'}")
    print(f"model: {api_config.get('model') or '<empty>'}")
    print(f"reasoning_effort: {api_config.get('reasoning_effort') or '<empty>'}")
    print(f"api_key: {_mask_api_key(api_key)}")
    print(f"timeout_seconds: {api_config.get('lint_fix_timeout_seconds')}")

    files = collect_python_files(input_dir, max_files=max(0, int(args.max_files)))
    logging.info("Discovered %d Python files", len(files))
    if not files:
        empty_report = merge_reports([], input_dir)
        report_path.write_text(json.dumps(empty_report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"No python files found under: {input_dir}")
        print(f"Report: {report_path}")
        return

    if workers <= 1 or len(files) == 1:
        logging.info("Using single-process mode")
        agent = LintFixAgent(api_config=api_config, output_dir=str(report_path.parent))
        merged_report = agent.run_on_files(files, generated_root=input_dir)
    else:
        chunk_size = int(args.chunk_size)
        if chunk_size <= 0:
            chunk_size = max(1, math.ceil(len(files) / workers))
        chunks = chunk_list(files, chunk_size)
        logging.info(
            "Using multiprocess mode: workers=%d chunk_size=%d chunks=%d",
            workers,
            chunk_size,
            len(chunks),
        )
        reports: List[Dict[str, Any]] = []
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_chunk = {}
                for idx, chunk in enumerate(chunks):
                    worker_output_dir = report_path.parent / f".lint_fix_worker_{idx}"
                    worker_output_dir.mkdir(parents=True, exist_ok=True)
                    logging.info(
                        "Submitting chunk #%d with %d files (worker_output_dir=%s)",
                        idx,
                        len(chunk),
                        worker_output_dir,
                    )
                    future = executor.submit(
                        _run_chunk,
                        [str(path) for path in chunk],
                        api_config,
                        str(worker_output_dir),
                        str(input_dir),
                    )
                    future_to_chunk[future] = idx

                for future in as_completed(future_to_chunk):
                    idx = future_to_chunk[future]
                    result = future.result()
                    reports.append(result)
                    logging.info(
                        "Completed chunk #%d: checked=%s issues=%s unresolved=%s",
                        idx,
                        result.get("checked_files", 0),
                        result.get("files_with_issues", 0),
                        result.get("unresolved", 0),
                    )
        except (PermissionError, OSError) as exc:
            print(f"Multiprocessing unavailable in current environment ({exc}); fallback to single process.")
            logging.warning("Multiprocessing unavailable (%s), fallback to single process", exc)
            agent = LintFixAgent(api_config=api_config, output_dir=str(report_path.parent))
            merged_report = agent.run_on_files(files, generated_root=input_dir)
            report_path.write_text(json.dumps(merged_report, indent=2, ensure_ascii=False), encoding="utf-8")
            print("=== Lint Fix Summary ===")
            print(f"Input dir: {input_dir}")
            print(f"Report: {report_path}")
            print("Workers: 1 (fallback)")
            print(f"LLM enabled: {llm_enabled}")
            print(
                "checked={checked}, issues={issues}, static_fixed={static_fixed}, llm_fixed={llm_fixed}, unresolved={unresolved}".format(
                    checked=merged_report.get("checked_files", 0),
                    issues=merged_report.get("files_with_issues", 0),
                    static_fixed=merged_report.get("fixed_by_static", 0),
                    llm_fixed=merged_report.get("fixed_by_llm", 0),
                    unresolved=merged_report.get("unresolved", 0),
                )
            )
            return

        merged_report = merge_reports(reports, input_dir)

    report_path.write_text(json.dumps(merged_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Report written: %s", report_path)

    print("=== Lint Fix Summary ===")
    print(f"Input dir: {input_dir}")
    print(f"Report: {report_path}")
    print(f"Workers: {workers}")
    print(f"LLM enabled: {llm_enabled}")
    print(
        "checked={checked}, issues={issues}, static_fixed={static_fixed}, llm_fixed={llm_fixed}, unresolved={unresolved}".format(
            checked=merged_report.get("checked_files", 0),
            issues=merged_report.get("files_with_issues", 0),
            static_fixed=merged_report.get("fixed_by_static", 0),
            llm_fixed=merged_report.get("fixed_by_llm", 0),
            unresolved=merged_report.get("unresolved", 0),
        )
    )


if __name__ == "__main__":
    main()
