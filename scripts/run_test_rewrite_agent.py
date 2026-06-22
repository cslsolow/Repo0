#!/usr/bin/env python3
"""CLI runner for TestRewriteAgent."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import TestRewriteAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite tests for generated repo and evaluate pass rate")
    parser.add_argument("--original-repo", type=Path, required=True, help="Path to original repository")
    parser.add_argument("--generated-repo", type=Path, required=True, help="Path to generated repository")
    parser.add_argument(
        "--original-tests-root",
        type=Path,
        default=None,
        help="Optional original tests root (default: <original-repo>/tests or original repo)",
    )
    parser.add_argument(
        "--rewritten-tests-root",
        type=Path,
        default=None,
        help="Optional rewritten tests output path (default: <output-dir>/rewritten_tests)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Working output directory for artifacts",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Optional JSON file path to persist evaluation result",
    )
    parser.add_argument("--base-url", type=str, default="", help="LLM base URL")
    parser.add_argument("--api-key", type=str, default="", help="LLM API key")
    parser.add_argument("--reasoning-effort", type=str, default="", help="Optional reasoning_effort to pass through to the LLM API.")
    parser.add_argument("--model", type=str, default="", help="LLM model")
    parser.add_argument(
        "--max-fix-rounds",
        type=int,
        default=2,
        help="Maximum iterative fix rounds based on pytest failures (default: 2, set 0 to disable).",
    )
    parser.add_argument(
        "--strict-api-mapping",
        action="store_true",
        help="Fail fast when import-to-API mapping cannot be resolved before LLM rewrite.",
    )
    parser.add_argument(
        "--no-continue-on-collection-errors",
        action="store_true",
        help="Disable pytest --continue-on-collection-errors (enabled by default).",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip rewrite and only evaluate existing rewritten tests under rewritten-tests-root.",
    )
    parser.add_argument(
        "--skip-rewrite-if-exists",
        action="store_true",
        help="Reuse existing rewritten tests if found; otherwise run rewrite workflow.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra args for pytest. Put after '--'.",
    )
    return parser.parse_args()


def normalize_pytest_args(raw: List[str]) -> List[str]:
    if not raw:
        return []
    if raw[0] == "--":
        return raw[1:]
    return raw


def ensure_collection_continuation(pytest_args: List[str], enabled: bool) -> List[str]:
    if not enabled:
        return pytest_args
    if "--continue-on-collection-errors" in pytest_args:
        return pytest_args
    return ["--continue-on-collection-errors", *pytest_args]


def write_rewrite_manifest(
    *,
    output_dir: Path,
    original_repo: Path,
    generated_repo: Path,
    tests_root: Path,
    rewritten_root: Path,
    result_json: Path,
    pytest_args: List[str],
    llm_config: Dict[str, Any],
    max_fix_rounds: int,
    strict_api_mapping: bool,
    evaluate_only: bool,
    skip_rewrite_if_exists: bool,
) -> Path:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_repo_root": str(original_repo),
        "generated_repo_root": str(generated_repo),
        "original_tests_root": str(tests_root),
        "rewritten_tests_root": str(rewritten_root),
        "result_json": str(result_json),
        "pytest_args": list(pytest_args),
        "llm_config": {
            "base_url": llm_config.get("base_url", ""),
            "model": llm_config.get("model", ""),
            "reasoning_effort": llm_config.get("reasoning_effort", ""),
            "api_key_present": bool(llm_config.get("api_key")),
        },
        "max_fix_rounds": int(max_fix_rounds),
        "strict_api_mapping": bool(strict_api_mapping),
        "evaluate_only": bool(evaluate_only),
        "skip_rewrite_if_exists": bool(skip_rewrite_if_exists),
    }
    manifest_path = output_dir / "test_rewrite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _summarize_only_existing_rewrites(
    agent: TestRewriteAgent,
    original_repo: Path,
    generated_repo: Path,
    tests_root: Path,
    rewritten_root: Path,
    pytest_args: List[str],
    strict_api_mapping: bool,
    max_fix_rounds: int,
    mode: str,
) -> Dict[str, Any]:
    original_test_files = list(agent._discover_test_files(tests_root))
    estimated_original_test_cases = agent._count_test_cases_from_files(original_test_files)

    rewritten_test_files = list(agent._discover_test_files(rewritten_root)) if rewritten_root.exists() else []
    if rewritten_test_files:
        run_result = agent.run_pytest(
            repo_root=generated_repo,
            tests_path=rewritten_root,
            extra_args=pytest_args,
        )
    else:
        run_result = {
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "summary": {
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "xfailed": 0,
                "xpassed": 0,
                "deselected": 0,
                "total": 0,
            },
            "test_cases": [],
            "skipped_reason": "no_existing_rewritten_tests",
        }

    summary = run_result.get("summary", {})
    total = int(summary.get("total", 0))
    passed = int(summary.get("passed", 0))
    executed_pass_rate = (passed / total) if total > 0 else 0.0
    rewrite_success_rate = (
        (len(rewritten_test_files) / len(original_test_files)) if original_test_files else 0.0
    )
    all_tests_pass_rate = (
        min(1.0, passed / estimated_original_test_cases) if estimated_original_test_cases > 0 else 0.0
    )

    return {
        "original_repo_root": str(original_repo),
        "generated_repo_root": str(generated_repo),
        "original_tests_root": str(tests_root),
        "rewritten_tests_root": str(rewritten_root),
        "rewrite_mode": mode,
        "rewrite_skipped": True,
        "test_files_count": len(original_test_files),
        "estimated_original_test_cases": int(estimated_original_test_cases),
        "remaining_test_files_count": len(rewritten_test_files),
        "rewrite_success_count": len(rewritten_test_files),
        "rewrite_failure_count": 0,
        "rewrite_success_rate": round(rewrite_success_rate, 4),
        "rewrite_results": [],
        "failed_list": [],
        "pytest": run_result,
        "executed_test_cases": int(total),
        "passed_test_cases": int(passed),
        "pass_rate": round(executed_pass_rate, 4),
        "remaining_pass_rate": round(executed_pass_rate, 4),
        "all_tests_pass_rate": round(all_tests_pass_rate, 4),
        "max_fix_rounds": int(max_fix_rounds),
        "strict_api_mapping": bool(strict_api_mapping),
        "adaptation_history": [],
        "adaptation_rounds_run": 0,
    }


def main() -> None:
    args = parse_args()

    api_config = {}
    if args.api_key:
        api_config = {
            "base_url": args.base_url,
            "api_key": args.api_key,
            "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    agent = TestRewriteAgent(api_config=api_config, output_dir=str(args.output_dir))
    original_repo = args.original_repo.resolve()
    generated_repo = args.generated_repo.resolve()
    tests_root = (
        args.original_tests_root.resolve()
        if args.original_tests_root
        else agent._default_tests_root(original_repo)
    )
    rewritten_root = (
        args.rewritten_tests_root.resolve()
        if args.rewritten_tests_root
        else args.output_dir.resolve() / "rewritten_tests"
    )
    pytest_args = ensure_collection_continuation(
        normalize_pytest_args(args.pytest_args),
        enabled=not args.no_continue_on_collection_errors,
    )

    existing_rewritten_tests = (
        list(agent._discover_test_files(rewritten_root))
        if rewritten_root.exists()
        else []
    )

    if args.evaluate_only:
        result = _summarize_only_existing_rewrites(
            agent=agent,
            original_repo=original_repo,
            generated_repo=generated_repo,
            tests_root=tests_root,
            rewritten_root=rewritten_root,
            pytest_args=pytest_args,
            strict_api_mapping=bool(args.strict_api_mapping),
            max_fix_rounds=max(0, int(args.max_fix_rounds)),
            mode="evaluate_only",
        )
    elif args.skip_rewrite_if_exists and existing_rewritten_tests:
        result = _summarize_only_existing_rewrites(
            agent=agent,
            original_repo=original_repo,
            generated_repo=generated_repo,
            tests_root=tests_root,
            rewritten_root=rewritten_root,
            pytest_args=pytest_args,
            strict_api_mapping=bool(args.strict_api_mapping),
            max_fix_rounds=max(0, int(args.max_fix_rounds)),
            mode="reuse_existing_rewrites",
        )
    else:
        result = agent.rewrite_tests_and_evaluate(
            original_repo_root=original_repo,
            generated_repo_root=generated_repo,
            original_tests_root=tests_root,
            rewritten_tests_root=rewritten_root,
            pytest_args=pytest_args,
            max_fix_rounds=max(0, int(args.max_fix_rounds)),
            strict_api_mapping=bool(args.strict_api_mapping),
        )
        result["rewrite_mode"] = "rewrite_and_evaluate"
        result["rewrite_skipped"] = False

    result_json = args.result_json or (args.output_dir / "test_rewrite_result.json")
    manifest_path = write_rewrite_manifest(
        output_dir=args.output_dir.resolve(),
        original_repo=original_repo,
        generated_repo=generated_repo,
        tests_root=tests_root,
        rewritten_root=rewritten_root,
        result_json=result_json.resolve(),
        pytest_args=pytest_args,
        llm_config=api_config,
        max_fix_rounds=max(0, int(args.max_fix_rounds)),
        strict_api_mapping=bool(args.strict_api_mapping),
        evaluate_only=bool(args.evaluate_only),
        skip_rewrite_if_exists=bool(args.skip_rewrite_if_exists),
    )
    result["rewrite_manifest"] = str(manifest_path)
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = result.get("pytest", {}).get("summary", {})
    print("=== Test Rewrite Result ===")
    print(f"Original repo: {result.get('original_repo_root')}")
    print(f"Generated repo: {result.get('generated_repo_root')}")
    print(f"Rewritten tests: {result.get('rewritten_tests_root')}")
    print(f"Test files: {result.get('test_files_count')}")
    print(f"Rewrite success: {result.get('rewrite_success_count')} / {result.get('test_files_count')}")
    print(f"Rewrite failures: {result.get('rewrite_failure_count')}")
    print(f"Rewrite success rate: {result.get('rewrite_success_rate')}")
    print(f"Rewrite mode: {result.get('rewrite_mode')}")
    print(f"Rewrite skipped: {result.get('rewrite_skipped')}")
    print(f"Continue on collection errors: {not args.no_continue_on_collection_errors}")
    print(f"Fix rounds: {result.get('adaptation_rounds_run', 0)} / {result.get('max_fix_rounds', 0)}")
    print(f"Strict API mapping: {result.get('strict_api_mapping')}")
    print(f"Estimated original test cases: {result.get('estimated_original_test_cases')}")
    print(f"Executed test cases: {result.get('executed_test_cases')}")
    print(f"Passed test cases: {result.get('passed_test_cases')}")
    print(f"All-tests pass rate: {result.get('all_tests_pass_rate')}")
    print(f"Remaining pass rate: {result.get('remaining_pass_rate')}")
    print(f"Pass rate: {result.get('pass_rate')}")
    print(f"Pytest return code: {result.get('pytest', {}).get('returncode')}")
    print(
        "Summary: passed={passed}, failed={failed}, errors={errors}, skipped={skipped}, total={total}".format(
            passed=summary.get("passed", 0),
            failed=summary.get("failed", 0),
            errors=summary.get("errors", 0),
            skipped=summary.get("skipped", 0),
            total=summary.get("total", 0),
        )
    )
    print(f"Result JSON: {result_json}")


if __name__ == "__main__":
    main()
