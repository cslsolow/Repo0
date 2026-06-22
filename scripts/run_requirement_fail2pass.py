#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.infra.llm_client import LLMClient


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


EVAL_MOD = _load_module(ROOT / "scripts" / "eval_requirement_iteration.py", "eval_requirement_iteration_mod")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fail2pass tests from original PRs and rewrite them to the iteration repo.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-repo", type=Path, required=True, help="Generated iteration repo root or generated_code root.")
    parser.add_argument("--original-repo", type=Path, required=True, help="Original upstream repository checkout.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Working directory for extracted/synthesized tests and rewrite artifacts.")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--reasoning-effort", type=str, default="", help="Optional reasoning_effort to pass through to the LLM API.")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-fix-rounds", type=int, default=2)
    parser.add_argument("--strict-api-mapping", action="store_true")
    parser.add_argument("--copy-existing-pr-tests", action="store_true", help="Copy original repo test files referenced by PR diffs.")
    parser.add_argument("--synthesize-missing-tests", action="store_true", help="Synthesize original fail2pass tests when PR diff has no test file.")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _resolve_generated_repo_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "generated_code").exists():
        return (path / "generated_code").resolve()
    return path


def _normalize_test_relpath(relpath: str) -> str:
    rel = relpath.strip().lstrip("./")
    if rel.startswith("a/") or rel.startswith("b/"):
        rel = rel[2:]
    return rel


def copy_pr_tests_from_original(
    items: List[Dict[str, Any]],
    *,
    original_repo: Path,
    extracted_tests_root: Path,
) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    for item in items:
        copied_paths: List[str] = []
        missing_paths: List[str] = []
        for relpath in item.get("changed_test_files", []):
            normalized = _normalize_test_relpath(str(relpath))
            source_path = (original_repo / normalized).resolve()
            if not source_path.exists() or not source_path.is_file():
                missing_paths.append(normalized)
                continue
            target_path = (extracted_tests_root / normalized).resolve()
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied_paths.append(str(target_path))
        copied.append(
            {
                "repository": item.get("repository"),
                "pr_number": item.get("pr_number"),
                "copied_test_files": copied_paths,
                "missing_test_files": missing_paths,
                "copied_count": len(copied_paths),
            }
        )
    return copied


def synthesize_original_fail2pass_tests(
    items: List[Dict[str, Any]],
    *,
    original_repo: Path,
    output_dir: Path,
    model: str,
    base_url: str,
    api_key: str,
    reasoning_effort: str,
) -> Dict[str, Any]:
    llm_client = LLMClient(
        {"api_key": api_key, "base_url": base_url, "model": model, "reasoning_effort": reasoning_effort},
        str(output_dir),
        agent_name="requirement_fail2pass_original_synth",
    )
    results: List[Dict[str, Any]] = []
    for item in items:
        if not item.get("needs_llm_test_synthesis"):
            continue
        pr_number = item.get("pr_number")
        repo = item.get("repository", "")
        description = item.get("description_with_link_context") or item.get("description") or ""
        prompt = f"""
You are generating an original-repository regression pytest file from a PR.

Repository: {repo}
PR number: {pr_number}
Original repository root: {original_repo}
Title: {item.get("title", "")}
Description:
{description}

Diff:
{item.get("diff", "")}

Return strict JSON:
{{
  "test_file": "tests/test_<short_name>.py",
  "test_purpose": "...",
  "test_code": "full pytest code targeting the original repository"
}}

Constraints:
- This test is for the original repository semantics, not the generated repository.
- Use pytest.
- Generate one focused fail-to-pass regression.
- No placeholders.
"""
        try:
            payload = llm_client.call_json(
                messages=[
                    {
                        "role": "system",
                        "content": "You generate precise original-repository regression tests and return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=32768,
                operation_name="requirement_fail2pass_original_test_synthesis",
            )
        except Exception as exc:
            results.append(
                {
                    "repository": repo,
                    "pr_number": pr_number,
                    "synthesized": False,
                    "error": str(exc),
                }
            )
            continue

        test_file = str(payload.get("test_file", "")).strip() or f"tests/test_pr_{pr_number}.py"
        test_code = str(payload.get("test_code", "")).rstrip()
        target_path = (output_dir / _normalize_test_relpath(test_file)).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(test_code + "\n", encoding="utf-8")
        results.append(
            {
                "repository": repo,
                "pr_number": pr_number,
                "synthesized": True,
                "test_file": test_file,
                "artifact_path": str(target_path),
                "test_purpose": payload.get("test_purpose", ""),
            }
        )
    return {
        "artifact_dir": str(output_dir),
        "items": results,
        "synthesized_count": sum(1 for item in results if item.get("synthesized")),
        "failed_count": sum(1 for item in results if not item.get("synthesized")),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pr_rows = load_jsonl(Path(manifest["selected_pr_data"]).resolve())
    raw_fail2pass = EVAL_MOD.extract_changed_tests(Path(manifest["selected_pr_data"]).resolve())
    pr_map = {(item.get("repository"), item.get("pr_number")): item for item in pr_rows}

    enriched: List[Dict[str, Any]] = []
    for item in raw_fail2pass:
        source = pr_map.get((item.get("repository"), item.get("pr_number")), {})
        merged = dict(item)
        merged["description"] = source.get("description", "")
        merged["description_with_link_context"] = source.get("description_with_link_context", "")
        merged["diff"] = source.get("diff", "")
        enriched.append(merged)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted_tests_root = (args.output_dir / "original_pr_tests").resolve()
    extracted_tests_root.mkdir(parents=True, exist_ok=True)

    copied = []
    if args.copy_existing_pr_tests:
        copied = copy_pr_tests_from_original(
            enriched,
            original_repo=args.original_repo.resolve(),
            extracted_tests_root=extracted_tests_root,
        )

    synthesis = None
    if args.synthesize_missing_tests:
        if not args.base_url or not args.api_key:
            raise SystemExit("--synthesize-missing-tests requires --base-url and --api-key")
        synthesis = synthesize_original_fail2pass_tests(
            enriched,
            original_repo=args.original_repo.resolve(),
            output_dir=extracted_tests_root,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            reasoning_effort=args.reasoning_effort,
        )

    run_rewrite_script = ROOT / "scripts" / "run_test_rewrite_agent.py"
    result_json = args.output_dir / "test_rewrite_result.json"
    cmd = [
        sys.executable,
        str(run_rewrite_script),
        "--original-repo",
        str(args.original_repo.resolve()),
        "--generated-repo",
        str(_resolve_generated_repo_root(args.generated_repo)),
        "--original-tests-root",
        str(extracted_tests_root),
        "--output-dir",
        str((args.output_dir / "test_rewrite").resolve()),
        "--result-json",
        str(result_json),
        "--max-fix-rounds",
        str(max(0, int(args.max_fix_rounds))),
    ]
    if args.base_url and args.api_key:
        cmd.extend(["--base-url", args.base_url, "--api-key", args.api_key, "--model", args.model])
    if args.strict_api_mapping:
        cmd.append("--strict-api-mapping")

    subprocess.run(cmd, check=True)

    summary = {
        "manifest": str(args.manifest.resolve()),
        "original_repo": str(args.original_repo.resolve()),
        "generated_repo": str(_resolve_generated_repo_root(args.generated_repo)),
        "copied_pr_tests": copied,
        "llm_synthesized_original_tests": synthesis,
        "extracted_tests_root": str(extracted_tests_root),
        "test_rewrite_result_json": str(result_json),
        "test_rewrite_command": cmd,
    }
    summary_path = args.output_dir / "requirement_fail2pass_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
