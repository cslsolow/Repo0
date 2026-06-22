"""Agent for rewriting legacy tests to fit a generated repository and computing pass rate."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from agents.infra.llm_client import LLMClient
except Exception:
    try:
        from llm_client import LLMClient  # type: ignore
    except Exception:
        LLMClient = None  # type: ignore[assignment]


EXCLUDE_DIRS = {
    ".git",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "__pycache__",
}

COMMON_EXTERNAL_MODULES = {
    "pytest",
    "numpy",
    "pandas",
    "scipy",
    "dask",
    "matplotlib",
    "sklearn",
    "json",
    "re",
    "os",
    "sys",
    "pathlib",
    "typing",
    "collections",
    "itertools",
    "functools",
    "datetime",
    "math",
    "warnings",
    "unittest",
    "packaging",
}


class TestRewriteAgent:
    """Rewrite tests from an original repo to a generated repo and evaluate pass rate."""

    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.output_dir = output_dir
        self.llm_client = (
            LLMClient(self.api_config, output_dir, agent_name="test_rewrite")
            if self.api_config.get("api_key") and LLMClient is not None
            else None
        )

    def rewrite_tests_and_evaluate(
        self,
        original_repo_root: str | Path,
        generated_repo_root: str | Path,
        original_tests_root: str | Path | None = None,
        rewritten_tests_root: str | Path | None = None,
        pytest_args: Optional[List[str]] = None,
        max_fix_rounds: int = 0,
        strict_api_mapping: bool = True,
    ) -> Dict[str, Any]:
        """Rewrite original tests for generated repo APIs and run pytest to compute pass rate."""
        original_repo = Path(original_repo_root).resolve()
        generated_repo = Path(generated_repo_root).resolve()
        tests_root = Path(original_tests_root).resolve() if original_tests_root else self._default_tests_root(original_repo)
        rewritten_root = (
            Path(rewritten_tests_root).resolve()
            if rewritten_tests_root
            else Path(self.output_dir).resolve() / "rewritten_tests"
        )
        rewritten_root.mkdir(parents=True, exist_ok=True)

        test_files = list(self._discover_test_files(tests_root))
        estimated_original_test_cases = self._count_test_cases_from_files(test_files)
        api_index = self._build_generated_api_index(generated_repo)

        rewrite_results: List[Dict[str, Any]] = []
        failed_list: List[Dict[str, Any]] = []
        successful_rewrites: List[Dict[str, Any]] = []
        for test_file in test_files:
            rel = test_file.relative_to(tests_root)
            target_path = rewritten_root / rel
            target_path.parent.mkdir(parents=True, exist_ok=True)

            result = self.rewrite_single_test(
                test_file=test_file,
                target_path=target_path,
                generated_repo_root=generated_repo,
                api_index=api_index,
                strict_api_mapping=strict_api_mapping,
            )
            rewrite_results.append(result)
            if result.get("success"):
                successful_rewrites.append(result)
            else:
                failed_list.append(
                    {
                        "source": result.get("source", str(test_file)),
                        "target": result.get("target", str(target_path)),
                        "stage": result.get("stage", "rewrite"),
                        "reason": result.get("failure_reason", "unknown_rewrite_failure"),
                    }
                )

        if successful_rewrites:
            run_result = self.run_pytest(
                repo_root=generated_repo,
                tests_path=rewritten_root,
                extra_args=pytest_args or [],
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
                "skipped_reason": "no_successfully_rewritten_tests",
            }

        adaptation_history: List[Dict[str, Any]] = []
        if successful_rewrites and self.llm_client and max_fix_rounds > 0:
            run_result, adaptation_history = self._iteratively_fix_failed_tests(
                initial_run_result=run_result,
                rewritten_root=rewritten_root,
                generated_repo_root=generated_repo,
                api_index=api_index,
                pytest_args=pytest_args or [],
                max_fix_rounds=max_fix_rounds,
            )

        summary = run_result.get("summary", {})
        total = int(summary.get("total", 0))
        passed = int(summary.get("passed", 0))
        executed_pass_rate = (passed / total) if total > 0 else 0.0
        rewrite_success_rate = (len(successful_rewrites) / len(test_files)) if test_files else 0.0
        all_tests_pass_rate = (
            min(1.0, passed / estimated_original_test_cases) if estimated_original_test_cases > 0 else 0.0
        )

        return {
            "original_repo_root": str(original_repo),
            "generated_repo_root": str(generated_repo),
            "original_tests_root": str(tests_root),
            "rewritten_tests_root": str(rewritten_root),
            "test_files_count": len(test_files),
            "estimated_original_test_cases": int(estimated_original_test_cases),
            "remaining_test_files_count": len(successful_rewrites),
            "rewrite_success_count": len(successful_rewrites),
            "rewrite_failure_count": len(failed_list),
            "rewrite_success_rate": round(rewrite_success_rate, 4),
            "rewrite_results": rewrite_results,
            "failed_list": failed_list,
            "pytest": run_result,
            "executed_test_cases": int(total),
            "passed_test_cases": int(passed),
            "pass_rate": round(executed_pass_rate, 4),
            "remaining_pass_rate": round(executed_pass_rate, 4),
            "all_tests_pass_rate": round(all_tests_pass_rate, 4),
            "max_fix_rounds": int(max_fix_rounds),
            "strict_api_mapping": bool(strict_api_mapping),
            "adaptation_history": adaptation_history,
            "adaptation_rounds_run": len(adaptation_history),
        }

    def rewrite_single_test(
        self,
        test_file: Path,
        target_path: Path,
        generated_repo_root: Path,
        api_index: Dict[str, Any],
        strict_api_mapping: bool = True,
    ) -> Dict[str, Any]:
        """Rewrite one test file and persist it to target_path."""
        try:
            source = test_file.read_text(encoding="utf-8")
        except Exception as exc:
            return {
                "success": False,
                "source": str(test_file),
                "target": str(target_path),
                "stage": "read",
                "failure_reason": f"read_test_failed: {exc}",
            }

        if not self.llm_client:
            return {
                "success": False,
                "source": str(test_file),
                "target": str(target_path),
                "stage": "llm",
                "failure_reason": "llm_not_configured",
            }

        mapping_status = self._resolve_api_mapping_status(source, api_index)
        mapping_hints = self._build_mapping_hints(
            source=source,
            api_index=api_index,
            mapping_status=mapping_status,
        )
        if not mapping_status["ok"] and strict_api_mapping:
            return {
                "success": False,
                "source": str(test_file),
                "target": str(target_path),
                "stage": "api_mapping",
                "failure_reason": mapping_status["reason"],
                "mapping_details": mapping_status,
            }

        try:
            rewritten = self._rewrite_with_llm(
                source,
                test_file,
                generated_repo_root,
                api_index,
                mapping_hints=mapping_hints,
            )
        except Exception as exc:
            return {
                "success": False,
                "source": str(test_file),
                "target": str(target_path),
                "stage": "llm",
                "failure_reason": f"llm_request_failed: {exc}",
            }

        if not rewritten.strip():
            return {
                "success": False,
                "source": str(test_file),
                "target": str(target_path),
                "stage": "rewrite",
                "failure_reason": "rewrite_empty_output",
            }

        if not self._is_valid_python(rewritten):
            return {
                "success": False,
                "source": str(test_file),
                "target": str(target_path),
                "stage": "rewrite",
                "failure_reason": "rewrite_invalid_python",
            }

        rewritten = self._sanitize_rewritten_source(rewritten, target_path)
        if not rewritten.strip():
            return {
                "success": False,
                "source": str(test_file),
                "target": str(target_path),
                "stage": "rewrite",
                "failure_reason": "rewrite_empty_after_sanitize",
            }

        if not self._is_valid_python(rewritten):
            return {
                "success": False,
                "source": str(test_file),
                "target": str(target_path),
                "stage": "rewrite",
                "failure_reason": "rewrite_invalid_python_after_sanitize",
            }

        incompatible_reason = self._detect_incompatible_api_usage(rewritten)
        if incompatible_reason:
            return {
                "success": False,
                "source": str(test_file),
                "target": str(target_path),
                "stage": "validation",
                "failure_reason": f"rewrite_incompatible_with_generated_api: {incompatible_reason}",
            }

        try:
            target_path.write_text(rewritten, encoding="utf-8")
        except Exception as exc:
            return {
                "success": False,
                "source": str(test_file),
                "target": str(target_path),
                "stage": "write",
                "failure_reason": f"write_rewritten_test_failed: {exc}",
            }

        return {
            "success": True,
            "source": str(test_file),
            "target": str(target_path),
            "mode": "llm",
            "stage": "rewrite",
            "changed": rewritten != source,
            "mapping_details": mapping_status,
        }

    def _iteratively_fix_failed_tests(
        self,
        initial_run_result: Dict[str, Any],
        rewritten_root: Path,
        generated_repo_root: Path,
        api_index: Dict[str, Any],
        pytest_args: List[str],
        max_fix_rounds: int,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Iteratively fix failing rewritten tests by feeding pytest failures back to the LLM.

        Each round:
        1) group failing/error test cases by rewritten file
        2) ask LLM to patch each failing file
        3) rerun pytest if any file changed
        """
        run_result = initial_run_result
        history: List[Dict[str, Any]] = []

        for round_idx in range(1, max_fix_rounds + 1):
            test_cases = run_result.get("test_cases", []) or []
            failed_cases = [
                case
                for case in test_cases
                if str(case.get("status", "")).lower() in {"failed", "error"}
            ]
            if not failed_cases:
                break

            grouped, unresolved = self._group_failures_by_test_file(failed_cases, rewritten_root)
            round_info: Dict[str, Any] = {
                "round": round_idx,
                "failed_case_count": len(failed_cases),
                "target_file_count": len(grouped),
                "unresolved_case_count": len(unresolved),
                "updated_files_count": 0,
                "updated_files": [],
                "unchanged_files": [],
                "failed_files": [],
                "unresolved_cases": unresolved,
            }

            if not grouped:
                round_info["stop_reason"] = "no_mappable_failed_test_files"
                history.append(round_info)
                break

            for file_path, file_failures in grouped.items():
                try:
                    current_source = file_path.read_text(encoding="utf-8")
                except Exception as exc:
                    round_info["failed_files"].append(
                        {
                            "file": str(file_path),
                            "reason": f"read_failed: {exc}",
                        }
                    )
                    continue

                try:
                    mapping_hints = self._build_mapping_hints(
                        source=current_source,
                        api_index=api_index,
                        mapping_status=None,
                    )
                    rewritten = self._rewrite_failed_test_with_llm(
                        current_source=current_source,
                        test_file=file_path,
                        generated_repo_root=generated_repo_root,
                        api_index=api_index,
                        failures=file_failures,
                        run_result=run_result,
                        mapping_hints=mapping_hints,
                    )
                except Exception as exc:
                    round_info["failed_files"].append(
                        {
                            "file": str(file_path),
                            "reason": f"llm_request_failed: {exc}",
                        }
                    )
                    continue

                if not rewritten.strip():
                    round_info["failed_files"].append(
                        {
                            "file": str(file_path),
                            "reason": "rewrite_empty_output",
                        }
                    )
                    continue

                rewritten = self._sanitize_rewritten_source(rewritten, file_path)
                if not rewritten.strip():
                    round_info["failed_files"].append(
                        {
                            "file": str(file_path),
                            "reason": "rewrite_empty_after_sanitize",
                        }
                    )
                    continue

                if not self._is_valid_python(rewritten):
                    round_info["failed_files"].append(
                        {
                            "file": str(file_path),
                            "reason": "rewrite_invalid_python",
                        }
                    )
                    continue

                incompatible_reason = self._detect_incompatible_api_usage(rewritten)
                if incompatible_reason:
                    round_info["failed_files"].append(
                        {
                            "file": str(file_path),
                            "reason": f"rewrite_incompatible_with_generated_api:{incompatible_reason}",
                        }
                    )
                    continue

                if rewritten == current_source:
                    round_info["unchanged_files"].append(str(file_path))
                    continue

                try:
                    file_path.write_text(rewritten, encoding="utf-8")
                    round_info["updated_files_count"] += 1
                    round_info["updated_files"].append(str(file_path))
                except Exception as exc:
                    round_info["failed_files"].append(
                        {
                            "file": str(file_path),
                            "reason": f"write_failed: {exc}",
                        }
                    )

            history.append(round_info)
            if round_info["updated_files_count"] == 0:
                break

            run_result = self.run_pytest(
                repo_root=generated_repo_root,
                tests_path=rewritten_root,
                extra_args=pytest_args,
            )
            round_info["post_pytest_returncode"] = run_result.get("returncode")
            round_info["post_pytest_summary"] = run_result.get("summary", {})

            post_summary = run_result.get("summary", {}) or {}
            if int(post_summary.get("failed", 0)) == 0 and int(post_summary.get("errors", 0)) == 0:
                break

        return run_result, history

    def _group_failures_by_test_file(
        self,
        failed_cases: List[Dict[str, str]],
        rewritten_root: Path,
    ) -> tuple[Dict[Path, List[Dict[str, str]]], List[Dict[str, str]]]:
        grouped: Dict[Path, List[Dict[str, str]]] = defaultdict(list)
        unresolved: List[Dict[str, str]] = []

        for case in failed_cases:
            nodeid = str(case.get("test", ""))
            file_path = self._resolve_case_to_rewritten_file(nodeid, rewritten_root)
            if file_path is None:
                unresolved.append(case)
                continue
            grouped[file_path].append(case)

        return grouped, unresolved

    @staticmethod
    def _resolve_case_to_rewritten_file(nodeid: str, rewritten_root: Path) -> Optional[Path]:
        left = nodeid.split("::", 1)[0].strip()
        if not left:
            return None

        normalized = left.replace("\\", "/")
        candidates: List[str] = []

        if normalized.endswith(".py"):
            candidates.append(normalized)

        dotted = normalized[:-3].replace("/", ".") if normalized.endswith(".py") else normalized
        if dotted:
            if "rewritten_tests." in dotted:
                tail = dotted.split("rewritten_tests.", 1)[1]
                candidates.append(tail.replace(".", "/") + ".py")
            candidates.append(dotted.replace(".", "/") + ".py")

        for rel in candidates:
            rel_clean = rel.lstrip("./")
            direct = rewritten_root / rel_clean
            if direct.exists():
                return direct

            marker = "rewritten_tests/"
            if marker in rel_clean:
                suffix = rel_clean.split(marker, 1)[1]
                alt = rewritten_root / suffix
                if alt.exists():
                    return alt

        stem = dotted.rsplit(".", 1)[-1] if dotted else Path(normalized).stem
        if stem:
            matches = list(rewritten_root.rglob(f"{stem}.py"))
            if len(matches) == 1:
                return matches[0]

        return None

    def _rewrite_failed_test_with_llm(
        self,
        current_source: str,
        test_file: Path,
        generated_repo_root: Path,
        api_index: Dict[str, Any],
        failures: List[Dict[str, str]],
        run_result: Dict[str, Any],
        mapping_hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        assert self.llm_client is not None

        failure_lines = []
        for case in failures:
            status = case.get("status", "failed")
            test_name = case.get("test", "<unknown>")
            reason = case.get("reason", "")
            failure_lines.append(f"- [{status}] {test_name}: {reason}")
        failure_text = "\n".join(failure_lines) if failure_lines else "- <no failures provided>"

        api_summary = api_index.get("summary", "")
        mapping_hints_json = json.dumps(mapping_hints or {}, ensure_ascii=False, indent=2)
        stderr_tail = (run_result.get("stderr") or "")[-3000:]
        stdout_tail = (run_result.get("stdout") or "")[-3000:]

        prompt = f"""You are fixing a rewritten pytest file so it passes on the generated repository.

Rules:
- Keep the same test intent and assertions whenever possible.
- Adapt imports, construction, and API calls to the generated repo.
- When a module is already mapped, continue mapping to corresponding functions/classes in that module.
- Use function/module mapping hints when exact names differ.
- Do not delete tests just to make them pass.
- Return ONLY the full updated Python file content.

Generated Repo Root: {generated_repo_root}
Rewritten Test File: {test_file}

Generated Repo API Summary:
{api_summary}

Function/Module Mapping Hints:
{mapping_hints_json}

Current Rewritten Test Code:
{current_source}

Current Pytest Failures for This File:
{failure_text}

Pytest stdout tail:
{stdout_tail}

Pytest stderr tail:
{stderr_tail}
"""

        rewritten = self.llm_client.call(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise test migration engineer who fixes failing rewritten tests.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=32768,
        )
        return rewritten.strip()

    def run_pytest(
        self,
        repo_root: str | Path,
        tests_path: str | Path,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run pytest and return parsed summary."""
        root = Path(repo_root).resolve()
        test_target = Path(tests_path).resolve()

        junit_path = Path(self.output_dir).resolve() / "pytest_junit.xml"
        junit_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_target),
            f"--junitxml={junit_path}",
        ]
        if extra_args:
            cmd.extend(extra_args)

        env = dict(os.environ)
        current_pythonpath = env.get("PYTHONPATH", "")
        # Add repository root, rewritten-tests parent, and agents_output directory.
        # The generated code may use top-level imports like `from storage...`,
        # and rewritten tests may import package names rooted at agents_output.
        pythonpath_parts = [str(root), str(test_target.parent), str(test_target.parent.parent)]
        if current_pythonpath:
            pythonpath_parts.append(current_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
        )

        output_text = f"{proc.stdout}\n{proc.stderr}"
        summary = self.parse_pytest_summary(output_text)
        test_cases = self.parse_pytest_test_cases(
            junit_xml_path=junit_path,
            output_text=output_text,
        )

        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
            "summary": summary,
            "test_cases": test_cases,
        }

    @staticmethod
    def parse_pytest_summary(output_text: str) -> Dict[str, int]:
        """Parse pytest terminal summary counts from output text."""
        keys = [
            "passed",
            "failed",
            "errors",
            "error",
            "skipped",
            "xfailed",
            "xpassed",
            "deselected",
        ]
        counts = defaultdict(int)

        for key in keys:
            pattern = re.compile(rf"(\d+)\s+{key}\b")
            for m in pattern.finditer(output_text.lower()):
                counts[key] = max(counts[key], int(m.group(1)))

        # Normalize singular/plural error bucket.
        counts["errors"] = max(counts["errors"], counts["error"])

        total = (
            counts["passed"]
            + counts["failed"]
            + counts["errors"]
            + counts["skipped"]
            + counts["xfailed"]
            + counts["xpassed"]
        )

        return {
            "passed": counts["passed"],
            "failed": counts["failed"],
            "errors": counts["errors"],
            "skipped": counts["skipped"],
            "xfailed": counts["xfailed"],
            "xpassed": counts["xpassed"],
            "deselected": counts["deselected"],
            "total": total,
        }

    @staticmethod
    def parse_pytest_test_cases(junit_xml_path: Path, output_text: str) -> List[Dict[str, str]]:
        """
        Parse per-test execution results.

        Preferred source is pytest JUnit XML (includes passed/failed/errors/skipped).
        If XML is unavailable, fall back to terminal summary lines for failed/error tests.
        """
        if junit_xml_path.exists():
            try:
                tree = ET.parse(junit_xml_path)
                root = tree.getroot()
                results: List[Dict[str, str]] = []

                for testcase in root.iter("testcase"):
                    classname = testcase.attrib.get("classname", "").strip()
                    name = testcase.attrib.get("name", "").strip()
                    file_attr = testcase.attrib.get("file", "").strip()
                    if file_attr:
                        nodeid = f"{file_attr}::{name}" if name else file_attr
                    else:
                        nodeid = f"{classname}::{name}" if classname else name

                    status = "passed"
                    reason = ""

                    failure = testcase.find("failure")
                    error = testcase.find("error")
                    skipped = testcase.find("skipped")

                    if failure is not None:
                        status = "failed"
                        reason = (
                            failure.attrib.get("message")
                            or (failure.text or "").strip().splitlines()[0] if (failure.text or "").strip() else ""
                        )
                    elif error is not None:
                        status = "error"
                        reason = (
                            error.attrib.get("message")
                            or (error.text or "").strip().splitlines()[0] if (error.text or "").strip() else ""
                        )
                    elif skipped is not None:
                        status = "skipped"
                        reason = (
                            skipped.attrib.get("message")
                            or (skipped.text or "").strip().splitlines()[0] if (skipped.text or "").strip() else ""
                        )

                    results.append(
                        {
                            "test": nodeid,
                            "status": status,
                            "reason": reason,
                        }
                    )

                if results:
                    return results
            except Exception:
                # Fall through to summary-line parsing.
                pass

        # Fallback: parse failed/error lines from terminal summary.
        results: List[Dict[str, str]] = []
        line_re = re.compile(r"^(FAILED|ERROR)\s+(.+?)\s+-\s+(.+)$")
        for line in output_text.splitlines():
            m = line_re.match(line.strip())
            if not m:
                continue
            status = "failed" if m.group(1) == "FAILED" else "error"
            results.append(
                {
                    "test": m.group(2).strip(),
                    "status": status,
                    "reason": m.group(3).strip(),
                }
            )
        return results

    def _rewrite_with_llm(
        self,
        test_source: str,
        test_file: Path,
        generated_repo_root: Path,
        api_index: Dict[str, Any],
        mapping_hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        assert self.llm_client is not None

        api_summary = api_index.get("summary", "")
        mapping_hints_json = json.dumps(mapping_hints or {}, ensure_ascii=False, indent=2)
        prompt = f"""You are rewriting python tests to fit a new implementation repository.

Goal:
- Rewrite tests so they validate equivalent behavior on the generated repo.
- Keep assertions and test intent whenever possible.
- Update imports, object construction, and API calls to match new repo APIs.
- If source modules can be matched, continue mapping required symbols/functions from those modules.
- Prefer function-level mapping hints when exact names are unavailable.
- Return ONLY the rewritten test file content (plain Python code, no markdown).

Generated Repo Root: {generated_repo_root}
Test File Path: {test_file}

Generated Repo API Summary:
{api_summary}

Function/Module Mapping Hints:
{mapping_hints_json}

Original Test Code:
{test_source}
"""
        rewritten = self.llm_client.call(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise test migration engineer.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=32768,
        )
        return rewritten.strip()

    def _resolve_api_mapping_status(self, source: str, api_index: Dict[str, Any]) -> Dict[str, Any]:
        """Validate whether project-facing imports can map to generated repository APIs."""
        symbol_to_module = api_index.get("symbol_to_module", {})
        module_roots = api_index.get("module_roots", set())

        imported_symbols: List[str] = []
        imported_modules: List[str] = []

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return {
                "ok": False,
                "reason": f"source_test_parse_failed: {exc}",
                "required_symbols": [],
                "required_modules": [],
                "mapped_symbols": [],
                "mapped_modules": [],
            }

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if not node.module:
                    continue
                root = node.module.split(".")[0]
                if root and root not in COMMON_EXTERNAL_MODULES:
                    imported_modules.append(root)
                for alias in node.names:
                    if alias.name != "*" and alias.name not in COMMON_EXTERNAL_MODULES:
                        imported_symbols.append(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root and root not in COMMON_EXTERNAL_MODULES:
                        imported_modules.append(root)

        required_symbols = sorted(set(imported_symbols))
        required_modules = sorted(set(imported_modules))
        mapped_symbols = sorted([sym for sym in required_symbols if sym in symbol_to_module])
        mapped_modules = sorted([mod for mod in required_modules if mod in module_roots])

        if not required_symbols and not required_modules:
            return {
                "ok": False,
                "reason": "api_mapping_not_found: no_project_api_imports_detected",
                "required_symbols": required_symbols,
                "required_modules": required_modules,
                "mapped_symbols": mapped_symbols,
                "mapped_modules": mapped_modules,
            }

        if not mapped_symbols and not mapped_modules:
            return {
                "ok": False,
                "reason": "api_mapping_not_found: no_matching_symbols_or_modules",
                "required_symbols": required_symbols,
                "required_modules": required_modules,
                "mapped_symbols": mapped_symbols,
                "mapped_modules": mapped_modules,
            }

        return {
            "ok": True,
            "reason": "",
            "required_symbols": required_symbols,
            "required_modules": required_modules,
            "mapped_symbols": mapped_symbols,
            "mapped_modules": mapped_modules,
        }

    @staticmethod
    def _normalize_symbol_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    def _build_mapping_hints(
        self,
        source: str,
        api_index: Dict[str, Any],
        mapping_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build function/module mapping hints to help LLM map old APIs to new ones.
        """
        symbol_to_module: Dict[str, str] = api_index.get("symbol_to_module", {})
        module_roots = sorted(api_index.get("module_roots", set()))
        all_symbols = sorted(symbol_to_module.keys())

        required_symbols: List[str] = []
        required_modules: List[str] = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {
                "required_symbols": [],
                "required_modules": [],
                "module_root_suggestions": {},
                "symbol_suggestions": [],
                "mapped_module_roots": [],
            }

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if not node.module:
                    continue
                root = node.module.split(".")[0]
                if root and root not in COMMON_EXTERNAL_MODULES:
                    required_modules.append(root)
                for alias in node.names:
                    if alias.name != "*" and alias.name not in COMMON_EXTERNAL_MODULES:
                        required_symbols.append(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root and root not in COMMON_EXTERNAL_MODULES:
                        required_modules.append(root)

        required_symbols = sorted(set(required_symbols))
        required_modules = sorted(set(required_modules))

        mapped_module_roots = set((mapping_status or {}).get("mapped_modules", []))
        if not mapped_module_roots:
            for mod in required_modules:
                fuzzy = get_close_matches(mod, module_roots, n=1, cutoff=0.7)
                if fuzzy:
                    mapped_module_roots.add(fuzzy[0])

        module_root_suggestions: Dict[str, List[str]] = {}
        for mod in required_modules:
            if mod in module_roots:
                module_root_suggestions[mod] = [mod]
            else:
                module_root_suggestions[mod] = get_close_matches(mod, module_roots, n=3, cutoff=0.55)

        preferred_symbols = [
            sym
            for sym, mod in symbol_to_module.items()
            if mod.split(".")[0] in mapped_module_roots
        ]
        if not preferred_symbols:
            preferred_symbols = all_symbols

        symbol_suggestions: List[Dict[str, Any]] = []
        for sym in required_symbols:
            if sym in symbol_to_module:
                symbol_suggestions.append(
                    {
                        "source_symbol": sym,
                        "target_symbol": sym,
                        "target_module": symbol_to_module[sym],
                        "score": 1.0,
                        "strategy": "exact",
                    }
                )
                continue

            best_candidate = None
            best_score = 0.0
            norm_sym = self._normalize_symbol_name(sym)

            for cand in preferred_symbols:
                if norm_sym and norm_sym == self._normalize_symbol_name(cand):
                    best_candidate = cand
                    best_score = 0.99
                    break

            if best_candidate is None:
                fuzzy = get_close_matches(sym, preferred_symbols, n=3, cutoff=0.6)
                if fuzzy:
                    best_candidate = fuzzy[0]
                    best_score = SequenceMatcher(a=sym.lower(), b=best_candidate.lower()).ratio()

            if best_candidate is None and preferred_symbols is not all_symbols:
                fuzzy_global = get_close_matches(sym, all_symbols, n=3, cutoff=0.6)
                if fuzzy_global:
                    best_candidate = fuzzy_global[0]
                    best_score = SequenceMatcher(a=sym.lower(), b=best_candidate.lower()).ratio()

            if best_candidate is not None:
                symbol_suggestions.append(
                    {
                        "source_symbol": sym,
                        "target_symbol": best_candidate,
                        "target_module": symbol_to_module.get(best_candidate, ""),
                        "score": round(float(best_score), 4),
                        "strategy": "normalized_or_fuzzy",
                    }
                )

        return {
            "required_symbols": required_symbols,
            "required_modules": required_modules,
            "module_root_suggestions": module_root_suggestions,
            "symbol_suggestions": symbol_suggestions,
            "mapped_module_roots": sorted(mapped_module_roots),
        }

    @staticmethod
    def _sanitize_rewritten_source(source: str, target_path: Path) -> str:
        """Apply deterministic cleanup for common invalid rewrite artifacts."""
        normalized = source

        # Normalize legacy rewritten-tests import paths to generated_code layout.
        normalized = normalized.replace(
            "agents_output.test_rewrite.rewritten_tests",
            "agents_output.generated_code.rewritten_tests",
        )
        normalized = normalized.replace(
            '"agents_output.test_rewrite.rewritten_tests',
            '"agents_output.generated_code.rewritten_tests',
        )
        normalized = normalized.replace(
            "'agents_output.test_rewrite.rewritten_tests",
            "'agents_output.generated_code.rewritten_tests",
        )

        # Collapse repo-specific prefixes to generated_code (e.g. agents_output.tinydb.xxx).
        normalized = re.sub(
            r"\bagents_output\.(?!generated_code\b)([A-Za-z_][A-Za-z0-9_]*)\.",
            "agents_output.generated_code.",
            normalized,
        )

        if target_path.name != "conftest.py":
            return normalized

        cleaned_lines: List[str] = []
        # Remove self-referential conftest imports generated by LLM, e.g.:
        # from tinydb.rewritten_tests.conftest import db, storage
        self_import_re = re.compile(r"^\s*from\s+[\w\.]*rewritten_tests\.conftest\s+import\s+.+$")
        for line in normalized.splitlines():
            if self_import_re.match(line):
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines).strip()
        return (cleaned + "\n") if cleaned else ""

    def _build_generated_api_index(self, repo_root: Path) -> Dict[str, Any]:
        symbol_to_module: Dict[str, str] = {}
        module_exports: Dict[str, List[str]] = defaultdict(list)

        for file_path in self._iter_python_files(repo_root):
            if "tests" in file_path.parts:
                continue
            module_name = self._module_name(repo_root, file_path)
            if not module_name:
                continue
            for symbol in self._extract_top_level_symbols(file_path):
                # Prefer first match for deterministic heuristic rewrite.
                symbol_to_module.setdefault(symbol, module_name)
                module_exports[module_name].append(symbol)

        summary_lines: List[str] = []
        for module in sorted(module_exports.keys())[:120]:
            exports = sorted(set(module_exports[module]))[:20]
            summary_lines.append(f"- {module}: {', '.join(exports)}")
        module_roots = {m.split(".")[0] for m in module_exports.keys() if m}

        return {
            "symbol_to_module": symbol_to_module,
            "module_exports": module_exports,
            "module_roots": module_roots,
            "summary": "\n".join(summary_lines),
        }

    def _count_test_cases_from_files(self, test_files: Iterable[Path]) -> int:
        total = 0
        for test_file in test_files:
            try:
                source = test_file.read_text(encoding="utf-8")
            except Exception:
                continue
            total += self._count_test_cases_from_source(source)
        return total

    @staticmethod
    def _count_test_cases_from_source(source: str) -> int:
        """
        Estimate test case count from source code:
        - top-level `test_*` functions
        - `test_*` methods in `Test*` classes
        """
        try:
            tree = ast.parse(source)
        except Exception:
            # Fallback for invalid source text.
            return len(re.findall(r"(?m)^\s*def\s+test_[A-Za-z0-9_]*\s*\(", source))

        total = 0
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                total += 1
                continue

            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                        total += 1

        return total

    @staticmethod
    def _module_name(repo_root: Path, file_path: Path) -> str:
        rel = file_path.relative_to(repo_root)
        if rel.name == "__init__.py":
            rel = rel.parent
        return ".".join(rel.with_suffix("").parts)

    @staticmethod
    def _extract_top_level_symbols(file_path: Path) -> List[str]:
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            return []

        symbols: List[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(node.name)
        return symbols

    @staticmethod
    def _iter_python_files(root: Path) -> Iterable[Path]:
        for path in root.rglob("*.py"):
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            yield path

    @staticmethod
    def _default_tests_root(original_repo: Path) -> Path:
        candidate = original_repo / "tests"
        return candidate if candidate.exists() else original_repo

    @staticmethod
    def _discover_test_files(root: Path) -> Iterable[Path]:
        for path in root.rglob("*.py"):
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            if path.name.startswith("test_") or path.name.endswith("_test.py") or "tests" in path.parts:
                yield path

    @staticmethod
    def _is_valid_python(source: str) -> bool:
        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _detect_incompatible_api_usage(source: str) -> str:
        """
        Detect obvious legacy API usage that conflicts with common generated-code signatures.

        This catches high-confidence mismatches observed in rewritten tests, such as:
        - tinydb-style `Query().field` / `Query()['field']` against QueryBuilder APIs
        - `JSONFileStorage(path)` / `MemoryStorage().write(...)` when backend classes expose CRUD-style interface
        - passing storage *classes* (not instances) into create_caching_middleware
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ""

        query_aliases: set[str] = set()
        storage_aliases: set[str] = set()
        caching_factory_aliases: set[str] = set()
        caching_class_aliases: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    if module.endswith("query_system.query_builder") and alias.name == "QueryBuilder":
                        query_aliases.add(local_name)
                    if module.endswith("storage.storage_backend_registry") and alias.name in {"MemoryStorage", "JSONFileStorage"}:
                        storage_aliases.add(local_name)
                    if module.endswith("middleware.caching_middleware") and alias.name == "create_caching_middleware":
                        caching_factory_aliases.add(local_name)
                    if module.endswith("middleware.caching_middleware") and alias.name == "CachingMiddleware":
                        caching_class_aliases.add(local_name)

        # 1) QueryBuilder legacy syntax checks
        allowed_query_methods = {
            "field",
            "transform",
            "equals",
            "not_equals",
            "less_than",
            "greater_than",
            "less_than_or_equal",
            "greater_than_or_equal",
            "matches",
            "test",
            "and_",
            "or_",
            "build",
        }

        def _is_query_ctor_call(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in query_aliases
            )

        if query_aliases:
            for node in ast.walk(tree):
                # Query()['field'] style
                if isinstance(node, ast.Subscript) and _is_query_ctor_call(node.value):
                    return "query_builder_subscript_not_supported"

                # Query().value / Query().doc / ...
                if isinstance(node, ast.Attribute) and _is_query_ctor_call(node.value):
                    if node.attr not in allowed_query_methods:
                        return f"query_builder_legacy_attribute_access:{node.attr}"

        # 2) Storage backend signature/method checks
        storage_vars: dict[str, str] = {}
        if storage_aliases:
            for node in ast.walk(tree):
                # Track assignments like s = MemoryStorage() / storage = JSONFileStorage(...)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                    ctor = node.value.func.id
                    if ctor in storage_aliases:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                storage_vars[target.id] = ctor

                # Direct constructor calls with args/kwargs often indicate legacy style for these generated backends
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in storage_aliases:
                    if node.args or node.keywords:
                        return f"storage_constructor_legacy_signature:{node.func.id}"

            disallowed_storage_methods = {"write", "close", "flush", "read_all", "all"}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    var_name = node.func.value.id
                    if var_name in storage_vars and node.func.attr in disallowed_storage_methods:
                        return f"storage_legacy_method_call:{storage_vars[var_name]}.{node.func.attr}"

        # 3) Caching middleware factory checks (class passed instead of instance)
        if caching_factory_aliases and storage_aliases:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in caching_factory_aliases:
                    if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in storage_aliases:
                        return "caching_middleware_factory_expects_instance_not_class"

        # 4) Direct CachingMiddleware(class) misuse
        if caching_class_aliases and storage_aliases:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in caching_class_aliases:
                    if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in storage_aliases:
                        return "caching_middleware_ctor_expects_instance_not_class"

        return ""


__all__ = ["TestRewriteAgent"]
