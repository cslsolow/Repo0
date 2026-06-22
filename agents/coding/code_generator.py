"""Code Generator agent that generates actual code based on architecture and requirements."""

from __future__ import annotations

import ast
import fcntl
import json
import logging
import os
import py_compile
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.infra.llm_client import LLMClient
from agents.package_root import normalize_python_package_root

from .fix_agent import FixAgent
from .import_postcheck_fix_agent import ImportPostcheckFixAgent
from .patch_agent import PatchAgent
from .skeleton_review_agent import SkeletonReviewAgent
from .structured_contracts import extract_structured_contract_facts, find_structured_contract_issues
from .test_review_agent import TestReviewAgent
from .tdd_pip_heuristic import (
    missing_import_roots_from_pytest_log,
    sandbox_top_level_names,
    specs_from_project_import_closure,
    specs_from_sources_and_sandbox,
    specs_from_missing_roots,
)

# Default image for skeleton+TDD pytest (override with api_config ``tdd_docker_image``).
DEFAULT_TDD_DOCKER_IMAGE = "repo0-codegen-tdd:latest"


class CodeGeneratorAgent:
    """Generate actual code files based on architecture design and requirements."""

    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.output_dir = output_dir
        self.stats_dir = Path(output_dir)
        self.codegen_timing_events_file = self.stats_dir / "codegen_timing_events.json"
        self.codegen_timing_report_file = self.stats_dir / "codegen_timing_report.json"
        self.codegen_timing_lock_file = self.stats_dir / "codegen_timing.lock"
        self.llm_client = LLMClient(self.api_config, output_dir, agent_name="code_generator") if self.api_config.get("api_key") else None
        self.codegen_max_retry_times = int(self.api_config.get("codegen_max_retry_times", 3))
        self.skeleton_review_max_retries = max(
            0, int(self.api_config.get("skeleton_review_max_retries", 2))
        )
        self.skeleton_review_llm_max_retries = max(
            0, int(self.api_config.get("skeleton_review_llm_max_retries", 1))
        )
        self.test_review_max_retries = max(
            0, int(self.api_config.get("test_review_max_retries", 1))
        )
        self.codegen_timeout_seconds = float(self.api_config.get("codegen_timeout_seconds", 600))
        self.fix_agent = FixAgent()
        self.enable_syntax_autofix = self._parse_bool(
            self.api_config.get("enable_syntax_autofix", True)
        )
        self.enable_two_stage_file_plan = self._parse_bool(
            self.api_config.get("enable_two_stage_file_plan", True)
        )
        self.enable_skeleton_fill_tdd = self._parse_bool(
            self.api_config.get("enable_skeleton_fill_tdd", True)
        )
        self.tdd_max_fix_retries = max(0, int(self.api_config.get("tdd_max_fix_retries", 3)))
        self.tdd_pytest_timeout = max(10, int(self.api_config.get("tdd_pytest_timeout", 600)))
        self.tdd_disable_docker = self._parse_bool(self.api_config.get("tdd_disable_docker", False))
        _raw_img = str(self.api_config.get("tdd_docker_image", "")).strip()
        _local_aliases = {"local", "host", "none", "false", "0", "no", "off"}
        if self.tdd_disable_docker or _raw_img.lower() in _local_aliases:
            self.tdd_docker_image = ""
        elif _raw_img:
            self.tdd_docker_image = _raw_img
        elif self.enable_skeleton_fill_tdd:
            self.tdd_docker_image = DEFAULT_TDD_DOCKER_IMAGE
        else:
            self.tdd_docker_image = ""
        self.repo_name = str(self.api_config.get("repo", "")).strip().lower()
        self.tdd_pip_timeout = max(30, int(self.api_config.get("tdd_pip_timeout", 600)))
        _ppr = str(self.api_config.get("tdd_pip_project_root", "")).strip()
        _proj = Path(_ppr).expanduser() if _ppr else None
        self._tdd_pip_project_root: Optional[Path] = (
            _proj.resolve()
            if _proj is not None and _proj.is_dir()
            else None
        )
        self.tdd_missing_module_pip_retries = max(
            0, int(self.api_config.get("tdd_missing_module_pip_retries", 3))
        )
        self.post_generation_max_repair_rounds = max(
            1, int(self.api_config.get("post_generation_max_repair_rounds", 6))
        )
        self.tdd_docker_network_host = self._parse_bool(
            self.api_config.get("tdd_docker_network_host", True)
        )
        self.path_allowed_roots = self._parse_allowed_roots(
            self.api_config.get("path_allowed_roots")
        )
        self.last_file_plan: Dict[str, str] = {}
        self._patch_agent: PatchAgent | None = None
        self._import_postcheck_fix_agent: ImportPostcheckFixAgent | None = None
        self._skeleton_review_agent: SkeletonReviewAgent | None = None
        self._test_review_agent: TestReviewAgent | None = None
        self.import_postcheck_max_fix_attempts = max(
            0, int(self.api_config.get("import_postcheck_max_fix_attempts", 10))
        )
        self.package_postcheck_max_fix_attempts = max(
            0, int(self.api_config.get("package_postcheck_max_fix_attempts", 10))
        )

    def _primary_python_package_root(self) -> str:
        return normalize_python_package_root(self.repo_name, default="src")

    def _primary_python_package_pattern(self) -> str:
        return f"{self._primary_python_package_root()}.*"

    def _forbidden_peer_repo_roots(self) -> Set[str]:
        configured = self.api_config.get("peer_framework_roots")
        if not configured:
            configured = self.api_config.get("forbidden_peer_repo_roots")

        roots: Set[str] = set()
        if isinstance(configured, (list, tuple, set)):
            roots = {
                normalize_python_package_root(str(item), default="")
                for item in configured
                if str(item).strip()
            }
        elif isinstance(configured, str) and configured.strip():
            roots = {
                normalize_python_package_root(token, default="")
                for token in configured.split(",")
                if token.strip()
            }
        else:
            roots = {"statsmodels", "django", "sklearn", "scikit_learn"}

        current = self._primary_python_package_root()
        if current:
            roots.discard(current)
        if current == "scikit_learn":
            roots.discard("sklearn")
        if current == "sklearn":
            roots.discard("scikit_learn")
        return roots

    def _peer_repo_constraint_text(self, *, include_generic_utils_note: bool = True) -> str:
        roots = sorted(self._forbidden_peer_repo_roots())
        if roots:
            examples = ", ".join(f"`{root}`" for root in roots)
            base = (
                "Do not import or delegate the component's core behavior to peer framework "
                f"repositories such as {examples}."
            )
        else:
            base = (
                "Do not import or delegate the component's core behavior to peer framework "
                "repositories outside this generated repository."
            )
        if include_generic_utils_note:
            base += " Reusing already generated local components and generic utility libraries is allowed."
        return base

    @staticmethod
    def _action_guidance_block(component: Dict[str, Any]) -> str:
        action = str(component.get("recommended_action") or "").strip().lower()
        rationale = str(component.get("recommended_action_rationale") or "").strip()
        if action != "revise":
            return ""

        rationale_line = f"\nRevise rationale: {rationale}" if rationale else ""
        return (
            "\n=== ACTION GUIDANCE ===\n"
            "The strategist marked this component as `revise`.\n"
            "Use this as a soft hint: think through whether the skeleton should expose a cleaner internal API or class/function breakdown while keeping the current component boundary stable."
            f"{rationale_line}\n"
        )

    @staticmethod
    def _load_json_file(path: Path, default: Any) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    @staticmethod
    def _atomic_write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_codegen_timing_", suffix=".json", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def _with_codegen_timing_lock(self, callback) -> Any:
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        with open(self.codegen_timing_lock_file, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                return callback()
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _record_codegen_timing_event(
        self,
        *,
        component_name: str,
        stage: str,
        started_at_perf: float,
        status: str = "completed",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        duration_sec = max(0.0, time.perf_counter() - started_at_perf)
        event = {
            "timestamp": datetime.now().isoformat(),
            "component_name": str(component_name or ""),
            "stage": str(stage or ""),
            "duration_sec": round(duration_sec, 6),
            "status": str(status or "completed"),
            "meta": dict(meta or {}),
        }

        def _persist() -> None:
            events = self._load_json_file(self.codegen_timing_events_file, [])
            if not isinstance(events, list):
                events = []
            events.append(event)
            self._atomic_write_json(self.codegen_timing_events_file, events)
            summary = self._build_codegen_timing_report(events)
            self._atomic_write_json(self.codegen_timing_report_file, summary)

        self._with_codegen_timing_lock(_persist)

    def _build_codegen_timing_report(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "updated_at": datetime.now().isoformat(),
            "total_events": len(events),
            "total_duration_sec": 0.0,
            "by_stage": {},
            "by_component": {},
            "slowest_events": [],
        }

        def _bucket(container: Dict[str, Any], key: str) -> Dict[str, Any]:
            return container.setdefault(
                key or "<unknown>",
                {
                    "events": 0,
                    "total_duration_sec": 0.0,
                    "avg_duration_sec": 0.0,
                    "max_duration_sec": 0.0,
                    "failures": 0,
                },
            )

        slowest: List[Dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            duration = float(event.get("duration_sec", 0.0) or 0.0)
            report["total_duration_sec"] += duration
            status = str(event.get("status", "") or "completed")

            for container, key in (
                (report["by_stage"], str(event.get("stage", "") or "<unknown>")),
                (report["by_component"], str(event.get("component_name", "") or "<unknown>")),
            ):
                bucket = _bucket(container, key)
                bucket["events"] += 1
                bucket["total_duration_sec"] += duration
                bucket["max_duration_sec"] = max(bucket["max_duration_sec"], duration)
                if status != "completed":
                    bucket["failures"] += 1

            slowest.append(
                {
                    "component_name": event.get("component_name", ""),
                    "stage": event.get("stage", ""),
                    "duration_sec": duration,
                    "status": status,
                    "meta": event.get("meta", {}),
                }
            )

        for container in (report["by_stage"], report["by_component"]):
            for bucket in container.values():
                events_count = int(bucket.get("events", 0) or 0)
                if events_count:
                    bucket["avg_duration_sec"] = round(bucket["total_duration_sec"] / events_count, 6)
                bucket["total_duration_sec"] = round(bucket["total_duration_sec"], 6)
                bucket["max_duration_sec"] = round(bucket["max_duration_sec"], 6)

        report["total_duration_sec"] = round(report["total_duration_sec"], 6)
        report["slowest_events"] = sorted(
            slowest,
            key=lambda item: -(item.get("duration_sec", 0.0) or 0.0),
        )[:100]
        return report

    def generate_code(
        self,
        component: Dict[str, Any],
        requirement: Dict[str, Any],
        architecture: Dict[str, Any],
        language: str = "python",
        implemented_components_context: str = "",
        planned_file_path: Optional[str] = None,
        previous_attempt_feedback: str = "",
    ) -> Dict[str, Any]:
        """
        Generate code for a specific component.
        
        Args:
            component: Component specification with name and responsibilities
            requirement: The requirement being implemented
            architecture: Full architecture context
            language: Target programming language
            implemented_components_context: Formatted string of already implemented components
            
        Returns:
            Dictionary with:
            - file_path: Suggested file path
            - code: Generated code content
            - imports: Required imports/dependencies
            - tests: Suggested test cases
            - documentation: Code documentation
        """
        if not self.llm_client:
            return self._fallback_generate_code(component, requirement, language, planned_file_path)

        if self.enable_skeleton_fill_tdd and str(language).lower() == "python":
            component_name = component.get("name", "Component")
            max_skeleton_attempts = max(1, self.skeleton_review_max_retries + 1)
            retry_feedback = str(previous_attempt_feedback or "").strip()
            for attempt_idx in range(max_skeleton_attempts):
                try:
                    return self._generate_code_skeleton_tdd(
                        component,
                        requirement,
                        architecture,
                        implemented_components_context=implemented_components_context,
                        planned_file_path=planned_file_path,
                        previous_attempt_feedback=retry_feedback,
                    )
                except Exception as exc:
                    is_infra_or_parse = self._is_infra_or_parsing_error(exc)
                    if is_infra_or_parse and attempt_idx + 1 < max_skeleton_attempts:
                        logging.warning(
                            "Skeleton+TDD infra/parsing error for component '%s' (attempt %d/%d): %s; retrying same path",
                            component_name,
                            attempt_idx + 1,
                            max_skeleton_attempts,
                            exc,
                        )
                        continue
                    failure_feedback = (
                        f"Skeleton/TDD attempt {attempt_idx + 1} failed for component '{component_name}': {exc}"
                    )
                    retry_feedback = (
                        f"{retry_feedback}\n{failure_feedback}".strip()
                        if retry_feedback else failure_feedback
                    )
                    if attempt_idx + 1 < max_skeleton_attempts:
                        logging.warning(
                            "Skeleton+TDD generation failed for component '%s' (attempt %d/%d): %s; "
                            "retrying skeleton path with accumulated feedback",
                            component_name,
                            attempt_idx + 1,
                            max_skeleton_attempts,
                            exc,
                        )
                        continue
                    if is_infra_or_parse:
                        logging.warning(
                            "Skeleton+TDD infra/parsing retries exhausted for component '%s' (%s); "
                            "falling back to legacy single-shot prompt",
                            component_name,
                            exc,
                        )
                        break
                    logging.warning(
                        "Skeleton+TDD codegen failed for component '%s' (%s); falling back to legacy single-shot prompt",
                        component_name,
                        exc,
                    )
                    break

        return self._generate_code_single_shot(
            component,
            requirement,
            architecture,
            language,
            implemented_components_context,
            planned_file_path,
            retry_feedback if 'retry_feedback' in locals() else previous_attempt_feedback,
        )

    @staticmethod
    def _derive_generation_status(code_result: Dict[str, Any]) -> str:
        tdd_meta = code_result.get("skeleton_fill_tdd", {})
        if isinstance(tdd_meta, dict):
            final_rc = tdd_meta.get("final_pytest_rc")
            try:
                if final_rc is not None and int(final_rc) != 0:
                    return "retained_after_tdd_failure"
                if final_rc is not None and int(final_rc) == 0:
                    return "implemented"
            except Exception:
                pass
        return "implemented"

    def _get_patch_agent(self) -> PatchAgent:
        if self._patch_agent is None:
            self._patch_agent = PatchAgent(self.api_config, self.output_dir)
        return self._patch_agent

    def _get_import_postcheck_fix_agent(self) -> ImportPostcheckFixAgent:
        if self._import_postcheck_fix_agent is None:
            self._import_postcheck_fix_agent = ImportPostcheckFixAgent(self.api_config, self.output_dir)
        return self._import_postcheck_fix_agent

    def _get_skeleton_review_agent(self) -> SkeletonReviewAgent:
        if self._skeleton_review_agent is None:
            self._skeleton_review_agent = SkeletonReviewAgent(self.api_config, self.output_dir)
        return self._skeleton_review_agent

    def _get_test_review_agent(self) -> TestReviewAgent:
        if self._test_review_agent is None:
            self._test_review_agent = TestReviewAgent(self.api_config, self.output_dir)
        return self._test_review_agent

    def _apply_skeleton_review_with_retries(
        self,
        *,
        component_name: str,
        responsibilities: List[str],
        planned_file_path: str,
        skeleton_code: str,
        implemented_components_context: str = "",
        previous_attempt_feedback: str = "",
    ) -> str:
        reviewed_code = skeleton_code
        review_feedback = str(previous_attempt_feedback or "").strip()
        review_attempts = max(1, self.skeleton_review_llm_max_retries + 1)
        skeleton_review_agent = self._get_skeleton_review_agent()
        for attempt_idx in range(review_attempts):
            patch_result = skeleton_review_agent.review_skeleton(
                component_name=component_name,
                responsibilities=responsibilities,
                planned_file_path=planned_file_path,
                skeleton_code=reviewed_code,
                implemented_components_context=implemented_components_context,
                previous_attempt_feedback=review_feedback,
            )
            reviewed_candidate = skeleton_review_agent.extract_reviewed_skeleton_code(
                patch_result=patch_result,
                planned_file_path=planned_file_path,
                fallback_skeleton_code=reviewed_code,
            )
            reviewed_code = reviewed_candidate.strip() or reviewed_code
            remaining_alignment = self._find_skeleton_responsibility_alignment_issues(
                component_name=component_name,
                responsibilities=responsibilities,
                skeleton_code=reviewed_code,
                rel_path=planned_file_path,
            )
            if not remaining_alignment:
                return reviewed_code
            if attempt_idx + 1 >= review_attempts:
                logging.warning(
                    "Skeleton review exhausted for component '%s'; continuing with current skeleton despite remaining alignment issues: %s",
                    component_name,
                    "; ".join(remaining_alignment),
                )
                return reviewed_code
            review_feedback_addendum = "\n".join(remaining_alignment)
            review_feedback = (
                f"{review_feedback}\n{review_feedback_addendum}".strip()
                if review_feedback else review_feedback_addendum
            )
            logging.warning(
                "Skeleton review retry for component '%s' (%d/%d): %s",
                component_name,
                attempt_idx + 1,
                review_attempts,
                "; ".join(remaining_alignment),
            )
        return reviewed_code

    def _apply_test_review_with_retries(
        self,
        *,
        component_name: str,
        responsibilities: List[str],
        module_qualname: str,
        planned_file_path: str,
        skeleton_code: str,
        test_file_path: str,
        test_code: str,
        implemented_components_context: str = "",
        previous_attempt_feedback: str = "",
    ) -> str:
        reviewed_test = test_code
        review_feedback = str(previous_attempt_feedback or "").strip()
        review_agent = self._get_test_review_agent()
        review_attempts = max(1, self.test_review_max_retries + 1)
        for attempt_idx in range(review_attempts):
            patch_result = review_agent.review_test(
                component_name=component_name,
                responsibilities=responsibilities,
                module_qualname=module_qualname,
                planned_file_path=planned_file_path,
                skeleton_code=skeleton_code,
                test_file_path=test_file_path,
                test_code=reviewed_test,
                implemented_components_context=implemented_components_context,
                previous_attempt_feedback=review_feedback,
            )
            reviewed_candidate = review_agent.extract_reviewed_test_code(
                patch_result=patch_result,
                test_file_path=test_file_path,
                fallback_test_code=reviewed_test,
            )
            reviewed_test = reviewed_candidate.strip() or reviewed_test
            alignment_issues = self._find_test_notimplemented_alignment_issues(
                skeleton_code=skeleton_code,
                test_code=reviewed_test,
                rel_test=test_file_path,
            )
            if not alignment_issues:
                return reviewed_test
            if attempt_idx + 1 >= review_attempts:
                raise RuntimeError(
                    "Test generation/review left placeholder-behavior assertions for concrete APIs: "
                    + "; ".join(alignment_issues)
                )
            review_feedback_addendum = "\n".join(alignment_issues)
            review_feedback = (
                f"{review_feedback}\n{review_feedback_addendum}".strip()
                if review_feedback else review_feedback_addendum
            )
            logging.warning(
                "Test review retry for component '%s' (%d/%d): %s",
                component_name,
                attempt_idx + 1,
                review_attempts,
                "; ".join(alignment_issues),
            )
        return reviewed_test

    @staticmethod
    def _log_response_summary(component_name: str, response: Any) -> None:
        summary = CodeGeneratorAgent._summarize_response_payload(response)
        if summary is None:
            logging.info(
                "Generated code response summary for component '%s': type=%s",
                component_name,
                type(response).__name__,
            )
            return
        logging.info(
            "Generated code response summary for component '%s': file_path=%s code_len=%d test_len=%d doc_len=%d keys=%s",
            component_name,
            summary["file_path"],
            summary["code_len"],
            summary["test_len"],
            summary["doc_len"],
            summary["keys"],
        )

    @staticmethod
    def _summarize_response_payload(response: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(response, dict):
            return None
        code = str(response.get("code", "") or "")
        tests = response.get("tests", {}) if isinstance(response.get("tests"), dict) else {}
        test_code = str(response.get("test_code", "") or tests.get("test_code", "") or "")
        documentation = str(response.get("documentation", "") or "")
        return {
            "file_path": str(response.get("file_path", "") or response.get("test_file_path", "")),
            "code_len": len(code),
            "test_len": len(test_code),
            "doc_len": len(documentation),
            "keys": sorted(response.keys()),
        }

    @staticmethod
    def _summarize_failure_text(text: str, *, limit: int = 1200, tail_lines: int = 18) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        lines = [line.rstrip() for line in raw.splitlines() if line.strip()]
        if tail_lines > 0 and len(lines) > tail_lines:
            lines = lines[-tail_lines:]
        summary = "\n".join(lines)
        if len(summary) > limit:
            summary = summary[-limit:]
        return summary

    @staticmethod
    def _is_infra_or_parsing_error(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        signals = (
            "timeout",
            "timed out",
            "connection",
            "network",
            "temporarily unavailable",
            "service unavailable",
            "rate limit",
            "429",
            "502",
            "503",
            "504",
            "apierror",
            "transport",
            "invalid \\escape",
            "json parse",
            "json parsing",
            "malformed json",
            "failed to parse",
            "non-object json",
            "decode error",
            "patchagent",
            "tool error",
        )
        return any(signal in text for signal in signals)

    @staticmethod
    def _rel_path_to_module_qualname(rel_file_path: str) -> str:
        p = Path(str(rel_file_path).replace("\\", "/").strip("/"))
        if p.name == "__init__.py":
            p = p.parent
        elif p.suffix == ".py":
            p = p.with_suffix("")
        parts = [x for x in p.parts if x]
        return ".".join(parts) if parts else "module"

    def _run_saved_python_import_postcheck(
        self,
        *,
        repo_root: Path,
        module_name: str,
    ) -> Tuple[bool, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "").strip(os.pathsep)
        script = """import importlib
import sys
import traceback

repo_root = sys.argv[1]
module_name = sys.argv[2]
sys.path.insert(0, repo_root)
importlib.invalidate_caches()

try:
    importlib.import_module(module_name)
    print("POSTCHECK_OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script, str(repo_root), module_name],
                capture_output=True,
                text=True,
                timeout=max(30, self.tdd_pytest_timeout),
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return False, f"Postcheck import timed out after {max(30, self.tdd_pytest_timeout)}s: {exc}"
        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        return proc.returncode == 0, output

    @staticmethod
    def _extract_repo_python_paths_from_postcheck_output(
        output: str,
        repo_root: Path,
    ) -> List[str]:
        root = str(repo_root.resolve())
        matches = re.findall(r'File "([^"]+\.py)"', output or "")
        rel_paths: List[str] = []
        for raw in matches:
            try:
                path = Path(raw).resolve()
                rel = path.relative_to(repo_root.resolve())
            except Exception:
                continue
            rel_text = str(rel).replace("\\", "/")
            if rel_text not in rel_paths:
                rel_paths.append(rel_text)
        for raw in re.findall(r"(/[^\s:]+\.py)", output or ""):
            if not raw.startswith(root):
                continue
            try:
                rel = str(Path(raw).resolve().relative_to(repo_root.resolve())).replace("\\", "/")
            except Exception:
                continue
            if rel not in rel_paths:
                rel_paths.append(rel)
        return rel_paths[:8]

    @staticmethod
    def _is_import_conflict_output(output: str) -> bool:
        text = str(output or "")
        conflict_patterns = [
            r"partially initialized module",
            r"circular import",
            r"most likely due to a circular import",
            r"cannot import name .* from partially initialized module",
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in conflict_patterns)

    def postcheck_saved_component(
        self,
        *,
        code_result: Dict[str, Any],
        repo_root: str | Path,
        created_files: Dict[str, str],
        implemented_components_context: str = "",
        allowed_write_rel_paths: Optional[Set[str]] = None,
        max_fix_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        postcheck_started_at = time.perf_counter()
        component_name = str(code_result.get("component_name", "Component"))
        rel_impl = str(code_result.get("file_path", "")).strip()
        if not rel_impl.endswith(".py"):
            return {"enabled": False, "passed": True, "module": "", "attempts": 0, "error": ""}

        repo_root_path = Path(repo_root).resolve()
        max_fix_attempts = (
            self.import_postcheck_max_fix_attempts
            if max_fix_attempts is None else max(0, int(max_fix_attempts))
        )
        module_name = self._rel_path_to_module_qualname(rel_impl)
        logging.info(
            "Component import postcheck starting for '%s': module=%s path=%s",
            component_name,
            module_name,
            rel_impl,
        )
        postcheck_output = ""
        for attempt in range(1, max_fix_attempts + 2):
            ok, postcheck_output = self._run_saved_python_import_postcheck(
                repo_root=repo_root_path,
                module_name=module_name,
            )
            if ok:
                logging.info(
                    "Component postcheck passed for '%s' via import smoke: %s",
                    component_name,
                    module_name,
                )
                self._record_codegen_timing_event(
                    component_name=component_name,
                    stage="component_import_postcheck",
                    started_at_perf=postcheck_started_at,
                    meta={"passed": True, "module": module_name, "attempts": attempt},
                )
                return {
                    "enabled": True,
                    "passed": True,
                    "module": module_name,
                    "attempts": attempt,
                    "error": "",
                }

            if self._is_import_conflict_output(postcheck_output):
                failure_summary = self._summarize_failure_text(postcheck_output)
                logging.warning(
                    "Component postcheck detected import conflict for '%s' on attempt %d; "
                    "treating it as repairable and continuing targeted patch flow. Failure summary:\n%s",
                    component_name,
                    attempt,
                    failure_summary or "<empty>",
                )

            if attempt > max_fix_attempts:
                break

            related_rel_paths = [rel_impl]
            test_rel = ""
            if created_files.get("test"):
                try:
                    test_rel = str(Path(created_files["test"]).resolve().relative_to(repo_root_path)).replace("\\", "/")
                except Exception:
                    test_rel = ""
            if test_rel:
                related_rel_paths.append(test_rel)
            for rel_path in self._extract_repo_python_paths_from_postcheck_output(postcheck_output, repo_root_path):
                if rel_path not in related_rel_paths:
                    related_rel_paths.append(rel_path)

            import_fix_agent = self._get_import_postcheck_fix_agent()
            related_files = import_fix_agent.collect_related_files(
                repo_root=repo_root_path,
                file_paths=related_rel_paths[:6],
                max_file_chars=20000,
            )
            if rel_impl not in related_files:
                impl_path = repo_root_path / rel_impl
                if impl_path.exists():
                    related_files[rel_impl] = impl_path.read_text(encoding="utf-8")
            if test_rel and test_rel not in related_files:
                test_path = repo_root_path / test_rel
                if test_path.exists():
                    related_files[test_rel] = test_path.read_text(encoding="utf-8")

            patch = import_fix_agent.fix_import_failure(
                component_name=str(code_result.get("component_name", "Component")),
                module_name=module_name,
                planned_file_path=rel_impl,
                import_error=postcheck_output,
                related_files=related_files,
                implemented_components_context=implemented_components_context,
            )
            updated_files = patch.get("updated_files", {}) if isinstance(patch, dict) else {}
            if not isinstance(updated_files, dict) or not updated_files:
                break

            if allowed_write_rel_paths is not None:
                filtered_updated_files: Dict[str, str] = {}
                for rel_path, content in updated_files.items():
                    normalized_rel = str(rel_path).replace("\\", "/").lstrip("./")
                    if normalized_rel in allowed_write_rel_paths:
                        filtered_updated_files[normalized_rel] = str(content)
                dropped = sorted(set(updated_files.keys()) - set(filtered_updated_files.keys()))
                if dropped:
                    logging.warning(
                        "Import postcheck patch for '%s' proposed out-of-scope file updates; dropped: %s",
                        component_name,
                        ", ".join(dropped[:10]),
                    )
                updated_files = filtered_updated_files
                if not updated_files:
                    break

            sanitized_files: Dict[str, str] = {}
            for rel_path, content in updated_files.items():
                text = str(content)
                if str(rel_path).endswith(".py"):
                    text = self._autofix_python_syntax(
                        text,
                        component_name=str(code_result.get("component_name", "Component")),
                        rel_path=str(rel_path),
                    )
                    remaining = self._find_python_placeholder_issues(text, str(rel_path))
                    if remaining:
                        raise RuntimeError(
                            "Postcheck patch introduced unresolved placeholders: "
                            + "; ".join(sorted(set(remaining)))
                        )
                sanitized_files[str(rel_path)] = text

            import_fix_agent.write_files(repo_root_path, sanitized_files)
            failure_summary = self._summarize_failure_text(postcheck_output)
            logging.warning(
                "Component postcheck failed for '%s' on attempt %d; applied targeted patch and retrying import smoke. Failure summary:\n%s",
                component_name,
                attempt,
                failure_summary or "<empty>",
            )

        self._record_codegen_timing_event(
            component_name=component_name,
            stage="component_import_postcheck",
            started_at_perf=postcheck_started_at,
            status="failed",
            meta={"passed": False, "module": module_name, "attempts": max_fix_attempts + 1},
        )
        raise RuntimeError(
            f"Component postcheck failed for module '{module_name}': {postcheck_output[-4000:]}"
        )

    def postcheck_package_modules(
        self,
        *,
        package_modules: List[str],
        repo_root: str | Path,
        implemented_components_context: str = "",
        max_fix_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        repo_root_path = Path(repo_root).resolve()
        max_fix_attempts = (
            self.package_postcheck_max_fix_attempts
            if max_fix_attempts is None else max(0, int(max_fix_attempts))
        )
        normalized_modules = [str(name).strip() for name in package_modules if str(name).strip()]
        seen: Set[str] = set()
        ordered_modules: List[str] = []
        for name in normalized_modules:
            if name not in seen:
                seen.add(name)
                ordered_modules.append(name)

        results: List[Dict[str, Any]] = []
        for module_name in ordered_modules:
            logging.info("Parent/package import postcheck starting: module=%s", module_name)
            postcheck_output = ""
            for attempt in range(1, max_fix_attempts + 2):
                ok, postcheck_output = self._run_saved_python_import_postcheck(
                    repo_root=repo_root_path,
                    module_name=module_name,
                )
                if ok:
                    results.append(
                        {
                            "module": module_name,
                            "passed": True,
                            "attempts": attempt,
                            "error": "",
                        }
                    )
                    break
                if self._is_import_conflict_output(postcheck_output):
                    failure_summary = self._summarize_failure_text(postcheck_output)
                    logging.warning(
                        "Parent/package import postcheck detected import conflict for '%s' on attempt %d; "
                        "treating it as repairable and continuing targeted patch flow. Failure summary:\n%s",
                        module_name,
                        attempt,
                        failure_summary or "<empty>",
                    )
                if attempt > max_fix_attempts:
                    results.append(
                        {
                            "module": module_name,
                            "passed": False,
                            "attempts": attempt,
                            "error": postcheck_output[-4000:],
                        }
                    )
                    raise RuntimeError(
                        f"Parent/package import postcheck failed for '{module_name}': {postcheck_output[-4000:]}"
                    )

                import_fix_agent = self._get_import_postcheck_fix_agent()
                related_rel_paths = self._extract_repo_python_paths_from_postcheck_output(
                    postcheck_output,
                    repo_root_path,
                )
                init_rel = module_name.replace(".", "/") + "/__init__.py"
                if (repo_root_path / init_rel).exists() and init_rel not in related_rel_paths:
                    related_rel_paths.insert(0, init_rel)
                related_files = import_fix_agent.collect_related_files(
                    repo_root=repo_root_path,
                    file_paths=related_rel_paths[:8],
                    max_file_chars=20000,
                )
                patch = import_fix_agent.fix_import_failure(
                    component_name=f"package:{module_name}",
                    module_name=module_name,
                    planned_file_path=init_rel,
                    import_error=postcheck_output,
                    related_files=related_files,
                    implemented_components_context=implemented_components_context,
                )
                updated_files = patch.get("updated_files", {}) if isinstance(patch, dict) else {}
                if not isinstance(updated_files, dict) or not updated_files:
                    results.append(
                        {
                            "module": module_name,
                            "passed": False,
                            "attempts": attempt,
                            "error": postcheck_output[-4000:],
                        }
                    )
                    break

                sanitized_files: Dict[str, str] = {}
                for rel_path, content in updated_files.items():
                    text = str(content)
                    if str(rel_path).endswith(".py"):
                        text = self._autofix_python_syntax(
                            text,
                            component_name=f"package:{module_name}",
                            rel_path=str(rel_path),
                        )
                    sanitized_files[str(rel_path)] = text
                import_fix_agent.write_files(repo_root_path, sanitized_files)
                failure_summary = self._summarize_failure_text(postcheck_output)
                logging.warning(
                    "Parent/package postcheck failed for '%s' on attempt %d; applied targeted patch and retrying. Failure summary:\n%s",
                    module_name,
                    attempt,
                    failure_summary or "<empty>",
                )

        return {
            "enabled": True,
            "modules": results,
            "passed": all(item.get("passed") for item in results),
        }

    def _ensure_pkg_inits_under_root(self, root: Path, rel_file: str) -> None:
        rel = Path(str(rel_file).replace("\\", "/"))
        parent = rel.parent
        while parent.parts:
            init_path = root / parent / "__init__.py"
            if not init_path.exists():
                init_path.parent.mkdir(parents=True, exist_ok=True)
                init_path.write_text("", encoding="utf-8")
            parent = parent.parent

    def _write_sandbox_sources(self, root: Path, files: Dict[str, str]) -> None:
        for rel, content in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self._ensure_pkg_inits_under_root(root, rel)

    def _extract_context_file_paths(self, implemented_components_context: str) -> List[str]:
        rel_paths: List[str] = []
        seen: Set[str] = set()
        if not implemented_components_context:
            return rel_paths
        for raw_line in implemented_components_context.splitlines():
            line = raw_line.strip()
            if not line.startswith("File: "):
                continue
            rel_path = line[len("File: ") :].strip().replace("\\", "/")
            if not rel_path.endswith(".py"):
                continue
            if rel_path in seen:
                continue
            seen.add(rel_path)
            rel_paths.append(rel_path)
        return rel_paths

    def _write_sandbox_context_sources(self, root: Path, implemented_components_context: str) -> int:
        if not implemented_components_context:
            return 0
        if self._tdd_pip_project_root is None or not self._tdd_pip_project_root.is_dir():
            return 0
        copied = 0
        for rel_path in self._extract_context_file_paths(implemented_components_context):
            if rel_path in {".", ""}:
                continue
            src = self._tdd_pip_project_root / rel_path
            if not src.is_file():
                continue
            dst = root / rel_path
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            self._ensure_pkg_inits_under_root(root, rel_path)
            copied += 1
        return copied

    @staticmethod
    def _tdd_has_project_metadata(root: Path) -> bool:
        return any((root / name).is_file() for name in ("setup.py", "pyproject.toml", "setup.cfg"))

    def _tdd_needs_pip_prepare(self) -> bool:
        if self._tdd_pip_project_root is None:
            return False
        if self._tdd_has_project_metadata(self._tdd_pip_project_root):
            return True
        return False

    def _run_tdd_pip_prepare_local(self) -> Tuple[bool, str]:
        """Install editable project (if metadata exists) into the current interpreter."""
        chunks: List[str] = []
        pip = [sys.executable, "-m", "pip", "install", "--no-input", "-q"]
        timeout = self.tdd_pip_timeout

        def run_pip(extra_args: List[str]) -> bool:
            proc = subprocess.run(
                pip + extra_args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            chunks.append((proc.stdout or "") + "\n" + (proc.stderr or ""))
            return proc.returncode == 0

        if (
            self._tdd_pip_project_root is not None
            and self._tdd_has_project_metadata(self._tdd_pip_project_root)
        ):
            if not run_pip(["-e", str(self._tdd_pip_project_root)]):
                return False, "".join(chunks)

        return True, "".join(chunks)

    def _pip_install_specs_local(self, specs: List[str]) -> Tuple[bool, str]:
        """Install pip specs into the current interpreter (TDD heuristic / error-driven)."""
        chunks: List[str] = []
        pip = [sys.executable, "-m", "pip", "install", "--no-input", "-q"]
        for spec in specs:
            safe = str(spec).strip()
            if not safe:
                continue
            try:
                proc = subprocess.run(
                    pip + [safe],
                    capture_output=True,
                    text=True,
                    timeout=self.tdd_pip_timeout,
                )
            except subprocess.TimeoutExpired as te:
                return False, f"pip timeout after {self.tdd_pip_timeout}s: {safe}\n{te}\n" + "".join(
                    chunks
                )
            except FileNotFoundError:
                return False, "pip/python not executable\n" + "".join(chunks)
            chunks.append((proc.stdout or "") + "\n" + (proc.stderr or ""))
            if proc.returncode != 0:
                return False, "".join(chunks)
        return True, "".join(chunks)

    def _run_pytest_local_plain(self, root: Path, test_rel: str) -> Tuple[int, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "").strip(os.pathsep)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    test_rel,
                    "-q",
                    "--tb=short",
                ],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=self.tdd_pytest_timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as te:
            return 124, f"pytest timeout after {self.tdd_pytest_timeout}s\n{te}"
        except FileNotFoundError:
            return 127, "pytest/python not executable"
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return int(proc.returncode), out

    def _run_pytest_in_sandbox(
        self,
        root: Path,
        test_rel: str,
        *,
        heuristic_pip_specs: Optional[List[str]] = None,
    ) -> Tuple[int, str]:
        """Run pytest only. Heuristic installs run in the caller (local) or inside Docker."""
        if self.tdd_docker_image:
            return self._run_pytest_in_docker(root, test_rel, heuristic_pip_specs=heuristic_pip_specs)
        return self._run_pytest_local_plain(root, test_rel)

    def _run_pytest_in_docker(
        self,
        root: Path,
        test_rel: str,
        *,
        heuristic_pip_specs: Optional[List[str]] = None,
    ) -> Tuple[int, str]:
        """Run pytest inside ``tdd_docker_image`` with the sandbox tree mounted at ``/work``."""
        if not shutil.which("docker"):
            return 126, "docker CLI not found on PATH; install Docker or unset tdd_docker_image"
        root_abs = root.resolve()
        cmd: List[str] = ["docker", "run", "--rm"]
        if self.tdd_docker_network_host:
            cmd.extend(["--network", "host"])
        cmd.extend(
            [
                "-v",
                f"{root_abs}:/work",
                "-w",
                "/work",
                "-e",
                "PYTHONPATH=/work",
            ]
        )
        shell_parts: List[str] = []
        # Keep the virtualenv on the container-local filesystem.
        # Creating it on the bind-mounted /work tree can trigger EPERM on some hosts.
        shell_parts.extend(
            [
                "export PIP_ROOT_USER_ACTION=ignore",
                "mkdir -p /tmp/.tdd_state",
                "if [ ! -x /tmp/.tdd_venv/bin/python ]; then python -m venv --system-site-packages /tmp/.tdd_venv; fi",
                ". /tmp/.tdd_venv/bin/activate",
            ]
        )
        if self._tdd_pip_project_root is not None and self._tdd_pip_project_root.is_dir():
            cmd.extend(["-v", f"{self._tdd_pip_project_root.resolve()}:/project"])
            if self._tdd_has_project_metadata(self._tdd_pip_project_root):
                shell_parts.extend(
                    [
                        (
                            "if [ ! -f /tmp/.tdd_state/project_editable_ready ]; then "
                            "python -m pip install --no-input -q -e /project && "
                            "touch /tmp/.tdd_state/project_editable_ready; "
                            "fi"
                        ),
                    ]
                )
        for spec in heuristic_pip_specs or []:
            safe = str(spec).strip()
            if safe:
                shell_parts.append(
                    (
                        f"if [ ! -f /tmp/.tdd_state/{self._tdd_spec_marker_name(safe)} ]; then "
                        "python -m pip install --no-input -q "
                        + shlex.quote(safe)
                        + f" && touch /tmp/.tdd_state/{self._tdd_spec_marker_name(safe)}; fi"
                    )
                )
        shell_parts.append(
            "python -m pytest -p no:cacheprovider "
            + shlex.quote(test_rel)
            + " -q --tb=short"
        )
        inner = "set -e && " + " && ".join(shell_parts)
        cmd.extend([self.tdd_docker_image, "sh", "-c", inner])
        hlen = len(heuristic_pip_specs or [])
        deadline = self.tdd_pytest_timeout + (
            self.tdd_pip_timeout if self._tdd_needs_pip_prepare() else 0
        ) + hlen * self.tdd_pip_timeout
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=deadline,
            )
        except subprocess.TimeoutExpired as te:
            return 124, f"pytest (docker) timeout after {deadline}s\n{te}"
        except FileNotFoundError:
            return 126, "docker run failed (docker not executable)"
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return int(proc.returncode), out

    @staticmethod
    def _tdd_spec_marker_name(spec: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(spec).strip()).strip("._-")
        return f"pip_{safe or 'spec'}_ready"

    def _autofix_pair_before_pytest(
        self,
        *,
        impl_body: str,
        test_body: str,
        rel_impl: str,
        rel_test: str,
        component_name: str,
    ) -> Tuple[str, str]:
        """Run the same syntax static repair as ``save_generated_code`` before each pytest run."""
        if not self.enable_syntax_autofix:
            return impl_body, test_body
        impl_out = (
            self._autofix_python_syntax(
                impl_body,
                component_name=component_name,
                rel_path=rel_impl,
            )
            if str(rel_impl).lower().endswith(".py")
            else impl_body
        )
        test_out = (
            self._autofix_python_syntax(
                test_body,
                component_name=component_name,
                rel_path=rel_test,
            )
            if str(rel_test).lower().endswith(".py")
            else test_body
        )
        return impl_out, test_out

    def _tdd_fix_loop(
        self,
        *,
        rel_impl: str,
        impl_code: str,
        rel_test: str,
        test_code: str,
        component_name: str,
        implemented_components_context: str = "",
        previous_attempt_feedback: str = "",
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Run pytest in a temp tree; on failure apply PatchAgent up to ``tdd_max_fix_retries`` times."""
        tdd_started_at = time.perf_counter()
        patch_agent = self._get_patch_agent()
        rel_test = self._normalize_test_file_path(rel_test, component_name)
        meta: Dict[str, Any] = {"pytest_attempts": [], "patch_rounds": 0, "final_pytest_rc": None}
        impl_body = impl_code
        test_body = test_code
        installed_local: Set[str] = set()
        accumulated: List[str] = []
        seen_spec: Set[str] = set()

        if not self.tdd_docker_image and self._tdd_needs_pip_prepare():
            ok_pip, pip_log = self._run_tdd_pip_prepare_local()
            if not ok_pip:
                logging.warning(
                    "TDD pip prepare failed before pytest (local): %s",
                    pip_log[-4000:],
                )
                tail = pip_log[-12000:] if len(pip_log) > 12000 else pip_log
                meta["pytest_attempts"].append(
                    {
                        "round": -1,
                        "rc": 125,
                        "output": tail,
                        "backend": "pip_prepare",
                        "docker_image": None,
                    }
                )
                meta["final_pytest_rc"] = 125
                self._record_codegen_timing_event(
                    component_name=component_name,
                    stage="tdd_fix_loop",
                    started_at_perf=tdd_started_at,
                    status="failed",
                    meta={"final_pytest_rc": 125, "patch_rounds": 0, "pytest_attempts": 1, "reason": "pip_prepare_failed"},
                )
                return impl_body, test_body, meta

        with tempfile.TemporaryDirectory(prefix="codegen_tdd_") as tmp:
            root = Path(tmp)
            for round_idx in range(self.tdd_max_fix_retries + 1):
                impl_body, test_body = self._autofix_pair_before_pytest(
                    impl_body=impl_body,
                    test_body=test_body,
                    rel_impl=rel_impl,
                    rel_test=rel_test,
                    component_name=component_name,
                )
                root = Path(tmp)
                self._write_sandbox_sources(root, {rel_impl: impl_body, rel_test: test_body})
                copied_context_files = self._write_sandbox_context_sources(
                    root,
                    implemented_components_context,
                )
                local_top = sandbox_top_level_names(root, (rel_impl, rel_test))
                static_specs = specs_from_sources_and_sandbox(
                    impl_body, test_body, root, rel_impl, rel_test
                )
                for s in static_specs:
                    if s not in seen_spec:
                        seen_spec.add(s)
                        accumulated.append(s)
                if self._tdd_pip_project_root is not None and self._tdd_pip_project_root.is_dir():
                    project_specs = specs_from_project_import_closure(
                        self._tdd_pip_project_root,
                        (rel_impl, rel_test),
                    )
                    added_project_specs = [s for s in project_specs if s not in seen_spec]
                    if added_project_specs:
                        for s in added_project_specs:
                            seen_spec.add(s)
                            accumulated.append(s)
                        logging.info(
                            "TDD: preinstalling transitive project dependency specs for '%s': %s",
                            component_name,
                            added_project_specs,
                        )
                import_pass = 0
                rc = 1
                log = ""
                while True:
                    if not self.tdd_docker_image:
                        to_add = [s for s in accumulated if s not in installed_local]
                        if to_add:
                            ok_h, plog = self._pip_install_specs_local(to_add)
                            if not ok_h:
                                logging.warning(
                                    "TDD heuristic pip install failed: %s",
                                    plog[-4000:],
                                )
                                rc = 125
                                log = plog
                                tail = plog[-12000:] if len(plog) > 12000 else plog
                                meta["pytest_attempts"].append(
                                    {
                                        "round": round_idx,
                                        "rc": rc,
                                        "output": tail,
                                        "import_pass": import_pass,
                                        "heuristic_pip_specs": list(accumulated),
                                        "backend": "local",
                                        "docker_image": None,
                                        "phase": "heuristic_pip",
                                    }
                                )
                                meta["final_pytest_rc"] = rc
                                break
                            installed_local.update(to_add)

                    heur_for_docker: Optional[List[str]] = (
                        list(accumulated) if self.tdd_docker_image else None
                    )
                    rc, log = self._run_pytest_in_sandbox(
                        root,
                        rel_test,
                        heuristic_pip_specs=heur_for_docker,
                    )
                    tail = log[-12000:] if len(log) > 12000 else log
                    meta["pytest_attempts"].append(
                        {
                            "round": round_idx,
                            "rc": rc,
                            "output": tail,
                            "import_pass": import_pass,
                            "heuristic_pip_specs": list(accumulated),
                            "context_files_copied": copied_context_files,
                            "backend": "docker" if self.tdd_docker_image else "local",
                            "docker_image": self.tdd_docker_image or None,
                            "phase": "pytest",
                        }
                    )
                    meta["final_pytest_rc"] = rc
                    if rc == 0:
                        return impl_body, test_body, meta

                    logging.warning(
                        "TDD pytest failed for component '%s' round=%d rc=%s backend=%s. Failure summary:\n%s",
                        component_name,
                        round_idx + 1,
                        rc,
                        "docker" if self.tdd_docker_image else "local",
                        self._summarize_failure_text(tail, limit=1600, tail_lines=24) or "<empty>",
                    )

                    missing_roots = missing_import_roots_from_pytest_log(log)
                    new_specs = specs_from_missing_roots(missing_roots, local=local_top)
                    added = False
                    if new_specs and import_pass < self.tdd_missing_module_pip_retries:
                        for spec in new_specs:
                            if spec not in seen_spec:
                                seen_spec.add(spec)
                                accumulated.append(spec)
                                added = True
                        if added:
                            import_pass += 1
                            logging.info(
                                "TDD: retrying pytest after pip install (missing modules %s, pass %s)",
                                missing_roots,
                                import_pass,
                            )
                            continue

                    break

                if round_idx >= self.tdd_max_fix_retries:
                    logging.warning(
                        "TDD fix loop exhausted (%d rounds); last pytest rc=%s",
                        self.tdd_max_fix_retries,
                        rc,
                    )
                    self._record_codegen_timing_event(
                        component_name=component_name,
                        stage="tdd_fix_loop",
                        started_at_perf=tdd_started_at,
                        status="failed",
                        meta={
                            "final_pytest_rc": rc,
                            "patch_rounds": int(meta.get("patch_rounds", 0)),
                            "pytest_attempts": len(meta.get("pytest_attempts", [])),
                        },
                    )
                    return impl_body, test_body, meta

                tail = log[-12000:] if len(log) > 12000 else log
                related = {rel_impl: impl_body, rel_test: test_body}
                prior_feedback_block = (
                    "\nPrevious attempt feedback:\n" + previous_attempt_feedback.strip()
                    if previous_attempt_feedback.strip() else ""
                )
                proposal = patch_agent.generate_patch(
                    task_description=(
                        f"Pytest failed for component {component_name}. "
                        f"Fix implementation and/or tests so tests pass."
                        f"{prior_feedback_block}"
                    ),
                    related_files=related,
                    compile_error=tail,
                    incremental_goal=(
                        "Prefer fixing implementation to match tests; "
                        "only change tests for incorrect imports or API mismatches with skeleton. "
                        f"TDD patch round {round_idx + 1}/{self.tdd_max_fix_retries + 1}. "
                        f"Latest pytest failure summary:\n{tail[-2000:]}"
                    ),
                    failure_kind="test_failure",
                    telemetry_context={
                        "component_name": component_name,
                        "stage": "tdd_fix_loop",
                        "round": round_idx + 1,
                        "file_role": "impl_and_test",
                    },
                )
                updated = dict(related)
                uf = proposal.get("updated_files")
                if isinstance(uf, dict):
                    for k, v in uf.items():
                        if isinstance(v, str):
                            updated[str(k)] = v
                impl_body = updated.get(rel_impl, impl_body)
                test_body = updated.get(rel_test, test_body)
                meta["patch_rounds"] = int(meta.get("patch_rounds", 0)) + 1

        self._record_codegen_timing_event(
            component_name=component_name,
            stage="tdd_fix_loop",
            started_at_perf=tdd_started_at,
            status="completed",
            meta={
                "final_pytest_rc": meta.get("final_pytest_rc"),
                "patch_rounds": int(meta.get("patch_rounds", 0)),
                "pytest_attempts": len(meta.get("pytest_attempts", [])),
            },
        )
        return impl_body, test_body, meta

    @staticmethod
    def _summarize_tdd_meta(meta: Dict[str, Any]) -> str:
        if not isinstance(meta, dict):
            return "no TDD metadata available"
        attempts = meta.get("pytest_attempts", [])
        final_rc = meta.get("final_pytest_rc")
        if not isinstance(attempts, list) or not attempts:
            return f"final_pytest_rc={final_rc}"
        last_attempt = attempts[-1] if isinstance(attempts[-1], dict) else {}
        phase = last_attempt.get("phase", "unknown")
        backend = last_attempt.get("backend", "unknown")
        rc = last_attempt.get("rc", final_rc)
        output = str(last_attempt.get("output", "") or "").strip()
        output_tail = output[-600:] if len(output) > 600 else output
        summary = f"final_pytest_rc={rc}, phase={phase}, backend={backend}"
        if output_tail:
            summary += f", last_output_tail={output_tail}"
        return summary

    def _generate_code_skeleton_tdd(
        self,
        component: Dict[str, Any],
        requirement: Dict[str, Any],
        architecture: Dict[str, Any],
        *,
        implemented_components_context: str = "",
        planned_file_path: Optional[str] = None,
        language: str = "python",
        previous_attempt_feedback: str = "",
    ) -> Dict[str, Any]:
        assert self.llm_client is not None
        total_started_at = time.perf_counter()
        component_name = component.get("name", "Component")
        responsibilities = component.get("responsibilities", [])
        req_name = requirement.get("name", "Requirement")
        req_description = requirement.get("description", "")
        architecture_rationale = architecture.get("rationale", "")
        dag_summary = architecture.get("dag_summary", {})
        parent_node = requirement.get("parent_node")
        parent_prev_node = requirement.get("parent_prev_node")

        other_components = [
            comp for comp in architecture.get("components", [])
            if comp.get("name") != component_name
        ]
        context_components = "\n".join([
            f"- {comp.get('name')}: {', '.join(comp.get('responsibilities', []))}"
            for comp in other_components[:5]
        ])
        req_dependencies = requirement.get("dependencies", [])
        req_metadata = requirement.get("metadata", {})
        planned_path_instruction = (
            f"\nPlanned File Path (MUST use exactly this path): {planned_file_path}"
            if planned_file_path else ""
        )
        action_guidance_block = self._action_guidance_block(component)
        previous_feedback_block = (
            "\n=== PREVIOUS ATTEMPT FEEDBACK ===\n"
            f"{previous_attempt_feedback}\n"
            "Do not repeat these failure modes. Fix them directly in this attempt.\n"
            if previous_attempt_feedback else ""
        )

        # --- Phase 1: skeleton ---
        skeleton_prompt = f"""You are an expert Python engineer. Produce a SKELETON module only (phase 1 of 2).

=== PROJECT REQUIREMENT ===
Name: {req_name}
Description: {req_description}
Dependencies: {', '.join(req_dependencies) if req_dependencies else 'None'}
{f"Additional Context: {req_metadata}" if req_metadata else ""}

=== ARCHITECTURE DESIGN ===
Rationale: {architecture_rationale}
{f"DAG Context: {dag_summary.get('node_count', 0)} requirements, {dag_summary.get('edge_count', 0)} dependencies" if dag_summary else ""}

=== PARENT CONTEXT ===
Parent Node: 
{json.dumps(parent_node, ensure_ascii=False) if parent_node else "None"}
Previous Parent Node: 
{json.dumps(parent_prev_node, ensure_ascii=False) if parent_prev_node else "None"}

=== COMPONENT ===
Name: {component_name}
Responsibilities:
{chr(10).join(f"{i+1}. {resp}" for i, resp in enumerate(responsibilities))}

=== RELATED COMPONENTS ===
{context_components if context_components else "None"}

{implemented_components_context}
{action_guidance_block}
{previous_feedback_block}

Rules for SKELETON:
{planned_path_instruction}
1. Include imports, module docstring, public classes/functions with full signatures and docstrings.
2. Method/function bodies MUST be minimal and explicit for TDD: use `raise NotImplementedError("TDD")` only for concrete members. Do not use `pass` or `...` as placeholders in concrete public members.
3. Preserve names and shapes implied by responsibilities; this API will be tested in phase 2.
4. Reuse names and import paths from IMPLEMENTED COMPONENTS exactly when needed, but keep skeleton importable on PYTHONPATH.
5. If you reference an implemented component, use the exact exported symbol and module path from IMPLEMENTED COMPONENTS or MODULE REGISTRY. Do not invent nearby names.
6. {self._state_contract_guidance_block()}
7. Every `class` and `def` declaration MUST be valid Python syntax and use only legal Python identifiers. Do not put `/`, `,`, `:` or prose fragments inside interface names.
8. Before returning, self-check that the module compiles and that every public interface line is a legal Python declaration.
9. Core algorithms must be designed and implemented inside this generated repository. {self._peer_repo_constraint_text(include_generic_utils_note=True)}

Return ONLY JSON: {{"file_path": "<path>.py", "code": "<full skeleton source>"}}"""

        logging.info("Codegen LLM request starting: phase=skeleton component=%s", component_name)
        skeleton_started_at = time.perf_counter()
        skel_resp = self.llm_client.call_json(
            [
                {"role": "system", "content": "You output strict JSON only. Python skeletons only."},
                {"role": "user", "content": skeleton_prompt},
            ],
            temperature=0.0,
            max_tokens=32768,
            max_retry_times=self.codegen_max_retry_times,
            timeout_seconds=self.codegen_timeout_seconds,
            operation_name=f"code_generator.skeleton:{component_name}",
        )
        self._record_codegen_timing_event(
            component_name=component_name,
            stage="skeleton_llm",
            started_at_perf=skeleton_started_at,
        )
        if not isinstance(skel_resp, dict):
            raise RuntimeError("skeleton phase returned non-object JSON")
        resolved_file_path = planned_file_path or skel_resp.get(
            "file_path", f"src/{self._to_snake_case(component_name)}.py"
        )
        resolved_file_path = self.normalize_file_path(resolved_file_path, language=language)
        skeleton_code = str(skel_resp.get("code", "")).strip()
        if not skeleton_code:
            raise RuntimeError("empty skeleton code")
        skeleton_review_started_at = time.perf_counter()
        skeleton_code = self._review_python_skeleton(
            component_name=component_name,
            rel_path=resolved_file_path,
            code=skeleton_code,
        )
        skeleton_code = self._apply_skeleton_review_with_retries(
            component_name=component_name,
            responsibilities=responsibilities,
            planned_file_path=resolved_file_path,
            skeleton_code=skeleton_code,
            implemented_components_context=implemented_components_context,
            previous_attempt_feedback=previous_attempt_feedback,
        )
        skeleton_code = self._repair_and_validate_python_interfaces(
            component_name=component_name,
            rel_path=resolved_file_path,
            code=skeleton_code,
            stage="skeleton_post_review",
        )
        self._record_codegen_timing_event(
            component_name=component_name,
            stage="skeleton_review",
            started_at_perf=skeleton_review_started_at,
        )

        module_q = self._rel_path_to_module_qualname(resolved_file_path)

        # --- Phase 2a: tests from skeleton ---
        test_prompt = f"""You write pytest for the following skeleton. Repo root will be on PYTHONPATH.

Module relative path: {resolved_file_path}
Import module as: ``import {module_q}`` or ``from {module_q} import ...`` as appropriate.

Component: {component_name}
Responsibilities:
{chr(10).join(f"- {r}" for r in responsibilities)}

Skeleton source:
```python
{skeleton_code}
```

Rules:
1. Use pytest style (functions or classes).
2. Tests may initially fail because the skeleton is not implemented yet, but do NOT assert that concrete APIs should keep raising `NotImplementedError` or other placeholder behavior in the final implementation.
3. Only expect `NotImplementedError` for explicitly abstract interfaces (for example ABC/Protocol/abstractmethod) that are meant to remain abstract.
4. Cover the main final public API implied by the skeleton and responsibilities, including return shapes, state changes, and integration behavior where possible.
5. test_file_path should be under ``tests/`` and use a stable name, e.g. ``tests/test_{self._to_snake_case(component_name)}.py``.
6. Treat the reviewed skeleton interfaces as the source of truth; do not invent alternate function/class names.
7. {self._peer_repo_constraint_text(include_generic_utils_note=False)}

Return ONLY JSON:
{{
  "test_file_path": "tests/test_....py",
  "test_code": "<full pytest file source>"
}}"""

        logging.info("Codegen LLM request starting: phase=test component=%s", component_name)
        test_started_at = time.perf_counter()
        test_resp = self.llm_client.call_json(
            [
                {"role": "system", "content": "You output strict JSON only. Valid pytest."},
                {"role": "user", "content": test_prompt},
            ],
            temperature=0.0,
            max_tokens=32768,
            max_retry_times=self.codegen_max_retry_times,
            timeout_seconds=self.codegen_timeout_seconds,
            operation_name=f"code_generator.test:{component_name}",
        )
        self._record_codegen_timing_event(
            component_name=component_name,
            stage="test_llm",
            started_at_perf=test_started_at,
        )
        if not isinstance(test_resp, dict):
            raise RuntimeError("test phase returned non-object JSON")
        test_rel = self._normalize_test_file_path(
            str(test_resp.get("test_file_path") or f"tests/test_{self._to_snake_case(component_name)}.py"),
            component_name=component_name,
        )
        test_code = str(test_resp.get("test_code", "")).strip()
        if not test_code:
            raise RuntimeError("empty test code")
        test_code = self._repair_and_validate_python_interfaces(
            component_name=component_name,
            rel_path=test_rel,
            code=test_code,
            stage="test_generation",
        )

        test_review_started_at = time.perf_counter()
        review_agent = self._get_test_review_agent()
        reviewed_test_code = review_agent.extract_reviewed_test_code(
            patch_result=review_agent.review_test(
                component_name=component_name,
                responsibilities=responsibilities,
                module_qualname=module_q,
                planned_file_path=resolved_file_path,
                skeleton_code=skeleton_code,
                test_file_path=test_rel,
                test_code=test_code,
                implemented_components_context=implemented_components_context,
                previous_attempt_feedback=previous_attempt_feedback,
            ),
            test_file_path=test_rel,
            fallback_test_code=test_code,
        )
        test_code = reviewed_test_code.strip() or test_code
        test_alignment_issues = self._find_test_notimplemented_alignment_issues(
            skeleton_code=skeleton_code,
            test_code=test_code,
            rel_test=test_rel,
        )
        if test_alignment_issues:
            review_feedback = "\n".join(test_alignment_issues)
            reviewed_test_code = review_agent.extract_reviewed_test_code(
                patch_result=review_agent.review_test(
                    component_name=component_name,
                    responsibilities=responsibilities,
                    module_qualname=module_q,
                    planned_file_path=resolved_file_path,
                    skeleton_code=skeleton_code,
                    test_file_path=test_rel,
                    test_code=test_code,
                    implemented_components_context=implemented_components_context,
                    previous_attempt_feedback=(
                        f"{previous_attempt_feedback}\n{review_feedback}".strip()
                        if previous_attempt_feedback else review_feedback
                    ),
                ),
                test_file_path=test_rel,
                fallback_test_code=test_code,
            )
            test_code = reviewed_test_code.strip() or test_code
            remaining_alignment_issues = self._find_test_notimplemented_alignment_issues(
                skeleton_code=skeleton_code,
                test_code=test_code,
                rel_test=test_rel,
            )
            if remaining_alignment_issues:
                raise RuntimeError(
                    "Test generation/review left placeholder-behavior assertions for concrete APIs: "
                    + "; ".join(remaining_alignment_issues)
                )
        test_code = self._repair_and_validate_python_interfaces(
            component_name=component_name,
            rel_path=test_rel,
            code=test_code,
            stage="test_post_review",
        )
        self._record_codegen_timing_event(
            component_name=component_name,
            stage="test_review",
            started_at_perf=test_review_started_at,
        )

        # --- Phase 2b: fill implementation ---
        fill_prompt = f"""You now FULLY implement the module so that the provided pytest passes.

Module file path: {resolved_file_path}
Import root qualification: {module_q}

Reviewed skeleton (this is the source of truth for the public API and intended behavior; replace bodies but keep API and documented behavior stable unless impossible):
```python
{skeleton_code}
```

Pytest file ({test_rel}):
```python
{test_code}
```

{implemented_components_context}
{previous_feedback_block}

Rules:
1. Return complete module source in key "code" only (single file).
2. Production-ready logic; no placeholder bodies remain in concrete public functions or methods. Eliminate every `raise NotImplementedError("TDD")`, `pass`, or `...` placeholder from concrete implementations.
3. Implement according to the reviewed skeleton's public API, docstrings, and stated semantics. Do not invent extra behavior from the raw responsibilities list.
4. Match imports and names from skeleton exactly.
5. Match import paths and symbol names from IMPLEMENTED COMPONENTS exactly. Do not invent new names for already implemented APIs.
6. If a dependency API is ambiguous, add a small local compatibility wrapper in this file rather than importing a symbol name that does not exist.
7. {self._state_contract_guidance_block()}
8. The final module must compile as valid Python. Every `class` and `def` declaration must use a legal Python identifier and a valid parameter list.
9. Implement the component's core algorithm locally in this repository or by reusing already generated local modules. {self._peer_repo_constraint_text(include_generic_utils_note=False)}

Return ONLY JSON: {{"code": "<full implemented source>"}}"""

        logging.info("Codegen LLM request starting: phase=fill component=%s", component_name)
        fill_started_at = time.perf_counter()
        fill_resp = self.llm_client.call_json(
            [
                {"role": "system", "content": "You output strict JSON only. Full Python implementation."},
                {"role": "user", "content": fill_prompt},
            ],
            temperature=0.0,
            max_tokens=32768,
            max_retry_times=self.codegen_max_retry_times,
            timeout_seconds=self.codegen_timeout_seconds,
            operation_name=f"code_generator.fill:{component_name}",
        )
        self._record_codegen_timing_event(
            component_name=component_name,
            stage="fill_llm",
            started_at_perf=fill_started_at,
        )
        if not isinstance(fill_resp, dict):
            raise RuntimeError("fill phase returned non-object JSON")
        filled_code = str(fill_resp.get("code", "")).strip()
        if not filled_code:
            raise RuntimeError("empty filled code")
        filled_code = self._repair_and_validate_python_interfaces(
            component_name=component_name,
            rel_path=resolved_file_path,
            code=filled_code,
            stage="fill_generation",
        )
        placeholder_started_at = time.perf_counter()
        filled_code, test_code, placeholder_meta = self._repair_placeholder_issues(
            component_name=component_name,
            responsibilities=responsibilities,
            rel_impl=resolved_file_path,
            impl_code=filled_code,
            rel_test=test_rel,
            test_code=test_code,
            implemented_components_context=implemented_components_context,
        )
        self._record_codegen_timing_event(
            component_name=component_name,
            stage="placeholder_repair",
            started_at_perf=placeholder_started_at,
            meta={
                "attempted": bool(placeholder_meta.get("attempted")),
                "remaining_hard_issues": len(placeholder_meta.get("remaining_hard_issues", [])),
            },
        )

        # --- Pre-TDD reconciliation: check responsibility realization before spending time on test repair ---
        pre_tdd_impl, pre_tdd_test = filled_code, test_code
        tdd_meta_out: Dict[str, Any] = {"placeholder_repair": placeholder_meta}
        try:
            pre_tdd_started_at = time.perf_counter()
            pre_tdd_impl, pre_tdd_test = self._postprocess_python_generation(
                component_name=component_name,
                responsibilities=responsibilities,
                rel_impl=resolved_file_path,
                impl_code=filled_code,
                rel_test=test_rel,
                test_code=test_code,
                implemented_components_context=implemented_components_context,
                stage="pre_tdd_reconciliation",
            )
            self._record_codegen_timing_event(
                component_name=component_name,
                stage="pre_tdd_postprocess",
                started_at_perf=pre_tdd_started_at,
                meta={"passed": True},
            )
            tdd_meta_out["pre_tdd_reconciliation_passed"] = True
        except Exception as exc:
            logging.warning(
                "Pre-TDD reconciliation failed for component '%s'; continuing into TDD loop with current artifacts. Error: %s",
                component_name,
                exc,
            )
            self._record_codegen_timing_event(
                component_name=component_name,
                stage="pre_tdd_postprocess",
                started_at_perf=pre_tdd_started_at,
                status="failed",
                meta={"passed": False, "error": str(exc)},
            )
            tdd_meta_out["pre_tdd_reconciliation_passed"] = False
            tdd_meta_out["pre_tdd_reconciliation_error"] = str(exc)

        # --- TDD: pytest + patch retries ---
        final_impl, final_test, tdd_meta = self._tdd_fix_loop(
            rel_impl=resolved_file_path,
            impl_code=pre_tdd_impl,
            rel_test=test_rel,
            test_code=pre_tdd_test,
            component_name=component_name,
            implemented_components_context=implemented_components_context,
            previous_attempt_feedback=previous_attempt_feedback,
        )
        tdd_meta_out.update(dict(tdd_meta) if isinstance(tdd_meta, dict) else {"tdd_meta": tdd_meta})
        try:
            post_tdd_started_at = time.perf_counter()
            final_impl, final_test = self._postprocess_python_generation(
                component_name=component_name,
                responsibilities=responsibilities,
                rel_impl=resolved_file_path,
                impl_code=final_impl,
                rel_test=test_rel,
                test_code=final_test,
                implemented_components_context=implemented_components_context,
                stage="post_tdd_reconciliation",
            )
            self._record_codegen_timing_event(
                component_name=component_name,
                stage="post_tdd_postprocess",
                started_at_perf=post_tdd_started_at,
                meta={"passed": True},
            )
            tdd_meta_out["post_tdd_reconciliation_passed"] = True
        except Exception as exc:
            logging.warning(
                "Post-TDD reconciliation failed for component '%s'; keeping implementation/tests from TDD loop "
                "(last pytest summary: %s). Error: %s",
                component_name,
                self._summarize_tdd_meta(tdd_meta),
                exc,
            )
            self._record_codegen_timing_event(
                component_name=component_name,
                stage="post_tdd_postprocess",
                started_at_perf=post_tdd_started_at,
                status="failed",
                meta={"passed": False, "error": str(exc)},
            )
            tdd_meta_out["post_tdd_reconciliation_passed"] = False
            tdd_meta_out["post_tdd_reconciliation_error"] = str(exc)

        self._record_codegen_timing_event(
            component_name=component_name,
            stage="generate_code_total",
            started_at_perf=total_started_at,
            meta={
                "mode": "skeleton_fill_tdd",
                "final_pytest_rc": tdd_meta_out.get("final_pytest_rc"),
                "pre_tdd_reconciliation_passed": tdd_meta_out.get("pre_tdd_reconciliation_passed"),
                "post_tdd_reconciliation_passed": tdd_meta_out.get("post_tdd_reconciliation_passed"),
            },
        )

        return {
            "component_name": component_name,
            "file_path": resolved_file_path,
            "code": final_impl,
            "imports": fill_resp.get("imports", []),
            "tests": {"test_file_path": test_rel, "test_code": final_test},
            "documentation": fill_resp.get("documentation", ""),
            "integration_notes": fill_resp.get("integration_notes", ""),
            "language": language,
            "skeleton_fill_tdd": tdd_meta_out,
        }

    def _generate_code_single_shot(
        self,
        component: Dict[str, Any],
        requirement: Dict[str, Any],
        architecture: Dict[str, Any],
        language: str = "python",
        implemented_components_context: str = "",
        planned_file_path: Optional[str] = None,
        previous_attempt_feedback: str = "",
    ) -> Dict[str, Any]:
        if not self.llm_client:
            return self._fallback_generate_code(component, requirement, language, planned_file_path)

        total_started_at = time.perf_counter()
        component_name = component.get("name", "Component")
        responsibilities = component.get("responsibilities", [])
        req_name = requirement.get("name", "Requirement")
        req_description = requirement.get("description", "")

        # Extract architecture rationale and design decisions
        architecture_rationale = architecture.get("rationale", "")
        dag_summary = architecture.get("dag_summary", {})
        parent_node = requirement.get("parent_node")
        parent_prev_node = requirement.get("parent_prev_node")
        
        # Get related components with their responsibilities
        other_components = [
            comp for comp in architecture.get("components", [])
            if comp.get("name") != component_name
        ]
        
        context_components = "\n".join([
            f"- {comp.get('name')}: {', '.join(comp.get('responsibilities', []))}"
            for comp in other_components[:5]
        ])
        
        # Build detailed requirement context
        req_dependencies = requirement.get("dependencies", [])
        req_metadata = requirement.get("metadata", {})
        
        planned_path_instruction = (
            f"\nPlanned File Path (MUST use exactly this path): {planned_file_path}"
            if planned_file_path else ""
        )
        previous_feedback_block = (
            "\n=== PREVIOUS ATTEMPT FEEDBACK ===\n"
            f"{previous_attempt_feedback}\n"
            "Do not repeat these failure modes. Address them explicitly in the code you return.\n"
            if previous_attempt_feedback else ""
        )

        prompt = f"""You are an expert software engineer. Generate production-ready, FULLY IMPLEMENTED code for the following component based on the architecture design and requirements.

=== PROJECT REQUIREMENT ===
Name: {req_name}
Description: {req_description}
Dependencies: {', '.join(req_dependencies) if req_dependencies else 'None'}
{f"Additional Context: {req_metadata}" if req_metadata else ""}

=== ARCHITECTURE DESIGN ===
Rationale: {architecture_rationale}
{f"DAG Context: {dag_summary.get('node_count', 0)} requirements, {dag_summary.get('edge_count', 0)} dependencies" if dag_summary else ""}

=== PARENT CONTEXT ===
Parent Node: 
{json.dumps(parent_node, ensure_ascii=False) if parent_node else "None"}
Previous Parent Node: 
{json.dumps(parent_prev_node, ensure_ascii=False) if parent_prev_node else "None"}

=== COMPONENT TO IMPLEMENT ===
Component Name: {component_name}
Responsibilities:
{chr(10).join(f"{i+1}. {resp}" for i, resp in enumerate(responsibilities))}

=== RELATED COMPONENTS (for integration) ===
{context_components if context_components else "No related components"}

{implemented_components_context}
{previous_feedback_block}

=== IMPLEMENTATION REQUIREMENTS ===
Language: {language}
{planned_path_instruction}

Based on the above architecture and requirements, generate COMPLETE, WORKING code that:

1. **Fully implements** each responsibility with actual logic (not just TODOs)
2. **Aligns with the requirement description** - the code should solve the stated problem
3. **Integrates with related components** - consider how this component interacts with others
4. **REUSES implemented components** - Import and use the classes/functions from already implemented components when applicable. Don't reimplement what's already available!
5. **Follows {language} best practices** - proper structure, naming conventions, design patterns
6. **Includes comprehensive error handling** - validate inputs, handle edge cases
7. **Has detailed documentation** - docstrings explaining what, why, and how
8. **Is production-ready** - can be used immediately with minimal modifications
9. **Uses stable contracts** - if importing another generated component, use the exact module path and exported symbol names shown in IMPLEMENTED COMPONENTS / MODULE REGISTRY
10. **Keeps core algorithms local** - {self._peer_repo_constraint_text(include_generic_utils_note=False)}

CRITICAL: 
- Generate REAL implementation code, not placeholder templates. Use the requirement description to understand WHAT to build, and the responsibilities to understand HOW to structure it.
- Check the "IMPLEMENTED COMPONENTS" section carefully. If a component provides functionality you need, IMPORT and USE it instead of reimplementing.
- When using implemented components, use the correct import paths and function signatures shown above.
- Do not leave `raise NotImplementedError(...)`, `pass`, or `...` as the sole body of any concrete public function or method.
- Do not invent new import paths or symbol names for existing generated components. Reuse the exact exported names from IMPLEMENTED COMPONENTS or MODULE REGISTRY.
- If a dependency needs adaptation, add a local compatibility wrapper or alias in this file instead of referencing a non-existent external API.
- {self._peer_repo_constraint_text(include_generic_utils_note=True)} Generic utilities such as numpy/scipy are fine when they are not standing in for the target repository's implementation.
- {self._state_contract_guidance_block()}

Return ONLY a JSON object (no markdown formatting):
{{
  "file_path": "{planned_file_path or 'appropriate/path/based/on/architecture.py'}",
  "code": "COMPLETE, WORKING CODE with full implementations"
}}"""
        
#         ,
#   "imports": ["all", "required", "dependencies"],
#   "tests": {{
#     "test_file_path": "tests/test_component.py",
#     "test_code": "comprehensive test cases covering main functionality"
#   }},
#   "documentation": "User guide and API documentation",
#   "integration_notes": "How to use this component with related components"
        
        response: Any | None = None
        try:
            # logging.warning(prompt)
            # raise Exception("Failed to generate requirements")
            logging.info("Codegen LLM request starting: phase=single_shot component=%s", component_name)
            single_started_at = time.perf_counter()
            response = self.llm_client.call_json([
                {"role": "system", "content": f"You are an expert {language} developer who writes clean, production-ready code."},
                {"role": "user", "content": prompt}
            ], temperature=0.0, max_tokens=32768, max_retry_times=self.codegen_max_retry_times, timeout_seconds=self.codegen_timeout_seconds, operation_name=f"code_generator.single_shot:{component_name}")
            self._record_codegen_timing_event(
                component_name=component_name,
                stage="single_shot_llm",
                started_at_perf=single_started_at,
            )

            self._log_response_summary(component_name, response)
            
            # Handle case where LLM returns a list instead of dict
            if isinstance(response, list):
                logging.warning(
                    "LLM returned list instead of dict for component '%s'; using single-shot fallback handling",
                    component_name,
                )
                return self._fallback_generate_code(component, requirement, language, planned_file_path)
            
            # Validate and structure response
            resolved_file_path = planned_file_path or response.get(
                "file_path", f"src/{self._to_snake_case(component_name)}.py"
            )
            resolved_file_path = self.normalize_file_path(resolved_file_path, language=language)
            result = {
                "component_name": component_name,
                "file_path": resolved_file_path,
                "code": response.get("code", ""),
                "imports": response.get("imports", []),
                "tests": response.get("tests", {}),
                "documentation": response.get("documentation", ""),
                "integration_notes": response.get("integration_notes", ""),
                "language": language,
            }
            if str(language).lower() == "python":
                tests_dict = result.get("tests") if isinstance(result.get("tests"), dict) else {}
                normalized_rel_test = (
                    self._normalize_test_file_path(
                        str(tests_dict.get("test_file_path", "")),
                        component_name=component_name,
                    )
                    if tests_dict
                    else ""
                )
                fixed_code, fixed_test = self._postprocess_python_generation(
                    component_name=component_name,
                    responsibilities=responsibilities,
                    rel_impl=resolved_file_path,
                    impl_code=str(result.get("code", "")),
                    rel_test=normalized_rel_test,
                    test_code=str(tests_dict.get("test_code", "")),
                    implemented_components_context=implemented_components_context,
                    stage="legacy_single_shot_generation",
                )
                self._record_codegen_timing_event(
                    component_name=component_name,
                    stage="single_shot_postprocess",
                    started_at_perf=single_started_at,
                    meta={"has_tests": bool(tests_dict)},
                )
                result["code"] = fixed_code
                if tests_dict:
                    result.setdefault("tests", {})
                    if isinstance(result["tests"], dict):
                        result["tests"]["test_file_path"] = normalized_rel_test
                        result["tests"]["test_code"] = fixed_test

            logging.info(f"Successfully generated code for component '{component_name}'")
            self._record_codegen_timing_event(
                component_name=component_name,
                stage="generate_code_total",
                started_at_perf=total_started_at,
                meta={"mode": "single_shot"},
            )
            return result
        
        except Exception as e:
            logging.warning(f"LLM code generation failed for component '{component_name}': {e}, using fallback")
            if response is not None:
                logging.warning("Last LLM response for '%s': %s", component_name, response)
            return self._fallback_generate_code(component, requirement, language, planned_file_path)

    def generate_batch(
        self,
        architecture: Dict[str, Any],
        requirement: Dict[str, Any],
        language: str = "python",
        implemented_components_context: str = "",
        max_components: int = 5,
        max_workers: int = 3,
        retry_feedback_by_component: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate code for all components in an architecture (with parallel processing).
        
        Args:
            architecture: Architecture with components
            requirement: Requirement being implemented
            language: Programming language
            implemented_components_context: Already implemented components
            max_components: Maximum number of components to generate
            max_workers: Maximum number of parallel workers for component generation
        
        Returns:
            List of code generation results for each component
        """
        components = architecture.get("components", [])[:max_components]
        file_plan = self.generate_file_plan(architecture, requirement, language=language)
        self.last_file_plan = file_plan
        
        logging.info(f"Starting batch code generation for {len(components)} components (workers: {max_workers})")
        
        # For small batches, use serial processing
        if len(components) <= 1 or max_workers <= 1:
            results = []
            for component in components:
                try:
                    component_name = str(component.get("name", ""))
                    result = self.generate_code(
                        component,
                        requirement,
                        architecture,
                        language,
                        implemented_components_context,
                        planned_file_path=file_plan.get(component_name),
                        previous_attempt_feedback=(retry_feedback_by_component or {}).get(component_name, ""),
                    )
                    results.append(result)
                    logging.info(f"Successfully processed component '{component.get('name', 'Unknown')}' in batch")
                except Exception as e:
                    logging.error(f"Failed to generate code for component '{component.get('name', 'Unknown')}' in batch: {e}")
            return results
        
        # Parallel processing for multiple components
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all component generation tasks
            future_to_component = {
                    executor.submit(
                        self.generate_code,
                        component,
                        requirement,
                        architecture,
                        language,
                        implemented_components_context,
                        file_plan.get(str(component.get("name", ""))),
                        (retry_feedback_by_component or {}).get(str(component.get("name", "")), ""),
                    ): component for component in components
                }
            
            # Collect results as they complete
            for future in as_completed(future_to_component):
                component = future_to_component[future]
                component_name = component.get('name', 'Unknown')
                try:
                    result = future.result()
                    results.append(result)
                    logging.info(f"Successfully processed component '{component_name}' in batch")
                except Exception as e:
                    logging.error(f"Failed to generate code for component '{component_name}' in batch: {e}")
                    # Continue with other components
        
        logging.info(f"Batch code generation completed. Generated {len(results)} out of {len(components)} components")
        return results

    def get_last_file_plan(self) -> Dict[str, str]:
        return dict(self.last_file_plan)

    def generate_file_plan(
        self,
        architecture: Dict[str, Any],
        requirement: Dict[str, Any],
        language: str = "python",
    ) -> Dict[str, str]:
        components = architecture.get("components", []) or []
        if not isinstance(components, list):
            return {}
        if not self.enable_two_stage_file_plan:
            return self._fallback_file_plan(components, language=language)

        if not self.llm_client:
            return self._fallback_file_plan(components, language=language)

        component_list = [
            {
                "name": str(comp.get("name", "")).strip(),
                "responsibilities": comp.get("responsibilities", []),
                "serves_subrequirements": comp.get("serves_subrequirements", []),
            }
            for comp in components if isinstance(comp, dict)
        ]

        requirement_name = str(requirement.get("name", "unknown_requirement"))
        requirement_description = str(requirement.get("description", "")).strip()
        prompt = f"""You are planning file layout for a Python project.

Requirement: {requirement_name}
Description: {requirement_description}
Components:
{json.dumps(component_list, ensure_ascii=False, indent=2)}

Rules:
1. Output one file path for each component.
2. File paths must be relative paths and end with .py
3. Keep paths under these roots when possible: {self.path_allowed_roots}
4. Prefer stable package layout and avoid creating new top-level roots.
5. Preserve domain semantics from the requirement and served sub-requirements. If the architecture exposes domain families, keep them visible in subpackage names.
6. Avoid collapsing unrelated domain families into generic buckets such as core, common, runtime, api, or services unless a domain-specific nested subpackage is also present.
7. When a generic package is unavoidable, prefer paths like core/<domain_family>/module.py over core/module.py.
8. The file path should make it easy to recover the original requirement taxonomy from package names alone.
9. Do NOT create one subpackage per named feature by default. Multiple nearby features should share the same module family when they belong to one stable domain area.

Return ONLY JSON:
{{
  "plans": [
    {{
      "component_name": "ComponentName",
      "file_path": "{self._primary_python_package_root()}/subpkg/module_name.py"
    }}
  ]
}}"""
        try:
            logging.info("Codegen LLM request starting: phase=file_plan requirement=%s", requirement_name)
            response = self.llm_client.call_json(
                [
                    {"role": "system", "content": "You output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=32768,
                max_retry_times=self.codegen_max_retry_times,
                timeout_seconds=self.codegen_timeout_seconds,
                operation_name=f"code_generator.file_plan:{requirement_name}",
            )
            plans = response.get("plans", []) if isinstance(response, dict) else []
            if not isinstance(plans, list):
                plans = []
            mapping: Dict[str, str] = {}
            for item in plans:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("component_name", "")).strip()
                file_path = str(item.get("file_path", "")).strip()
                if not name:
                    continue
                normalized = self.normalize_file_path(file_path, language=language)
                mapping[name] = normalized
            if not mapping:
                return self._fallback_file_plan(components, language=language)
            # Ensure all components have paths.
            fallback = self._fallback_file_plan(components, language=language)
            for k, v in fallback.items():
                mapping.setdefault(k, v)
            return mapping
        except Exception as exc:
            logging.warning("File plan stage failed (%s), using fallback planning", exc)
            return self._fallback_file_plan(components, language=language)

    def _fallback_file_plan(
        self,
        components: List[Dict[str, Any]],
        language: str = "python",
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for comp in components:
            if not isinstance(comp, dict):
                continue
            name = str(comp.get("name", "")).strip()
            if not name:
                continue
            snake = self._to_snake_case(name)
            default_rel = (
                f"{self._primary_python_package_root()}/generated/{snake}.py"
                if language.lower() == "python"
                else f"src/{snake}.{language}"
            )
            mapping[name] = self.normalize_file_path(default_rel, language=language)
        return mapping

    def _parse_allowed_roots(self, value: Any) -> List[str]:
        pkg_root = self._primary_python_package_root()
        default_roots = [pkg_root, "docs", "tools", "examples", "tests", f"{pkg_root}/tests"]
        if value is None:
            return default_roots
        if isinstance(value, list):
            roots = [
                normalize_python_package_root(str(v).strip().strip("/"), default="")
                for v in value
                if str(v).strip()
            ]
            roots = [root for root in roots if root]
            return roots or default_roots
        text = str(value).strip()
        if not text:
            return default_roots
        roots = [
            normalize_python_package_root(part.strip().strip("/"), default="")
            for part in text.split(",")
            if part.strip()
        ]
        roots = [root for root in roots if root]
        return roots or default_roots

    def normalize_file_path(self, raw_path: str, language: str = "python") -> str:
        path = str(raw_path or "").strip().replace("\\", "/")
        path = path.lstrip("./").lstrip("/")
        if not path:
            path = (
                f"{self._primary_python_package_root()}/generated/unnamed_component.py"
                if language.lower() == "python"
                else "src/unnamed_component.txt"
            )
        if language.lower() == "python" and not path.endswith(".py"):
            path = f"{path}.py"

        parts = [p for p in path.split("/") if p and p != "."]
        if not parts:
            parts = [self._primary_python_package_root(), "generated", "unnamed_component.py"]

        # Keep docs/tools/examples/tests as top-level roots. Everything else goes under the primary package root.
        pkg_root = self._primary_python_package_root()
        keep_roots = {"docs", "tools", "examples", "tests", pkg_root}
        if language.lower() == "python" and normalize_python_package_root(parts[0], default="") == pkg_root:
            parts[0] = pkg_root
        if parts[0] not in keep_roots:
            parts = [pkg_root] + parts

        return "/".join(parts)

    def _normalize_test_file_path(self, raw_path: str, component_name: str) -> str:
        path = str(raw_path or "").strip().replace("\\", "/").lstrip("./").lstrip("/")
        if not path:
            path = f"tests/test_{self._to_snake_case(component_name)}.py"
        if not path.startswith("tests/"):
            path = f"tests/{path}"
        parts = [p for p in path.split("/") if p and p != "."]
        if not parts:
            parts = ["tests", f"test_{self._to_snake_case(component_name)}.py"]
        parts[0] = "tests"
        filename = parts[-1]
        stem = filename[:-3] if filename.endswith(".py") else filename
        if not stem.startswith("test_"):
            stem = f"test_{stem}"
        stem = self._to_snake_case(stem)
        stem = re.sub(r"[^a-z0-9_]+", "_", stem)
        stem = re.sub(r"_+", "_", stem).strip("_") or f"test_{self._to_snake_case(component_name)}"
        parts[-1] = f"{stem}.py"
        return "/".join(parts)

    def save_generated_code(
        self,
        code_result: Dict[str, Any],
        output_dir: str,
        create_tests: bool = True
    ) -> Dict[str, str]:
        """
        Save generated code to files.
        
        Returns:
            Dictionary with created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_started_at = time.perf_counter()
        
        created_files = {}
        component_name = code_result.get("component_name", "Unknown")
        code_result["test_save_succeeded"] = True
        code_result["test_save_error"] = ""
        code_result["skipped_test_files"] = []
        
        try:
            generation_status = self._derive_generation_status(code_result)
            code_result["generation_status"] = generation_status
            tdd_meta = code_result.get("skeleton_fill_tdd", {})
            final_pytest_rc = None
            if isinstance(tdd_meta, dict):
                final_pytest_rc = tdd_meta.get("final_pytest_rc")
            code_result["tdd_final_pytest_rc"] = final_pytest_rc
            code_result["tdd_passed"] = generation_status != "retained_after_tdd_failure"

            # Save main code file
            code_file = output_path / code_result["file_path"]
            code_file.parent.mkdir(parents=True, exist_ok=True)

            code_content = str(code_result.get("code", ""))
            if code_file.suffix == ".py":
                code_content = self._autofix_python_syntax(
                    code_content,
                    component_name=component_name,
                    rel_path=code_result.get("file_path", ""),
                )
                remaining = self._find_python_placeholder_issues(
                    code_content,
                    code_result.get("file_path", ""),
                )
                warnings = self._find_python_placeholder_warnings(
                    code_content,
                    code_result.get("file_path", ""),
                )
                if remaining:
                    raise RuntimeError(
                        "Refusing to save Python file with unresolved placeholders: "
                        + "; ".join(remaining)
                    )
                if warnings:
                    code_result.setdefault("placeholder_warnings", {})
                    code_result["placeholder_warnings"]["code"] = warnings
                    logging.warning(
                        "Saving Python file for component '%s' with soft placeholder warnings: %s",
                        component_name,
                        "; ".join(warnings),
                    )
                code_result["code"] = code_content

            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
            created_files["code"] = str(code_file)
            if code_file.suffix == ".py":
                peer_report = self._file_level_peer_repo_postcheck_and_repair(
                    file_path=code_file,
                    component_name=component_name,
                    rel_path=code_result.get("file_path", ""),
                )
                code_result["peer_repo_postcheck"] = dict(peer_report)
                postchecks = self._compile_postcheck_and_repair_file(
                    file_path=code_file,
                    component_name=component_name,
                    rel_path=code_result.get("file_path", ""),
                )
                code_result["syntax_postcheck"] = dict(postchecks.get("syntax_postcheck", {}))
                code_result["compile_postcheck"] = dict(postchecks.get("compile_postcheck", {}))
            
            # Save test file if available
            if create_tests and code_result.get("tests"):
                tests = code_result["tests"]
                raw_test_file_path = str(
                    tests.get("test_file_path", f"tests/test_{code_result['file_path']}")
                )
                test_file_path = self._normalize_test_file_path(
                    raw_test_file_path,
                    component_name=component_name,
                )
                tests["test_file_path"] = test_file_path
                test_file = output_path / test_file_path
                test_file.parent.mkdir(parents=True, exist_ok=True)
                stale_test_rel = str(raw_test_file_path or "").strip().replace("\\", "/").lstrip("./").lstrip("/")
                if stale_test_rel and stale_test_rel != test_file_path:
                    stale_test_file = output_path / stale_test_rel
                    if stale_test_file.exists() and stale_test_file.is_file():
                        try:
                            stale_test_file.unlink()
                            logging.info(
                                "Removed stale non-normalized test file for component '%s': %s",
                                component_name,
                                stale_test_rel,
                            )
                        except Exception as exc:
                            logging.warning(
                                "Failed to remove stale non-normalized test file for component '%s' (%s): %s",
                                component_name,
                                stale_test_rel,
                                exc,
                            )
                
                test_code = tests.get("test_code", "")
                if test_code:
                    if test_file.suffix == ".py":
                        test_code = self._autofix_python_syntax(
                            test_code,
                            component_name=component_name,
                            rel_path=test_file_path,
                        )
                        remaining = self._find_python_placeholder_issues(
                            test_code,
                            test_file_path,
                        )
                        warnings = self._find_python_placeholder_warnings(
                            test_code,
                            test_file_path,
                        )
                        if remaining:
                            skip_reason = (
                                "Skipped saving Python test file with unresolved placeholders: "
                                + "; ".join(remaining)
                            )
                            code_result["test_save_succeeded"] = False
                            code_result["test_save_error"] = skip_reason
                            code_result["skipped_test_files"].append(
                                {
                                    "path": test_file_path,
                                    "reason": skip_reason,
                                }
                            )
                            logging.error(
                                "Component '%s' skipped all generated test files during save: %s",
                                component_name,
                                skip_reason,
                            )
                            tests["test_code"] = test_code
                            test_code = ""
                        if warnings:
                            code_result.setdefault("placeholder_warnings", {})
                            code_result["placeholder_warnings"]["test"] = warnings
                            logging.warning(
                                "Saving Python test file for component '%s' with soft placeholder warnings: %s",
                                component_name,
                                "; ".join(warnings),
                            )
                    if test_code:
                        with open(test_file, 'w', encoding='utf-8') as f:
                            f.write(test_code)
                        created_files["test"] = str(test_file)
                        if test_file.suffix == ".py":
                            postchecks = self._compile_postcheck_and_repair_file(
                                file_path=test_file,
                                component_name=component_name,
                                rel_path=test_file_path,
                            )
                            code_result["test_syntax_postcheck"] = dict(postchecks.get("syntax_postcheck", {}))
                            code_result["test_compile_postcheck"] = dict(postchecks.get("compile_postcheck", {}))
                        tests["test_code"] = test_code
            
            # Save documentation
            if code_result.get("documentation"):
                doc_file = output_path / f"docs/{component_name}.md"
                doc_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(doc_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {component_name}\n\n")
                    f.write(code_result["documentation"])
                    if code_result.get("integration_notes"):
                        f.write(f"\n\n## Integration Notes\n\n{code_result['integration_notes']}")
                created_files["documentation"] = str(doc_file)
            
            if generation_status == "retained_after_tdd_failure":
                logging.warning(
                    "Saved retained component '%s' after TDD failure (final_pytest_rc=%s) to %d files",
                    component_name,
                    final_pytest_rc,
                    len(created_files),
                )
            else:
                logging.info(f"Successfully saved generated code for component '{component_name}' to {len(created_files)} files")
            code_result["save_succeeded"] = True
            code_result["save_error"] = ""
            self._record_codegen_timing_event(
                component_name=component_name,
                stage="save_generated_code",
                started_at_perf=save_started_at,
                meta={
                    "saved_files": len(created_files),
                    "test_save_succeeded": bool(code_result.get("test_save_succeeded", True)),
                },
            )
            return created_files
            
        except Exception as e:
            logging.error(f"Failed to save generated code for component '{component_name}': {e}")
            code_result["save_succeeded"] = False
            code_result["save_error"] = str(e)
            if created_files:
                code_result["partial_created_files"] = dict(created_files)
                logging.warning(
                    "Retaining %d partially written file(s) for failed component '%s': %s",
                    len(created_files),
                    component_name,
                    sorted(created_files.values()),
                )
            self._record_codegen_timing_event(
                component_name=component_name,
                stage="save_generated_code",
                started_at_perf=save_started_at,
                status="failed",
                meta={"error": str(e), "partial_files": dict(created_files)},
            )
            return created_files

    def _compile_postcheck_and_repair_file(
        self,
        *,
        file_path: Path,
        component_name: str,
        rel_path: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Compile a just-written Python file; on failure, attempt one local repair and retry."""
        syntax_status = self._evaluate_python_syntax_postcheck(file_path)
        compile_status = self._evaluate_python_compile_postcheck(file_path)
        if syntax_status.get("passed") and compile_status.get("passed"):
            logging.info(
                "Component file compile postcheck passed for '%s': %s",
                component_name,
                rel_path,
            )
            return {
                "syntax_postcheck": syntax_status,
                "compile_postcheck": compile_status,
            }
        before_error = str(compile_status.get("error") or syntax_status.get("error") or "")

        original = file_path.read_text(encoding="utf-8")
        repaired = self._autofix_python_syntax(
            original,
            component_name=component_name,
            rel_path=rel_path,
        )
        if repaired != original:
            file_path.write_text(repaired, encoding="utf-8")
            syntax_status = self._evaluate_python_syntax_postcheck(file_path)
            compile_status = self._evaluate_python_compile_postcheck(file_path)
            if syntax_status.get("passed") and compile_status.get("passed"):
                logging.info(
                    "Component file compile postcheck repaired for '%s': %s",
                    component_name,
                    rel_path,
                )
                return {
                    "syntax_postcheck": syntax_status,
                    "compile_postcheck": compile_status,
                }
            after_error = str(compile_status.get("error") or syntax_status.get("error") or "")
        else:
            after_error = before_error

        raise RuntimeError(
            f"file compile postcheck failed for {rel_path}: before={before_error} after={after_error}"
        )

    @staticmethod
    def _evaluate_python_syntax_postcheck(file_path: Path) -> Dict[str, Any]:
        try:
            source = file_path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(file_path))
            return {"passed": True, "code_file": str(file_path), "error": ""}
        except SyntaxError as exc:
            return {"passed": False, "code_file": str(file_path), "error": str(exc)}

    @staticmethod
    def _evaluate_python_compile_postcheck(file_path: Path) -> Dict[str, Any]:
        try:
            py_compile.compile(str(file_path), doraise=True)
            return {"passed": True, "code_file": str(file_path), "error": ""}
        except py_compile.PyCompileError as exc:
            return {"passed": False, "code_file": str(file_path), "error": str(exc)}

    def refine_code(
        self,
        existing_code: str,
        refinement_request: str,
        context: str = ""
    ) -> str:
        """
        Refine existing code based on feedback or requirements.
        
        Args:
            existing_code: Current code to refine
            refinement_request: What needs to be improved
            context: Additional context
            
        Returns:
            Refined code
        """
        if not self.llm_client:
            return existing_code
        
        prompt = f"""Refine the following code based on the request.

Current Code:
```
{existing_code}
```

Refinement Request:
{refinement_request}

Context:
{context}

Return ONLY the improved code without explanations or markdown formatting."""
        
        try:
            logging.info("Codegen LLM request starting: phase=refine_code")
            refined_code = self.llm_client.call([
                {"role": "system", "content": "You are an expert code reviewer and refactorer."},
                {"role": "user", "content": prompt}
            ], temperature=0.0, max_tokens=32768, max_retry_times=self.codegen_max_retry_times, timeout_seconds=self.codegen_timeout_seconds, operation_name="code_generator.refine_code")
            
            # Remove markdown code blocks if present
            if "```" in refined_code:
                lines = refined_code.split("\n")
                code_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or not any(line.strip().startswith(c) for c in ["```", "#", "**"]):
                        code_lines.append(line)
                refined_code = "\n".join(code_lines)
            
            return refined_code.strip()
        
        except Exception:
            return existing_code

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return bool(value)

    def _autofix_python_syntax(self, code: str, component_name: str, rel_path: str) -> str:
        if not self.enable_syntax_autofix:
            return code
        code = self._strip_patch_artifacts(code)
        result = self.fix_agent.fix_python_content(code)
        if result.get("fixed"):
            logging.info(
                "FixAgent repaired syntax for component '%s' file '%s' in %s rounds",
                component_name,
                rel_path,
                result.get("rounds", 0),
            )
            return str(result.get("fixed_content", code))

        error_before = result.get("error_before")
        error_after = result.get("error_after")
        if error_before and error_after:
            logging.warning(
                "FixAgent could not repair syntax for component '%s' file '%s': %s",
                component_name,
                rel_path,
                error_after,
            )
        return code

    def _find_signature_structure_issues(self, code: str, rel_path: str) -> List[str]:
        issues: List[str] = []
        for idx, line in enumerate(code.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ")):
                if "*, **" in line or "(*, **" in line:
                    issues.append(
                        f"{rel_path}:{idx} invalid function signature uses bare '*' immediately before **kwargs"
                    )
                if re.search(r"^\s*(?:async\s+def|def)\s+[A-Za-z_][A-Za-z0-9_]*\s*[,/:]\s*\(", line):
                    issues.append(
                        f"{rel_path}:{idx} invalid function declaration contains punctuation immediately after the function name"
                    )
                if re.search(r"^\s*(?:async\s+def|def)\s+[A-Za-z_][A-Za-z0-9_]*[^\w\s(]", line):
                    issues.append(
                        f"{rel_path}:{idx} invalid function declaration contains illegal characters near the function name"
                    )
            elif stripped.startswith("class "):
                if re.search(r"^\s*class\s+[A-Za-z_][A-Za-z0-9_]*\s*[,/]", line):
                    issues.append(
                        f"{rel_path}:{idx} invalid class declaration contains punctuation immediately after the class name"
                    )
        return issues

    def _repair_and_validate_python_interfaces(
        self,
        *,
        component_name: str,
        rel_path: str,
        code: str,
        stage: str,
    ) -> str:
        repaired = self._autofix_python_syntax(code, component_name, rel_path)
        issues = self._find_signature_structure_issues(repaired, rel_path)
        if issues:
            second_pass = self.fix_agent.fix_python_content(repaired)
            if second_pass.get("fixed"):
                repaired = str(second_pass.get("fixed_content", repaired))
                issues = self._find_signature_structure_issues(repaired, rel_path)
        if issues:
            raise RuntimeError(
                f"{stage} left invalid Python interface declarations: " + "; ".join(issues)
            )
        return repaired

    @staticmethod
    def _normalize_python_import_module_paths(code: str) -> str:
        if not code:
            return code

        def _strip_py_segments(module_path: str) -> str:
            parts = [part for part in str(module_path).split(".") if part]
            cleaned = [part[:-3] if part.endswith(".py") else part for part in parts]
            cleaned = [part for part in cleaned if part != "py"]
            return ".".join(cleaned)

        text = str(code)
        text = re.sub(
            r"(?m)^(\s*from\s+)([A-Za-z_][A-Za-z0-9_\.]*)(\s+import\s+)",
            lambda m: f"{m.group(1)}{_strip_py_segments(m.group(2))}{m.group(3)}",
            text,
        )
        text = re.sub(
            r"(?m)^(\s*import\s+)([A-Za-z_][A-Za-z0-9_\.]*)(\s*(?:#.*)?)$",
            lambda m: f"{m.group(1)}{_strip_py_segments(m.group(2))}{m.group(3)}",
            text,
        )
        return text

    def _apply_local_validation_repairs(
        self,
        *,
        component_name: str,
        responsibilities: List[Any],
        rel_impl: str,
        impl_code: str,
        rel_test: str,
        test_code: str,
        implemented_components_context: str,
        stage: str,
    ) -> Tuple[str, str, Dict[str, Any]]:
        before_issues = self._merge_postprocess_python_issues(
            component_name=component_name,
            responsibilities=responsibilities,
            rel_impl=rel_impl,
            impl_code=impl_code,
            rel_test=rel_test,
            test_code=test_code,
            implemented_components_context=implemented_components_context,
        )
        patched_impl, patched_test = impl_code, test_code

        def _repair_one(code: str, rel_path: str) -> str:
            if not code or not rel_path:
                return code
            candidate = self._normalize_python_import_module_paths(code)
            candidate = self._autofix_python_syntax(candidate, component_name, rel_path)
            try:
                candidate = self._repair_and_validate_python_interfaces(
                    component_name=component_name,
                    rel_path=rel_path,
                    code=candidate,
                    stage=f"{stage}_local_validation_prepass",
                )
            except Exception as exc:
                logging.debug(
                    "Local validation prepass could not fully normalize '%s' for component '%s': %s",
                    rel_path,
                    component_name,
                    exc,
                )
            return candidate

        patched_impl = _repair_one(patched_impl, rel_impl)
        if rel_test and patched_test:
            patched_test = _repair_one(patched_test, rel_test)

        after_issues = self._merge_postprocess_python_issues(
            component_name=component_name,
            responsibilities=responsibilities,
            rel_impl=rel_impl,
            impl_code=patched_impl,
            rel_test=rel_test,
            test_code=patched_test,
            implemented_components_context=implemented_components_context,
        )
        meta = {
            "attempted": bool(before_issues),
            "changed": patched_impl != impl_code or patched_test != test_code,
            "issues_before": len(before_issues),
            "issues_after": len(after_issues),
            "issues_reduced": max(0, len(before_issues) - len(after_issues)),
        }
        if meta["attempted"] and (meta["changed"] or meta["issues_reduced"]):
            logging.info(
                "Local validation prepass for component '%s' at stage '%s': issues %d -> %d",
                component_name,
                stage,
                len(before_issues),
                len(after_issues),
            )
        return patched_impl, patched_test, meta

    def _review_python_skeleton(self, *, component_name: str, rel_path: str, code: str) -> str:
        return self._repair_and_validate_python_interfaces(
            component_name=component_name,
            rel_path=rel_path,
            code=code,
            stage="skeleton_review",
        )

    def _find_skeleton_responsibility_alignment_issues(
        self,
        *,
        component_name: str,
        responsibilities: List[Any],
        skeleton_code: str,
        rel_path: str,
    ) -> List[str]:
        issues = self._find_responsibility_realization_gaps(
            component_name=component_name,
            responsibilities=responsibilities,
            impl_code=skeleton_code,
            test_code="",
        )
        return [f"{rel_path}: {issue}" for issue in issues]

    @staticmethod
    def _strip_patch_artifacts(code: str) -> str:
        if not code:
            return code
        cleaned_lines: List[str] = []
        removed = False
        patch_prefixes = (
            "*** Begin Patch",
            "*** End Patch",
            "*** Update File:",
            "*** Add File:",
            "*** Delete File:",
            "*** Move to:",
            "*** End of File",
            "@@",
            "--- ",
            "+++ ",
        )
        for line in code.splitlines():
            stripped = line.strip()
            if any(stripped.startswith(prefix) for prefix in patch_prefixes):
                removed = True
                continue
            cleaned_lines.append(line)
        if not removed:
            return code
        cleaned = "\n".join(cleaned_lines)
        if code.endswith("\n"):
            cleaned += "\n"
        return cleaned

    def _find_patch_artifact_issues(self, code: str, rel_path: str) -> List[str]:
        issues: List[str] = []
        for idx, line in enumerate(code.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("*** ") or stripped.startswith("@@") or stripped.startswith("--- ") or stripped.startswith("+++ "):
                issues.append(f"{rel_path}:{idx} patch artifact marker remains")
        return issues

    @staticmethod
    def _decorator_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return CodeGeneratorAgent._decorator_name(node.func)
        return ""

    @staticmethod
    def _class_base_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            return CodeGeneratorAgent._class_base_name(node.value)
        return ""

    def _is_abstract_container(
        self,
        class_stack: List[ast.ClassDef],
        func_node: ast.AST,
    ) -> bool:
        if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in func_node.decorator_list:
                if self._decorator_name(deco) in {"abstractmethod", "abstractclassmethod", "abstractstaticmethod"}:
                    return True
        for class_node in class_stack:
            for base in class_node.bases:
                if self._class_base_name(base) in {"ABC", "ABCMeta", "Protocol"}:
                    return True
        return False

    @staticmethod
    def _is_test_rel_path(rel_path: str) -> bool:
        return str(rel_path or "").replace("\\", "/").startswith("tests/")

    @staticmethod
    def _compat_docstring_hint(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(token in lowered for token in ("fallback", "stand-in", "compatibility"))

    def _collect_python_placeholder_findings(self, code: str, rel_path: str) -> List[Dict[str, str]]:
        findings: List[Dict[str, str]] = []
        for issue in self._find_patch_artifact_issues(code, rel_path):
            findings.append(
                {
                    "severity": "hard",
                    "category": "patch_artifact",
                    "message": issue,
                }
            )

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return findings

        is_test_file = self._is_test_rel_path(rel_path)

        class PlaceholderVisitor(ast.NodeVisitor):
            def __init__(self, outer: "CodeGeneratorAgent") -> None:
                self.outer = outer
                self.class_stack: List[ast.ClassDef] = []
                self.compat_stack: List[bool] = []
                self.except_depth = 0

            def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
                self.except_depth += 1
                self.generic_visit(node)
                self.except_depth -= 1

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.class_stack.append(node)
                class_doc = ast.get_docstring(node) or ""
                is_compat_stub = self.except_depth > 0 and self.outer._compat_docstring_hint(class_doc)
                self.compat_stack.append(is_compat_stub)
                self.generic_visit(node)
                self.compat_stack.pop()
                self.class_stack.pop()

            def _append_finding(self, *, severity: str, category: str, message: str) -> None:
                findings.append(
                    {
                        "severity": severity,
                        "category": category,
                        "message": message,
                    }
                )

            def _record_if_placeholder(self, node: ast.AST) -> None:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return
                if not node.body or len(node.body) != 1:
                    return

                if self.outer._is_abstract_container(self.class_stack, node):
                    severity = "soft"
                    category = "abstract_stub"
                elif any(self.compat_stack):
                    severity = "soft"
                    category = "compat_stub"
                else:
                    severity = "hard"
                    category = "test_placeholder" if is_test_file else "concrete_placeholder"

                stmt = node.body[0]
                if isinstance(stmt, ast.Pass):
                    self._append_finding(
                        severity=severity,
                        category=category,
                        message=f"{rel_path}:{getattr(stmt, 'lineno', 0)} function '{node.name}' has only `pass`",
                    )
                    return
                if isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant):
                    if stmt.value.value is Ellipsis:
                        self._append_finding(
                            severity=severity,
                            category=category,
                            message=f"{rel_path}:{getattr(stmt, 'lineno', 0)} function '{node.name}' has only `...`",
                        )
                    return
                if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                    func = stmt.exc.func
                    func_name = ""
                    if isinstance(func, ast.Name):
                        func_name = func.id
                    elif isinstance(func, ast.Attribute):
                        func_name = func.attr
                    if func_name == "NotImplementedError":
                        exc_args = getattr(stmt.exc, "args", []) or []
                        if exc_args:
                            first_arg = exc_args[0]
                            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                                if "TDD" in first_arg.value:
                                    self._append_finding(
                                        severity=severity,
                                        category=category,
                                        message=f"{rel_path}: explicit TDD placeholder remains",
                                    )
                        self._append_finding(
                            severity=severity,
                            category=category,
                            message=f"{rel_path}:{getattr(stmt, 'lineno', 0)} function '{node.name}' raises NotImplementedError",
                        )

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._record_if_placeholder(node)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._record_if_placeholder(node)
                self.generic_visit(node)

        PlaceholderVisitor(self).visit(tree)
        return findings

    def _find_python_placeholder_issues(self, code: str, rel_path: str) -> List[str]:
        return sorted(
            {
                item["message"]
                for item in self._collect_python_placeholder_findings(code, rel_path)
                if item.get("severity") == "hard"
            }
        )

    def _find_python_placeholder_warnings(self, code: str, rel_path: str) -> List[str]:
        return sorted(
            {
                item["message"]
                for item in self._collect_python_placeholder_findings(code, rel_path)
                if item.get("severity") != "hard"
            }
        )

    def _extract_known_contracts(self, implemented_components_context: str) -> Tuple[Set[str], Set[str]]:
        known_symbols: Set[str] = set()
        known_modules: Set[str] = set()
        if not implemented_components_context:
            return known_symbols, known_modules

        for raw_line in implemented_components_context.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("File: "):
                rel_path = line[len("File: ") :].strip()
                if rel_path.endswith(".py"):
                    known_modules.add(self._rel_path_to_module_qualname(rel_path))
                continue
            if line.startswith("Public API:"):
                payload = line.split(":", 1)[1]
                for item in payload.split(","):
                    symbol = item.strip()
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol):
                        known_symbols.add(symbol)
                continue
            func_match = re.search(r"-\s+([A-Za-z_][A-Za-z0-9_]*)\(", line)
            if func_match:
                known_symbols.add(func_match.group(1))
                continue
            class_match = re.fullmatch(r"-\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if class_match:
                known_symbols.add(class_match.group(1))
        return known_symbols, known_modules

    def _find_forbidden_peer_repo_imports(self, code: str, rel_path: str) -> List[str]:
        forbidden_roots = self._forbidden_peer_repo_roots()
        if not forbidden_roots:
            return []
        issues: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = str(alias.name or "").split(".")[0]
                    if root in forbidden_roots:
                        issues.append(
                            f"{rel_path}:{getattr(node, 'lineno', 0)} imports peer framework repo '{alias.name}'"
                        )
            elif isinstance(node, ast.ImportFrom):
                root = str(node.module or "").split(".")[0]
                if root in forbidden_roots:
                    issues.append(
                        f"{rel_path}:{getattr(node, 'lineno', 0)} imports from peer framework repo '{node.module}'"
                    )
        return sorted(set(issues))

    def _repair_forbidden_peer_repo_usage(
        self,
        *,
        component_name: str,
        rel_path: str,
        code: str,
        responsibilities: Optional[List[Any]] = None,
        stage: str,
    ) -> Tuple[str, Dict[str, Any]]:
        issues = self._find_forbidden_peer_repo_imports(code, rel_path)
        if not issues:
            return code, {"passed": True, "issues": [], "repaired": False, "stage": stage}

        if not self.llm_client:
            return code, {
                "passed": False,
                "issues": issues,
                "repaired": False,
                "stage": stage,
                "error": "LLM unavailable",
            }

        patched = code
        repaired = False
        max_rounds = max(1, int(self.api_config.get("peer_repo_revise_max_retries", 2)))
        for round_idx in range(max_rounds):
            issues = self._find_forbidden_peer_repo_imports(patched, rel_path)
            if not issues:
                return patched, {
                    "passed": True,
                    "issues": [],
                    "repaired": repaired,
                    "stage": stage,
                    "rounds": round_idx,
                }

            task_description = (
                f"Revise only the peer-framework delegated portion of component '{component_name}' "
                f"for file '{rel_path}' at stage '{stage}' (round {round_idx + 1}/{max_rounds}).\n"
                "Constraints:\n"
                "1. Do not revise the whole file.\n"
                "2. Remove imports from peer framework repositories and locally reimplement only the affected algorithmic slice.\n"
                "3. Keep unaffected APIs, tests, and file paths unchanged.\n"
                "4. Preserve public function/class names and signatures.\n"
                "5. Prefer replacing the delegated call with a small local helper or direct local logic in this file.\n"
                "6. The final file must remain valid Python syntax.\n\n"
                "Detected forbidden imports:\n- "
                + "\n- ".join(issues)
            )
            if responsibilities:
                task_description += (
                    "\n\nComponent responsibilities:\n- "
                    + "\n- ".join(str(item) for item in responsibilities if str(item).strip())
                )

            patch = self._get_patch_agent().generate_patch(
                task_description=task_description,
                related_files={rel_path: patched},
                incremental_goal=(
                    "Perform a local, minimal revise of the delegated algorithmic portion and remove the peer-framework import."
                ),
                failure_kind="peer_repo_delegation",
                telemetry_context={
                    "component_name": component_name,
                    "stage": f"{stage}_peer_repo_revise",
                    "round": round_idx + 1,
                    "file_role": "impl_only",
                },
            )
            updated_files = patch.get("updated_files", {}) if isinstance(patch, dict) else {}
            candidate = str(updated_files.get(rel_path, patched))
            if candidate != patched:
                repaired = True
            patched = self._autofix_python_syntax(candidate, component_name, rel_path)
            patched = self._repair_and_validate_python_interfaces(
                component_name=component_name,
                rel_path=rel_path,
                code=patched,
                stage=f"{stage}_peer_repo_revise",
            )

        remaining = self._find_forbidden_peer_repo_imports(patched, rel_path)
        return patched, {
            "passed": not remaining,
            "issues": remaining,
            "repaired": repaired,
            "stage": stage,
            "rounds": max_rounds,
        }

    def _file_level_peer_repo_postcheck_and_repair(
        self,
        *,
        file_path: Path,
        component_name: str,
        rel_path: str,
    ) -> Dict[str, Any]:
        if file_path.suffix != ".py":
            return {"passed": True, "issues": [], "repaired": False, "stage": "file_save"}
        try:
            code = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return {"passed": False, "issues": [], "repaired": False, "stage": "file_save", "error": str(exc)}

        patched, report = self._repair_forbidden_peer_repo_usage(
            component_name=component_name,
            rel_path=rel_path,
            code=code,
            responsibilities=[],
            stage="file_save",
        )
        if patched != code:
            file_path.write_text(patched, encoding="utf-8")
        if report.get("passed"):
            logging.info(
                "Peer-repo delegation postcheck passed for '%s': %s",
                component_name,
                rel_path,
            )
        else:
            logging.warning(
                "Peer-repo delegation remains after file-save repair for '%s': %s",
                component_name,
                "; ".join(report.get("issues", [])),
            )
        return report

    def _find_contract_suspicions(
        self,
        code: str,
        rel_path: str,
        implemented_components_context: str,
    ) -> List[str]:
        if not implemented_components_context:
            return []
        known_symbols, known_modules = self._extract_known_contracts(implemented_components_context)
        if not known_symbols and not known_modules:
            return []

        suspicions: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return suspicions

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("statsmodels"):
                module_name = node.module
                if known_modules and module_name not in known_modules:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if known_symbols and alias.name not in known_symbols:
                        suspicions.append(
                            f"{rel_path}:{getattr(node, 'lineno', 0)} imports symbol '{alias.name}' "
                            "not present in IMPLEMENTED COMPONENTS public API"
                        )
        return sorted(set(suspicions))

    def _skeleton_declares_abstract_interface(self, skeleton_code: str) -> bool:
        try:
            tree = ast.parse(skeleton_code)
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if self._class_base_name(base) in {"ABC", "ABCMeta", "Protocol"}:
                        return True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for deco in node.decorator_list:
                    if self._decorator_name(deco) in {"abstractmethod", "abstractclassmethod", "abstractstaticmethod"}:
                        return True
        return False

    def _find_test_notimplemented_alignment_issues(
        self,
        *,
        skeleton_code: str,
        test_code: str,
        rel_test: str,
    ) -> List[str]:
        if not test_code:
            return []
        if self._skeleton_declares_abstract_interface(skeleton_code):
            return []
        issues: List[str] = []
        for idx, line in enumerate(test_code.splitlines(), start=1):
            lowered = line.lower()
            if "pytest.raises" in lowered and "notimplementederror" in lowered:
                issues.append(
                    f"{rel_test}:{idx} test asserts NotImplementedError for a concrete API; align it to final behavior instead"
                )
        return issues

    def _tokenize_responsibility_text(self, text: str) -> List[str]:
        tokens = [
            tok.lower()
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(text or ""))
        ]
        stop_words = {
            "the", "and", "for", "with", "from", "into", "onto", "that", "this",
            "using", "used", "via", "when", "where", "while", "across", "under",
            "over", "each", "main", "public", "api", "support", "provide",
            "returns", "return", "result", "results", "data", "model", "models",
            "component", "components", "service", "services", "library", "manager",
            "engine", "adapter", "adapters", "system", "systems", "utils",
            "utilities", "core", "layer", "layers",
        }
        return [tok for tok in tokens if tok not in stop_words]

    def _find_responsibility_realization_gaps(
        self,
        *,
        component_name: str,
        responsibilities: List[Any],
        impl_code: str,
        test_code: str = "",
    ) -> List[str]:
        if not responsibilities:
            return []

        haystack = f"{impl_code}\n{test_code}".lower()
        gaps: List[str] = []
        component_tokens = set(self._tokenize_responsibility_text(component_name))

        for idx, responsibility in enumerate(responsibilities, start=1):
            resp_text = str(responsibility or "").strip()
            if not resp_text:
                continue
            resp_tokens = [
                tok for tok in self._tokenize_responsibility_text(resp_text)
                if tok not in component_tokens
            ]
            if not resp_tokens:
                continue
            strong_tokens = [tok for tok in resp_tokens if len(tok) >= 5 or tok.isupper()]
            probe_tokens = strong_tokens[:4] or resp_tokens[:3]
            matched = sum(
                1 for tok in probe_tokens
                if tok in haystack or tok.replace("_", " ") in haystack
            )
            min_required = 2 if len(probe_tokens) >= 3 else 1
            if matched < min_required:
                gaps.append(f"Responsibility {idx} appears weakly realized: {resp_text}")

        return gaps

    def _repair_placeholder_issues(
        self,
        *,
        component_name: str,
        responsibilities: List[Any],
        rel_impl: str,
        impl_code: str,
        rel_test: str,
        test_code: str,
        implemented_components_context: str = "",
    ) -> Tuple[str, str, Dict[str, Any]]:
        rel_test = self._normalize_test_file_path(rel_test, component_name) if rel_test else ""
        hard_issues = self._find_python_placeholder_issues(impl_code, rel_impl)
        if test_code:
            hard_issues.extend(self._find_python_placeholder_issues(test_code, rel_test))
        hard_issues = sorted(set(hard_issues))
        meta: Dict[str, Any] = {
            "attempted": bool(hard_issues),
            "initial_hard_issues": list(hard_issues),
            "remaining_hard_issues": [],
            "soft_warnings": [],
        }
        if not hard_issues:
            warnings = self._find_python_placeholder_warnings(impl_code, rel_impl)
            if test_code:
                warnings.extend(self._find_python_placeholder_warnings(test_code, rel_test))
            meta["soft_warnings"] = sorted(set(warnings))
            return impl_code, test_code, meta
        if not self.llm_client:
            meta["remaining_hard_issues"] = list(hard_issues)
            return impl_code, test_code, meta

        patched_impl, patched_test = impl_code, test_code
        rounds = min(2, max(1, self.post_generation_max_repair_rounds))
        for round_idx in range(rounds):
            related_files = {rel_impl: patched_impl}
            if rel_test and patched_test:
                related_files[rel_test] = patched_test
            task_description = (
                f"Repair concrete placeholder implementations for component '{component_name}' "
                f"(round {round_idx + 1}/{rounds}).\n"
                "Only repair concrete/test placeholders.\n"
                "Do NOT modify abstract/protocol declarations or compatibility fallback stubs.\n"
                "Replace concrete `raise NotImplementedError(...)`, sole-body `pass`, and sole-body `...` "
                "with the smallest executable implementation consistent with the current API and tests.\n\n"
                "Component responsibilities:\n- "
                + "\n- ".join(str(item) for item in responsibilities if str(item).strip())
                + "\n\n"
                "Detected hard placeholder issues:\n- "
                + "\n- ".join(hard_issues)
            )
            if implemented_components_context:
                task_description += "\n\nIMPLEMENTED COMPONENTS CONTEXT:\n" + implemented_components_context
            patch = self._get_patch_agent().generate_patch(
                task_description=task_description,
                related_files=related_files,
                incremental_goal=(
                    "Fix only concrete placeholder implementations and test placeholders. "
                    "Keep abstract Protocol/ABC methods and compatibility fallback stubs untouched."
                ),
                failure_kind="validation_failure",
                telemetry_context={
                    "component_name": component_name,
                    "stage": "placeholder_repair",
                    "round": round_idx + 1,
                    "file_role": "impl_and_test" if rel_test else "impl_only",
                },
            )
            updated_files = self._select_validation_patch_updated_files(
                patch=patch if isinstance(patch, dict) else {},
                component_name=component_name,
                responsibilities=responsibilities,
                rel_impl=rel_impl,
                impl_code=patched_impl,
                rel_test=rel_test,
                test_code=patched_test,
                implemented_components_context=implemented_components_context,
                stage="placeholder_repair",
            )
            patched_impl = str(updated_files.get(rel_impl, patched_impl))
            patched_impl = self._autofix_python_syntax(patched_impl, component_name, rel_impl)
            if rel_test:
                patched_test = str(updated_files.get(rel_test, patched_test))
                if patched_test:
                    patched_test = self._autofix_python_syntax(patched_test, component_name, rel_test)

            hard_issues = self._find_python_placeholder_issues(patched_impl, rel_impl)
            if patched_test:
                hard_issues.extend(self._find_python_placeholder_issues(patched_test, rel_test))
            hard_issues = sorted(set(hard_issues))
            if not hard_issues:
                break

        warnings = self._find_python_placeholder_warnings(patched_impl, rel_impl)
        if patched_test:
            warnings.extend(self._find_python_placeholder_warnings(patched_test, rel_test))
        meta["remaining_hard_issues"] = list(hard_issues)
        meta["soft_warnings"] = sorted(set(warnings))
        meta["repaired"] = not hard_issues
        return patched_impl, patched_test, meta

    def _merge_postprocess_python_issues(
        self,
        *,
        component_name: str,
        responsibilities: List[Any],
        rel_impl: str,
        impl_code: str,
        rel_test: str,
        test_code: str,
        implemented_components_context: str,
    ) -> List[str]:
        issues: List[str] = []
        issues.extend(self._find_python_placeholder_issues(impl_code, rel_impl))
        if test_code:
            issues.extend(self._find_python_placeholder_issues(test_code, rel_test))
        issues.extend(self._find_contract_suspicions(impl_code, rel_impl, implemented_components_context))
        if test_code:
            issues.extend(self._find_contract_suspicions(test_code, rel_test, implemented_components_context))
        issues.extend(self._find_forbidden_peer_repo_imports(impl_code, rel_impl))
        issues.extend(find_structured_contract_issues(impl_code, rel_impl))
        if test_code:
            issues.extend(find_structured_contract_issues(test_code, rel_test))
        issues.extend(
            self._find_responsibility_realization_gaps(
                component_name=component_name,
                responsibilities=responsibilities,
                impl_code=impl_code,
                test_code=test_code,
            )
        )
        return sorted(set(issues))

    def _select_validation_patch_updated_files(
        self,
        *,
        patch: Dict[str, Any],
        component_name: str,
        responsibilities: List[Any],
        rel_impl: str,
        impl_code: str,
        rel_test: str,
        test_code: str,
        implemented_components_context: str,
        stage: str,
    ) -> Dict[str, str]:
        if not isinstance(patch, dict):
            return {}

        full_updates = patch.get("full_file_updated_files", {})
        if not isinstance(full_updates, dict):
            full_updates = {}
        diff_updates = patch.get("diff_updated_files", {})
        if not isinstance(diff_updates, dict):
            diff_updates = {}
        merged_updates = patch.get("updated_files", {})
        if not isinstance(merged_updates, dict):
            merged_updates = {}

        if not diff_updates:
            return dict(full_updates or merged_updates)

        diff_impl = str(diff_updates.get(rel_impl, impl_code))
        diff_test = str(diff_updates.get(rel_test, test_code)) if rel_test else test_code
        diff_impl = self._autofix_python_syntax(diff_impl, component_name, rel_impl)
        if rel_test and diff_test:
            diff_test = self._autofix_python_syntax(diff_test, component_name, rel_test)
        diff_issues = self._merge_postprocess_python_issues(
            component_name=component_name,
            responsibilities=responsibilities,
            rel_impl=rel_impl,
            impl_code=diff_impl,
            rel_test=rel_test,
            test_code=diff_test,
            implemented_components_context=implemented_components_context,
        )
        if not diff_issues:
            logging.info(
                "Validation patch for component '%s' at stage '%s' accepted diff pre-repair",
                component_name,
                stage,
            )
            return dict(diff_updates)

        fallback_updates = dict(full_updates or merged_updates)
        logging.info(
            "Validation patch for component '%s' at stage '%s' rejected diff pre-repair (%d issues remain); falling back to full-file repair",
            component_name,
            stage,
            len(diff_issues),
        )
        return fallback_updates

    def _raise_if_postprocess_still_fails(
        self,
        *,
        component_name: str,
        responsibilities: List[Any],
        rel_impl: str,
        impl_code: str,
        rel_test: str,
        test_code: str,
        implemented_components_context: str,
    ) -> None:
        """Raise RuntimeError with the same shape as the legacy single-round checker when issues remain."""
        remaining = self._find_python_placeholder_issues(impl_code, rel_impl)
        if test_code:
            remaining.extend(self._find_python_placeholder_issues(test_code, rel_test))
        if remaining:
            raise RuntimeError(
                "Post-generation repair left unresolved placeholders: " + "; ".join(sorted(set(remaining)))
            )
        remaining_contract = self._find_contract_suspicions(
            impl_code,
            rel_impl,
            implemented_components_context,
        )
        if test_code:
            remaining_contract.extend(
                self._find_contract_suspicions(
                    test_code,
                    rel_test,
                    implemented_components_context,
                )
            )
        if remaining_contract:
            raise RuntimeError(
                "Post-generation repair left unresolved contract suspicions: "
                + "; ".join(sorted(set(remaining_contract)))
            )
        remaining_forbidden_imports = self._find_forbidden_peer_repo_imports(impl_code, rel_impl)
        if remaining_forbidden_imports:
            raise RuntimeError(
                "Post-generation repair left forbidden peer-repo imports: "
                + "; ".join(sorted(set(remaining_forbidden_imports)))
            )
        remaining_state = find_structured_contract_issues(impl_code, rel_impl)
        if test_code:
            remaining_state.extend(find_structured_contract_issues(test_code, rel_test))
        if remaining_state:
            raise RuntimeError(
                "Post-generation repair left unresolved structured contract issues: "
                + "; ".join(sorted(set(remaining_state)))
            )
        remaining_realization = self._find_responsibility_realization_gaps(
            component_name=component_name,
            responsibilities=responsibilities,
            impl_code=impl_code,
            test_code=test_code,
        )
        if remaining_realization:
            raise RuntimeError(
                "Post-generation reconciliation left weak responsibility realization: "
                + "; ".join(sorted(set(remaining_realization)))
            )

    def _postprocess_python_generation(
        self,
        *,
        component_name: str,
        responsibilities: List[Any],
        rel_impl: str,
        impl_code: str,
        rel_test: str = "",
        test_code: str = "",
        implemented_components_context: str = "",
        stage: str,
    ) -> Tuple[str, str]:
        rel_test = self._normalize_test_file_path(rel_test, component_name) if rel_test else ""
        patched_impl, patched_test = impl_code, test_code
        patched_impl, peer_report = self._repair_forbidden_peer_repo_usage(
            component_name=component_name,
            rel_path=rel_impl,
            code=patched_impl,
            responsibilities=responsibilities,
            stage=stage,
        )
        if not peer_report.get("passed"):
            logging.warning(
                "Peer-repo delegation repair did not fully converge for component '%s' at stage '%s': %s",
                component_name,
                stage,
                "; ".join(peer_report.get("issues", [])),
            )
        issues = self._merge_postprocess_python_issues(
            component_name=component_name,
            responsibilities=responsibilities,
            rel_impl=rel_impl,
            impl_code=patched_impl,
            rel_test=rel_test,
            test_code=test_code,
            implemented_components_context=implemented_components_context,
        )
        patched_impl, patched_test, local_prepass_meta = self._apply_local_validation_repairs(
            component_name=component_name,
            responsibilities=responsibilities,
            rel_impl=rel_impl,
            impl_code=patched_impl,
            rel_test=rel_test,
            test_code=patched_test,
            implemented_components_context=implemented_components_context,
            stage=stage,
        )
        issues = self._merge_postprocess_python_issues(
            component_name=component_name,
            responsibilities=responsibilities,
            rel_impl=rel_impl,
            impl_code=patched_impl,
            rel_test=rel_test,
            test_code=patched_test,
            implemented_components_context=implemented_components_context,
        )
        if not issues:
            return patched_impl, patched_test

        if not self.llm_client:
            logging.warning(
                "Skipping post-generation repair for component '%s' at stage '%s' because LLM is unavailable",
                component_name,
                stage,
            )
            return patched_impl, patched_test

        postprocess_repair_rounds = self.post_generation_max_repair_rounds
        for round_idx in range(postprocess_repair_rounds):
            patched_impl, peer_report = self._repair_forbidden_peer_repo_usage(
                component_name=component_name,
                rel_path=rel_impl,
                code=patched_impl,
                responsibilities=responsibilities,
                stage=f"{stage}_round_{round_idx + 1}",
            )
            issues = self._merge_postprocess_python_issues(
                component_name=component_name,
                responsibilities=responsibilities,
                rel_impl=rel_impl,
                impl_code=patched_impl,
                rel_test=rel_test,
                test_code=patched_test,
                implemented_components_context=implemented_components_context,
            )
            if not issues:
                return patched_impl, patched_test

            task_description = (
                f"Repair generated Python artifacts for component '{component_name}' after {stage} "
                f"(repair round {round_idx + 1}/{postprocess_repair_rounds}).\n"
                "Requirements:\n"
                "1. Remove all TDD placeholders. No `raise NotImplementedError(...)`, no `pass`, and no `...` as the sole body "
                "of a concrete public function or method. Replace each placeholder with the smallest correct executable implementation, "
                "not another placeholder or stub.\n"
                f"2. Align all `{self._primary_python_package_pattern()}` imports and referenced symbols to the IMPLEMENTED COMPONENTS context exactly.\n"
                "3. Reuse the import paths, exported classes, and function names already established by implemented components.\n"
                "4. If the current file needs a small local compatibility alias/wrapper to match an established API, add it in "
                "this file instead of inventing a new external symbol.\n"
                "5. Ensure every listed responsibility is concretely represented by code, tests, public symbols, or inline docs/comments.\n"
                "6. Keep file paths unchanged.\n"
                "7. If the file is syntactically broken, fix syntax first before any broader refactor.\n\n"
                "Component responsibilities:\n- "
                + "\n- ".join(str(item) for item in responsibilities if str(item).strip())
                + "\n\n"
                "Detected issues:\n- "
                + "\n- ".join(issues)
            )
            if implemented_components_context:
                task_description = (
                    f"{task_description}\n\n"
                    "IMPLEMENTED COMPONENTS CONTEXT:\n"
                    f"{implemented_components_context}"
                )

            related_files = {rel_impl: patched_impl}
            if rel_test and patched_test:
                related_files[rel_test] = patched_test
            patch = self._get_patch_agent().generate_patch(
                task_description=task_description,
                related_files=related_files,
                incremental_goal=(
                    "Eliminate placeholders first, restore valid Python syntax, and then align imports/symbol contracts "
                    "without changing file paths."
                ),
                failure_kind="validation_failure",
                telemetry_context={
                    "component_name": component_name,
                    "stage": stage,
                    "round": round_idx + 1,
                    "file_role": "impl_and_test" if rel_test else "impl_only",
                    "local_prepass_issues_before": local_prepass_meta.get("issues_before", 0),
                    "local_prepass_issues_after": local_prepass_meta.get("issues_after", 0),
                },
            )
            updated_files = self._select_validation_patch_updated_files(
                patch=patch if isinstance(patch, dict) else {},
                component_name=component_name,
                responsibilities=responsibilities,
                rel_impl=rel_impl,
                impl_code=patched_impl,
                rel_test=rel_test,
                test_code=patched_test,
                implemented_components_context=implemented_components_context,
                stage=stage,
            )
            patched_impl = str(updated_files.get(rel_impl, patched_impl))
            patched_test = str(updated_files.get(rel_test, patched_test)) if rel_test else patched_test
            patched_impl = self._autofix_python_syntax(patched_impl, component_name, rel_impl)
            if rel_test and patched_test:
                patched_test = self._autofix_python_syntax(patched_test, component_name, rel_test)

        try:
            self._raise_if_postprocess_still_fails(
                component_name=component_name,
                responsibilities=responsibilities,
                rel_impl=rel_impl,
                impl_code=patched_impl,
                rel_test=rel_test,
                test_code=patched_test,
                implemented_components_context=implemented_components_context,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}; exhausted post-generation repair rounds ({postprocess_repair_rounds}) at stage={stage}"
            ) from exc

        return patched_impl, patched_test

    def _fallback_generate_code(
        self,
        component: Dict[str, Any],
        requirement: Dict[str, Any],
        language: str,
        planned_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fallback to template-based code generation when LLM is unavailable."""
        component_name = component.get("name", "Component")
        responsibilities = component.get("responsibilities", [])
        req_name = requirement.get("name", "Requirement")
        req_description = requirement.get("description", "")
        
        if language.lower() == "python":
            code = self._generate_python_template(
                component_name, 
                responsibilities,
                req_description
            )
            file_path = planned_file_path or f"src/{self._to_snake_case(component_name)}.py"
            test_code = self._generate_python_test_template(component_name)
            test_file_path = f"tests/test_{self._to_snake_case(component_name)}.py"
        else:
            code = f"// Template code for {component_name}\n// Requirement: {req_description}\n// TODO: Implement component"
            file_path = planned_file_path or f"src/{component_name}.{language}"
            test_code = ""
            test_file_path = ""
        file_path = self.normalize_file_path(file_path, language=language)
        
        return {
            "component_name": component_name,
            "file_path": file_path,
            "code": code,
            "imports": [],
            "tests": {
                "test_file_path": test_file_path,
                "test_code": test_code
            },
            "documentation": f"# {component_name}\n\nTemplate implementation.",
            "integration_notes": "Generated from template.",
            "language": language,
        }

    def _generate_python_template(self, component_name: str, responsibilities: List[str], requirement_description: str = "") -> str:
        """Generate a Python class template with basic implementation."""
        class_name = component_name.replace(" ", "").replace("-", "")
        methods = []
        
        for i, resp in enumerate(responsibilities[:5]):
            method_name = self._to_snake_case(resp.split()[0] if resp else f"method_{i}")
            # Add basic implementation based on responsibility keywords and requirement
            impl = self._generate_basic_implementation(resp, method_name, requirement_description)
            methods.append(f"""    def {method_name}(self, *args, **kwargs) -> Any:
        \"\"\"
        {resp}
        
        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
            
        Returns:
            Result of the operation
        \"\"\"
{impl}
""")
        
        methods_code = "\n".join(methods)
        
        req_comment = f"\n\nImplements requirement: {requirement_description}" if requirement_description else ""
        
        return f'''"""
{component_name} implementation.

This module provides functionality for {component_name.lower()}.{req_comment}
"""

import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class {class_name}:
    """
    {component_name} class.
    
    Responsibilities:
{chr(10).join(f"    - {resp}" for resp in responsibilities)}
    """
    
    def __init__(self, **config) -> None:
        """Initialize {component_name}.
        
        Args:
            **config: Configuration parameters
        """
        self.config = config
        logger.info(f"Initialized {class_name}")
    
{methods_code}
    def __repr__(self) -> str:
        """String representation."""
        return f"{class_name}(config={{len(self.config)}} items)"
'''

    def _generate_python_test_template(self, component_name: str) -> str:
        """Generate a Python test template."""
        class_name = component_name.replace(" ", "")
        snake_name = self._to_snake_case(component_name)
        
        return f'''"""
Tests for {component_name}.
"""

import pytest
from src.{snake_name} import {class_name}


class Test{class_name}:
    """Test suite for {class_name}."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.{snake_name} = {class_name}()
    
    def test_initialization(self):
        """Test {class_name} initialization."""
        assert self.{snake_name} is not None
    
    # TODO: Add more tests
'''

    def _to_snake_case(self, text: str) -> str:
        """Convert text to snake_case."""
        import re
        text = re.sub(r'[\s-]+', '_', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1_\2', text)
        return text.lower()
    
    def _generate_basic_implementation(self, responsibility: str, method_name: str, requirement_description: str = "") -> str:
        """Generate basic implementation based on responsibility description and requirement context."""
        resp_lower = responsibility.lower()
        req_lower = requirement_description.lower() if requirement_description else ""
        
        # Extract keywords from requirement for context-aware generation
        req_keywords = set(req_lower.split()) if req_lower else set()
        
        # Pattern matching for common operations with requirement context
        if any(word in resp_lower for word in ['validate', 'check', 'verify']):
            context = "Unicode" if "unicode" in req_keywords else "data"
            return f'''        logger.debug(f"Validating {{context}}: {{args}}")
        # Validate {context} according to requirement
        if not args:
            raise ValueError("No data provided for validation")
        # TODO: Add specific validation logic based on requirement
        return True'''
        
        elif any(word in resp_lower for word in ['calculate', 'compute', 'measure', 'width']):
            subject = "width" if "width" in req_keywords else "value"
            return f'''        logger.debug(f"Calculating {subject}: {{args}}")
        # Calculate {subject} based on requirement specifications
        if not args:
            return 0
        # TODO: Implement calculation algorithm from requirement
        result = 0  # Placeholder for actual calculation
        return result'''
        
        elif any(word in resp_lower for word in ['fetch', 'get', 'retrieve', 'load', 'download']):
            source = "Unicode data" if "unicode" in req_keywords else "data"
            return f'''        logger.debug(f"Fetching {source}: {{args}}")
        # Retrieve {source} as specified in requirement
        try:
            # TODO: Implement data retrieval from specified source
            data = {{}}  # Placeholder for fetched data
            return data
        except Exception as e:
            logger.error(f"Failed to fetch {source}: {{e}}")
            raise'''
        
        elif any(word in resp_lower for word in ['save', 'store', 'persist', 'write', 'cache']):
            target = "cache" if "cache" in req_keywords else "storage"
            return f'''        logger.debug(f"Saving to {target}: {{args}}")
        # Persist data to {target} according to requirement
        if not args:
            logger.warning("No data provided to save")
            return False
        # TODO: Implement storage logic
        return True'''
        
        elif any(word in resp_lower for word in ['parse', 'process', 'transform', 'convert']):
            format_type = "Unicode" if "unicode" in req_keywords else "text"
            return f'''        logger.debug(f"Processing {format_type}: {{args}}")
        # Transform {format_type} according to requirement specifications
        if not args:
            return None
        # TODO: Implement transformation logic
        result = args[0] if args else None
        return result'''
        
        elif any(word in resp_lower for word in ['update', 'modify', 'change', 'refresh']):
            return f'''        logger.debug(f"Updating: {{args}}")
        # Update data according to requirement
        if not args:
            logger.warning("No data provided to update")
            return False
        # TODO: Implement update logic based on requirement
        return True'''
        
        elif any(word in resp_lower for word in ['compare', 'diff', 'analyze']):
            return f'''        logger.debug(f"Comparing: {{args}}")
        # Analyze and compare data as per requirement
        if len(args) < 2:
            raise ValueError("Need at least 2 items to compare")
        # TODO: Implement comparison logic
        differences = {{}}
        return differences'''
        
        else:
            context_note = f" (for {requirement_description[:50]}...)" if requirement_description else ""
            return f'''        logger.debug(f"Executing {method_name}: {{args}}")
        # Implement {responsibility.lower()}{context_note}
        # TODO: Add implementation based on requirement specifications
        raise NotImplementedError(f"{method_name} not yet implemented")'''
    
    def extract_component_metadata(
        self,
        code_result: Dict[str, Any],
        requirement_node: str
    ) -> Dict[str, Any]:
        """Extract metadata from generated code for memory registration.
        
        Args:
            code_result: Result from generate_code
            requirement_node: DAG node this component implements
            
        Returns:
            Dictionary with extracted metadata suitable for memory registration
        """
        import re
        
        code = code_result.get("code", "")
        component_name = code_result.get("component_name", "Unknown")
        file_path = code_result.get("file_path", "")
        
        # Extract class names
        class_pattern = r'class\s+(\w+)(?:\(.*?\))?:'
        class_names = re.findall(class_pattern, code)
        
        # Extract function signatures
        function_signatures = []
        func_pattern = r'def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^:]+))?:'
        for match in re.finditer(func_pattern, code, re.MULTILINE):
            func_name = match.group(1)
            params_str = match.group(2).strip()
            return_type = match.group(3).strip() if match.group(3) else "Any"
            
            # Parse parameters
            params = []
            if params_str:
                for param in params_str.split(','):
                    param = param.strip()
                    if param and param != 'self' and param != 'cls':
                        # Extract parameter name (before colon if type hint exists)
                        param_name = param.split(':')[0].strip().split('=')[0].strip()
                        params.append(param_name)
            
            function_signatures.append({
                "name": func_name,
                "params": params,
                "return_type": return_type
            })
        
        # Extract imports to identify dependencies
        import_pattern = r'(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))'
        imports = re.findall(import_pattern, code)
        dependencies = []
        for from_module, import_module in imports:
            module = from_module or import_module
            if module and not module.startswith('.'):
                # Only include project-level imports (exclude stdlib)
                if not any(module.startswith(stdlib) for stdlib in ['os', 'sys', 'json', 'logging', 'typing', 're', 'pathlib']):
                    dependencies.append(module)
        
        # Exports are typically public classes and functions (not starting with _)
        exports = [name for name in class_names if not name.startswith('_')]
        public_functions = [sig['name'] for sig in function_signatures if not sig['name'].startswith('_')]
        exports.extend(public_functions[:10])  # Limit to avoid too many exports
        structured_contract_facts = extract_structured_contract_facts(code, file_path)
        structured_contract_issues = find_structured_contract_issues(code, file_path)
        generation_status = str(code_result.get("generation_status") or self._derive_generation_status(code_result))
        
        return {
            "component_name": component_name,
            "requirement_node": requirement_node,
            "file_path": file_path,
            "class_names": class_names,
            "function_signatures": function_signatures,
            "dependencies": list(set(dependencies)),
            "exports": list(set(exports)),
            "status": generation_status,
            "structured_contract_facts": structured_contract_facts,
            "structured_contract_issues": structured_contract_issues,
            "generation_status": generation_status,
            "tdd_final_pytest_rc": code_result.get("tdd_final_pytest_rc"),
            "tdd_passed": code_result.get("tdd_passed", generation_status != "retained_after_tdd_failure"),
        }
    @staticmethod
    def _state_contract_guidance_block() -> str:
        return (
            "State Contract Rules:\n"
            "1. Keep internal container/state shapes consistent across the file.\n"
            "2. If `self.x` is used with dict methods like `setdefault/get/items/update`, initialize and keep it as a dict-like object.\n"
            "3. Do not assign the same field to an opaque registry/manager object in one branch and use it like a dict in another branch.\n"
            "4. Prefer storing wrapper objects in distinct fields (for example `self._inner_registry`) instead of reusing container fields.\n"
            "5. Avoid heavy top-level side effects. Do not instantiate complex runtime/platform objects at import time unless strictly necessary.\n"
            "6. Prefer lazy initialization/factory helpers for registries, executors, adapters, and integration platforms.\n"
        )
