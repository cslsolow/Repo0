"""Patch agent for targeted fixes and incremental code evolution."""

from __future__ import annotations

import difflib
import fcntl
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from agents.infra.llm_client import LLMClient
except Exception:
    LLMClient = None  # type: ignore[assignment]


class PatchAgent:
    """Generate and apply code patches for bug fixes and incremental development."""

    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.output_dir = output_dir
        self.stats_dir = Path(os.path.dirname(output_dir) or ".")
        self.patch_events_file = self.stats_dir / "patch_agent_events.json"
        self.patch_summary_file = self.stats_dir / "patch_agent_analysis.json"
        self.patch_optimization_report_file = self.stats_dir / "patch_agent_optimization_report.json"
        self.patch_lock_file = self.stats_dir / "patch_agent_events.lock"
        self.patch_token_usage_file = self.stats_dir / "token_usage_patch_agent.json"
        self.llm_client = None
        if self.api_config.get("api_key") and LLMClient is not None:
            self.llm_client = LLMClient(self.api_config, output_dir, agent_name="patch_agent")

    def generate_patch(
        self,
        task_description: str,
        related_files: Dict[str, str],
        compile_error: str = "",
        incremental_goal: str = "",
        failure_kind: str = "",
        telemetry_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a patch proposal and optional full-file updates.

        Returns a dict with:
        - patch: unified diff text
        - updated_files: map of path -> new content
        - touched_files: list of file paths
        - mode: llm | heuristic
        """
        call_id = str(uuid.uuid4())
        telemetry = self._build_call_telemetry(
            call_id=call_id,
            task_description=task_description,
            related_files=related_files,
            compile_error=compile_error,
            incremental_goal=incremental_goal,
            failure_kind=failure_kind,
            telemetry_context=telemetry_context,
        )
        telemetry["scenario"] = self._detect_patch_scenario(
            task_description,
            compile_error,
            incremental_goal,
            failure_kind=failure_kind,
        )
        if self.llm_client:
            try:
                result = self._generate_patch_with_llm(
                    task_description=task_description,
                    related_files=related_files,
                    compile_error=compile_error,
                    incremental_goal=incremental_goal,
                    failure_kind=failure_kind,
                    telemetry=telemetry,
                )
                self._record_patch_event(
                    telemetry=telemetry,
                    result=result,
                    mode="llm",
                    llm_succeeded=True,
                )
                return result
            except Exception as exc:
                telemetry["llm_error"] = str(exc)
                logging.warning("PatchAgent LLM generation failed (%s), fallback to heuristic", exc)

        heuristic = self._generate_patch_heuristic(
            task_description=task_description,
            related_files=related_files,
            compile_error=compile_error,
            incremental_goal=incremental_goal,
            failure_kind=failure_kind,
        )
        heuristic["mode"] = "heuristic"
        self._record_patch_event(
            telemetry=telemetry,
            result=heuristic,
            mode="heuristic",
            llm_succeeded=False,
        )
        return heuristic

    def apply_patch_text(
        self,
        patch_text: str,
        related_files: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Apply standard unified diff patch text against in-memory files.

        Returns dict with:
        - updated_files: full updated file map
        - created_files: created file paths
        - deleted_files: deleted file paths
        - touched_files: changed file paths
        """
        if not self._is_strict_unified_diff(patch_text):
            raise ValueError("Patch text must be a standard unified diff starting with ---/+++ hunks; apply_patch-style markers are not accepted.")
        sections = self._split_unified_patch_sections(patch_text)
        updated_files = dict(related_files)
        created_files: List[str] = []
        deleted_files: List[str] = []
        touched_files: List[str] = []

        for old_path, new_path, hunks in sections:
            norm_old = self._normalize_diff_path(old_path)
            norm_new = self._normalize_diff_path(new_path)
            is_new_file = old_path.strip() == "/dev/null"
            is_deleted_file = new_path.strip() == "/dev/null"

            if is_deleted_file:
                if norm_old in updated_files:
                    del updated_files[norm_old]
                    deleted_files.append(norm_old)
                    touched_files.append(norm_old)
                continue

            target_path = norm_new or norm_old
            if not target_path:
                continue

            original = "" if is_new_file else updated_files.get(target_path, updated_files.get(norm_old, ""))
            patched = self._apply_hunks_to_text(original, hunks)

            updated_files[target_path] = patched
            touched_files.append(target_path)
            if is_new_file and target_path not in related_files:
                created_files.append(target_path)

        return {
            "updated_files": updated_files,
            "created_files": sorted(set(created_files)),
            "deleted_files": sorted(set(deleted_files)),
            "touched_files": sorted(set(touched_files)),
        }

    def write_files(self, repo_root: str | Path, updated_files: Dict[str, str]) -> List[str]:
        """Persist updated file map to disk under repo_root."""
        root = Path(repo_root)
        written: List[str] = []
        for rel_path, content in updated_files.items():
            full_path = root / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            written.append(str(full_path))
        return written

    def collect_related_files(
        self,
        repo_root: str | Path,
        file_paths: List[str],
        max_file_chars: int = 12000,
    ) -> Dict[str, str]:
        """Load candidate files as context for patch generation."""
        root = Path(repo_root)
        collected: Dict[str, str] = {}
        for rel in file_paths:
            path = root / rel
            if not path.exists() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
                collected[rel] = text[:max_file_chars]
            except Exception:
                continue
        return collected

    def _generate_patch_with_llm(
        self,
        task_description: str,
        related_files: Dict[str, str],
        compile_error: str,
        incremental_goal: str,
        failure_kind: str,
        telemetry: Dict[str, Any],
    ) -> Dict[str, Any]:
        assert self.llm_client is not None

        scenario = str(telemetry.get("scenario", "") or self._detect_patch_scenario(
            task_description,
            compile_error,
            incremental_goal,
            failure_kind=failure_kind,
        ))
        files_payload = self._build_related_files_prompt_payload(
            related_files=related_files,
            failure_text=compile_error,
            scenario=scenario,
        )
        context_label, repair_focus = self._classify_failure_context(
            scenario=scenario,
            failure_text=compile_error,
            task_description=task_description,
            incremental_goal=incremental_goal,
            failure_kind=failure_kind,
        )
        context_blocks: List[str] = [f"Primary Task:\\n{task_description or 'N/A'}"]
        compact_failure_text = self._compress_failure_context(compile_error, scenario=scenario)

        if scenario in {"compile_error_fix", "syntax_failure", "import_failure"}:
            context_blocks.append(
                f"{context_label}:\\n{compact_failure_text}"
            )
            if incremental_goal:
                context_blocks.append(f"Additional Goal:\\n{incremental_goal}")
        elif scenario in {"incremental_feature", "feature_update", "peer_repo_delegation"}:
            context_blocks.append(f"Feature Goal:\\n{incremental_goal or task_description}")
            if compact_failure_text:
                context_blocks.append(f"{context_label} (optional):\\n{compact_failure_text}")
        else:
            if incremental_goal:
                context_blocks.append(f"Bug Fix Scope:\\n{incremental_goal}")
            if compact_failure_text:
                context_blocks.append(f"{context_label} (if relevant):\\n{compact_failure_text}")

        context_text = "\n\n".join(context_blocks)
        telemetry["scenario"] = scenario
        telemetry["context_label"] = context_label
        telemetry["repair_focus"] = repair_focus
        telemetry["prompt_chars"] = sum(len(str(item)) for item in files_payload) + len(context_text)
        logging.info(
            "PatchAgent LLM request starting: scenario=%s files=%d raw_chars=%d prompt_chars=%d",
            scenario,
            len(related_files),
            sum(len(str(content)) for content in related_files.values()),
            telemetry["prompt_chars"],
        )

        full_response = self._call_patch_json(
            prompt=self._build_full_file_prompt(
                scenario=scenario,
                context_label=context_label,
                repair_focus=repair_focus,
                context_text=context_text,
                files_payload=files_payload,
            ),
            telemetry=telemetry,
            scenario=scenario,
            operation_suffix="full_file",
        )
        if isinstance(full_response, list):
            raise RuntimeError("PatchAgent expected a JSON object for full-file request, got list")

        updated_files_raw = full_response.get("updated_files", [])
        full_file_updated_files: Dict[str, str] = {}
        updated_files: Dict[str, str] = {}

        if isinstance(updated_files_raw, list):
            for item in updated_files_raw:
                if isinstance(item, dict) and item.get("path"):
                    full_file_updated_files[str(item["path"])] = str(item.get("content", ""))
        updated_files = dict(full_file_updated_files)

        touched = full_response.get("touched_files")
        touched_files = touched if isinstance(touched, list) else sorted(updated_files.keys())

        return {
            "patch": "",
            "updated_files": updated_files,
            "diff_updated_files": {},
            "full_file_updated_files": full_file_updated_files,
            "touched_files": touched_files,
            "summary": str(full_response.get("summary", "")),
            "mode": "llm",
        }

    def _call_patch_json(
        self,
        *,
        prompt: str,
        telemetry: Dict[str, Any],
        scenario: str,
        operation_suffix: str,
    ) -> Any:
        assert self.llm_client is not None
        usage_metadata = self._build_usage_metadata(telemetry, scenario)
        usage_metadata["patch_output_mode"] = operation_suffix
        return self.llm_client.call_json(
            [
                {
                    "role": "system",
                    "content": "You write precise, minimal patches and always output valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=32768,
            operation_name=f"patch_agent.{scenario}.{operation_suffix}",
            usage_metadata=usage_metadata,
        )

    @staticmethod
    def _build_full_file_prompt(
        *,
        scenario: str,
        context_label: str,
        repair_focus: str,
        context_text: str,
        files_payload: Any,
    ) -> str:
        return f"""You are a senior software engineer.

Scenario: {scenario}
Failure Context Type: {context_label}
Repair Focus: {repair_focus}

{context_text}

Related Files:
{files_payload}

Generate the corrected full-file outputs for the current scenario.
Use the failure context type above to adapt the repair strategy. Do not assume every failure is a compile error.
Return ONLY full updated file contents in JSON.
Do NOT return any diff syntax, patch markers, or apply_patch-style blocks in file content.
Return ONLY JSON:
{{
  "updated_files": [{{"path": "relative/path.py", "content": "full new file content"}}],
  "touched_files": ["relative/path.py"],
  "summary": "short summary"
}}
"""

    def _build_call_telemetry(
        self,
        *,
        call_id: str,
        task_description: str,
        related_files: Dict[str, str],
        compile_error: str,
        incremental_goal: str,
        failure_kind: str,
        telemetry_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        paths = sorted(str(path) for path in related_files.keys())
        file_sizes = {str(path): len(str(content)) for path, content in related_files.items()}
        failure_text = str(compile_error or "")
        signature = ""
        for line in failure_text.splitlines():
            stripped = line.strip()
            if stripped:
                signature = stripped[:240]
                break
        telemetry: Dict[str, Any] = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "failure_kind": str(failure_kind or ""),
            "task_description_chars": len(str(task_description or "")),
            "incremental_goal_chars": len(str(incremental_goal or "")),
            "failure_chars": len(failure_text),
            "file_count": len(related_files),
            "related_file_paths": paths,
            "related_file_sizes": file_sizes,
            "raw_chars": sum(file_sizes.values()),
            "largest_file_chars": max(file_sizes.values(), default=0),
            "failure_signature": signature,
        }
        if telemetry_context:
            telemetry.update({str(k): v for k, v in telemetry_context.items()})
        return telemetry

    @staticmethod
    def _build_usage_metadata(telemetry: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        metadata = {
            "call_id": telemetry.get("call_id"),
            "scenario": scenario,
            "failure_kind": telemetry.get("failure_kind", ""),
            "component_name": telemetry.get("component_name", ""),
            "parent_task": telemetry.get("parent_task", ""),
            "stage": telemetry.get("stage", ""),
            "round": telemetry.get("round"),
            "file_count": telemetry.get("file_count", 0),
            "raw_chars": telemetry.get("raw_chars", 0),
            "prompt_chars": telemetry.get("prompt_chars", 0),
            "failure_chars": telemetry.get("failure_chars", 0),
            "largest_file_chars": telemetry.get("largest_file_chars", 0),
            "failure_signature": telemetry.get("failure_signature", ""),
        }
        return metadata

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
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_patch_agent_", suffix=".json", dir=str(path.parent))
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

    def _with_patch_lock(self, callback) -> Any:
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        with open(self.patch_lock_file, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                return callback()
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _record_patch_event(
        self,
        *,
        telemetry: Dict[str, Any],
        result: Dict[str, Any],
        mode: str,
        llm_succeeded: bool,
    ) -> None:
        event = dict(telemetry)
        event["mode"] = mode
        event["llm_succeeded"] = bool(llm_succeeded)
        event["touched_files"] = sorted(str(x) for x in (result.get("touched_files") or []))
        updated_files = result.get("updated_files") if isinstance(result.get("updated_files"), dict) else {}
        event["updated_files_count"] = len(updated_files)
        event["touched_files_count"] = len(event["touched_files"])
        event["patch_chars"] = len(str(result.get("patch", "")))
        event["summary"] = str(result.get("summary", ""))
        event["zero_update"] = event["updated_files_count"] == 0 and event["touched_files_count"] == 0
        event["fallback_used"] = mode != "llm"
        event["llm_error"] = str(telemetry.get("llm_error", ""))

        def _persist() -> None:
            events = self._load_json_file(self.patch_events_file, [])
            if not isinstance(events, list):
                events = []
            events.append(event)
            self._atomic_write_json(self.patch_events_file, events)
            summary = self._build_patch_summary(events)
            self._atomic_write_json(self.patch_summary_file, summary)
            optimization = self._build_patch_optimization_report(events, summary)
            self._atomic_write_json(self.patch_optimization_report_file, optimization)

        self._with_patch_lock(_persist)

    def _build_patch_summary(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        token_records = self._load_json_file(self.patch_token_usage_file, [])
        token_by_call: Dict[str, Dict[str, Any]] = {}
        if isinstance(token_records, list):
            for record in token_records:
                if not isinstance(record, dict):
                    continue
                metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                call_id = str(metadata.get("call_id", "") or "")
                if not call_id:
                    continue
                agg = token_by_call.setdefault(
                    call_id,
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_prompt_tokens": 0},
                )
                agg["prompt_tokens"] += int(record.get("prompt_tokens", 0) or 0)
                agg["completion_tokens"] += int(record.get("completion_tokens", 0) or 0)
                agg["total_tokens"] += int(record.get("total_tokens", 0) or 0)
                agg["cached_prompt_tokens"] += int(record.get("cached_prompt_tokens", 0) or 0)

        summary: Dict[str, Any] = {
            "updated_at": datetime.now().isoformat(),
            "total_calls": len(events),
            "llm_calls": 0,
            "heuristic_calls": 0,
            "fallback_calls": 0,
            "zero_update_calls": 0,
            "llm_error_calls": 0,
            "totals": {
                "raw_chars": 0,
                "prompt_chars": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "by_scenario": {},
            "by_failure_kind": {},
            "by_component": {},
            "top_failure_signatures": [],
            "top_expensive_calls": [],
        }
        failure_signatures: Dict[str, int] = {}
        expensive_calls: List[Dict[str, Any]] = []

        def _bucket(container: Dict[str, Any], key: str) -> Dict[str, Any]:
            return container.setdefault(
                key or "<unknown>",
                {
                    "calls": 0,
                    "llm_calls": 0,
                    "heuristic_calls": 0,
                    "fallback_calls": 0,
                    "zero_update_calls": 0,
                    "llm_error_calls": 0,
                    "raw_chars": 0,
                    "prompt_chars": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )

        for event in events:
            if not isinstance(event, dict):
                continue
            mode = str(event.get("mode", ""))
            if mode == "llm":
                summary["llm_calls"] += 1
            else:
                summary["heuristic_calls"] += 1
            if event.get("fallback_used"):
                summary["fallback_calls"] += 1
            if event.get("zero_update"):
                summary["zero_update_calls"] += 1
            if event.get("llm_error"):
                summary["llm_error_calls"] += 1

            raw_chars = int(event.get("raw_chars", 0) or 0)
            prompt_chars = int(event.get("prompt_chars", 0) or 0)
            summary["totals"]["raw_chars"] += raw_chars
            summary["totals"]["prompt_chars"] += prompt_chars

            call_tokens = token_by_call.get(str(event.get("call_id", "")), {})
            for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                summary["totals"][token_key] += int(call_tokens.get(token_key, 0) or 0)

            for container, key in (
                (summary["by_scenario"], str(event.get("scenario", "") or "<unknown>")),
                (summary["by_failure_kind"], str(event.get("failure_kind", "") or "<unknown>")),
                (summary["by_component"], str(event.get("component_name", "") or "<unknown>")),
            ):
                bucket = _bucket(container, key)
                bucket["calls"] += 1
                if mode == "llm":
                    bucket["llm_calls"] += 1
                else:
                    bucket["heuristic_calls"] += 1
                if event.get("fallback_used"):
                    bucket["fallback_calls"] += 1
                if event.get("zero_update"):
                    bucket["zero_update_calls"] += 1
                if event.get("llm_error"):
                    bucket["llm_error_calls"] += 1
                bucket["raw_chars"] += raw_chars
                bucket["prompt_chars"] += prompt_chars
                bucket["prompt_tokens"] += int(call_tokens.get("prompt_tokens", 0) or 0)
                bucket["completion_tokens"] += int(call_tokens.get("completion_tokens", 0) or 0)
                bucket["total_tokens"] += int(call_tokens.get("total_tokens", 0) or 0)

            signature = str(event.get("failure_signature", "") or "").strip()
            if signature:
                failure_signatures[signature] = failure_signatures.get(signature, 0) + 1

            expensive_calls.append(
                {
                    "call_id": event.get("call_id"),
                    "component_name": event.get("component_name", ""),
                    "scenario": event.get("scenario", ""),
                    "failure_kind": event.get("failure_kind", ""),
                    "mode": mode,
                    "prompt_tokens": int(call_tokens.get("prompt_tokens", 0) or 0),
                    "total_tokens": int(call_tokens.get("total_tokens", 0) or 0),
                    "prompt_chars": prompt_chars,
                    "raw_chars": raw_chars,
                    "file_count": int(event.get("file_count", 0) or 0),
                    "zero_update": bool(event.get("zero_update")),
                    "failure_signature": signature,
                }
            )

        summary["top_failure_signatures"] = [
            {"failure_signature": sig, "calls": count}
            for sig, count in sorted(failure_signatures.items(), key=lambda item: (-item[1], item[0]))[:20]
        ]
        summary["top_expensive_calls"] = sorted(
            expensive_calls,
            key=lambda item: (-(item.get("prompt_tokens", 0) or 0), -(item.get("prompt_chars", 0) or 0)),
        )[:50]
        return summary

    def _build_patch_optimization_report(
        self,
        events: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "updated_at": datetime.now().isoformat(),
            "overview": {
                "total_calls": int(summary.get("total_calls", 0) or 0),
                "llm_calls": int(summary.get("llm_calls", 0) or 0),
                "heuristic_calls": int(summary.get("heuristic_calls", 0) or 0),
                "fallback_calls": int(summary.get("fallback_calls", 0) or 0),
                "zero_update_calls": int(summary.get("zero_update_calls", 0) or 0),
                "llm_error_calls": int(summary.get("llm_error_calls", 0) or 0),
                "overall_effective_update_rate": 0.0,
                "overall_fallback_rate": 0.0,
                "overall_zero_update_rate": 0.0,
            },
            "scenario_decisions": [],
            "failure_kind_decisions": [],
            "component_hotspots": [],
            "top_failure_signatures": list(summary.get("top_failure_signatures", [])),
            "top_expensive_calls": list(summary.get("top_expensive_calls", [])),
        }

        total_calls = report["overview"]["total_calls"]
        if total_calls:
            report["overview"]["overall_effective_update_rate"] = round(
                1.0 - (report["overview"]["zero_update_calls"] / total_calls), 4
            )
            report["overview"]["overall_fallback_rate"] = round(
                report["overview"]["fallback_calls"] / total_calls, 4
            )
            report["overview"]["overall_zero_update_rate"] = round(
                report["overview"]["zero_update_calls"] / total_calls, 4
            )

        def _decision_rows(container: Dict[str, Any], label: str) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            for key, bucket in sorted(container.items(), key=lambda item: (-int(item[1].get("total_tokens", 0) or 0), item[0])):
                calls = int(bucket.get("calls", 0) or 0)
                if not calls:
                    continue
                zero = int(bucket.get("zero_update_calls", 0) or 0)
                fallback = int(bucket.get("fallback_calls", 0) or 0)
                llm_error = int(bucket.get("llm_error_calls", 0) or 0)
                total_tokens = int(bucket.get("total_tokens", 0) or 0)
                prompt_tokens = int(bucket.get("prompt_tokens", 0) or 0)
                raw_chars = int(bucket.get("raw_chars", 0) or 0)
                prompt_chars = int(bucket.get("prompt_chars", 0) or 0)
                rows.append(
                    {
                        label: key,
                        "calls": calls,
                        "effective_updates": calls - zero,
                        "effective_update_rate": round((calls - zero) / calls, 4),
                        "fallback_rate": round(fallback / calls, 4),
                        "zero_update_rate": round(zero / calls, 4),
                        "llm_error_rate": round(llm_error / calls, 4),
                        "avg_total_tokens": round(total_tokens / calls, 2),
                        "avg_prompt_tokens": round(prompt_tokens / calls, 2),
                        "avg_raw_chars": round(raw_chars / calls, 2),
                        "avg_prompt_chars": round(prompt_chars / calls, 2),
                        "total_tokens": total_tokens,
                        "total_prompt_tokens": prompt_tokens,
                    }
                )
            return rows

        report["scenario_decisions"] = _decision_rows(summary.get("by_scenario", {}), "scenario")
        report["failure_kind_decisions"] = _decision_rows(summary.get("by_failure_kind", {}), "failure_kind")

        component_rows = _decision_rows(summary.get("by_component", {}), "component_name")
        report["component_hotspots"] = component_rows[:50]
        return report

    @staticmethod
    def _normalize_path_for_match(path: str) -> str:
        return str(path or "").strip().replace("\\", "/").lstrip("./")

    def _extract_focus_lines(
        self,
        failure_text: str,
        related_files: Dict[str, str],
    ) -> Dict[str, Set[int]]:
        focus: Dict[str, Set[int]] = {}
        if not failure_text or not related_files:
            return focus

        normalized_candidates = {
            self._normalize_path_for_match(path): path
            for path in related_files.keys()
        }
        patterns = [
            re.compile(r'File "([^"]+)", line (\d+)'),
            re.compile(r"([A-Za-z0-9_./\\\\-]+\\.py):(\d+)"),
            re.compile(r"^([^:\n]+\\.py):(\d+):"),
        ]
        for raw_line in failure_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if ".py:" in line:
                parts = line.split(":", 2)
                if len(parts) >= 2 and parts[0].endswith(".py") and parts[1].isdigit():
                    raw_path = self._normalize_path_for_match(parts[0])
                    line_no = max(1, int(parts[1]))
                    for candidate_norm, original_path in normalized_candidates.items():
                        if (
                            raw_path == candidate_norm
                            or raw_path.endswith(candidate_norm)
                            or candidate_norm.endswith(raw_path)
                        ):
                            focus.setdefault(original_path, set()).add(line_no)
            for pattern in patterns:
                for match in pattern.finditer(line):
                    raw_path = self._normalize_path_for_match(match.group(1))
                    try:
                        line_no = int(match.group(2))
                    except Exception:
                        continue
                    for candidate_norm, original_path in normalized_candidates.items():
                        if raw_path.endswith(candidate_norm) or candidate_norm.endswith(raw_path):
                            focus.setdefault(original_path, set()).add(max(1, line_no))
        return focus

    @staticmethod
    def _render_excerpt(content: str, focus_lines: Set[int], radius: int = 24) -> Tuple[str, List[Tuple[int, int]]]:
        lines = content.splitlines()
        if not lines:
            return "", []
        if not focus_lines:
            max_head = min(len(lines), 80)
            excerpt = "\n".join(f"{idx + 1:>5}: {lines[idx]}" for idx in range(max_head))
            return excerpt, [(1, max_head)]

        ranges: List[Tuple[int, int]] = []
        for line_no in sorted(focus_lines):
            start = max(1, line_no - radius)
            end = min(len(lines), line_no + radius)
            if ranges and start <= ranges[-1][1] + 1:
                prev_start, prev_end = ranges[-1]
                ranges[-1] = (prev_start, max(prev_end, end))
            else:
                ranges.append((start, end))

        rendered: List[str] = []
        for idx, (start, end) in enumerate(ranges):
            if idx:
                rendered.append("...")
            rendered.extend(
                f"{line_no:>5}: {lines[line_no - 1]}"
                for line_no in range(start, end + 1)
            )
        return "\n".join(rendered), ranges

    def _build_related_files_prompt_payload(
        self,
        *,
        related_files: Dict[str, str],
        failure_text: str,
        scenario: str,
        max_full_chars: int = 12000,
    ) -> List[Dict[str, Any]]:
        focus_map = self._extract_focus_lines(failure_text, related_files)
        payload: List[Dict[str, Any]] = []
        for path, content in related_files.items():
            normalized = self._normalize_path_for_match(path)
            focus_lines = focus_map.get(path, set()) or focus_map.get(normalized, set())
            payload.append(
                {
                    "path": path,
                    "content": content,
                    "content_mode": "full",
                    "total_chars": len(content),
                    "focus_lines": sorted(focus_lines),
                }
            )
        return payload

    @staticmethod
    def _compress_failure_context(
        failure_text: str,
        *,
        scenario: str,
        max_chars: Optional[int] = None,
    ) -> str:
        return str(failure_text or "").strip()

    @staticmethod
    def _detect_patch_scenario(
        task_description: str,
        compile_error: str,
        incremental_goal: str,
        *,
        failure_kind: str = "",
    ) -> str:
        kind = str(failure_kind or "").strip().lower()
        if kind == "import_failure":
            return "import_failure"
        if kind == "syntax_failure":
            return "syntax_failure"
        if kind == "validation_failure":
            return "validation_failure"
        if kind == "feature_update":
            return "feature_update"
        if kind == "peer_repo_delegation":
            return "peer_repo_delegation"
        if kind == "test_failure":
            lower_failure = str(compile_error or "").lower()
            if any(tok in lower_failure for tok in ["modulenotfounderror", "importerror", "cannot import name", "no module named"]):
                return "import_failure"
            if any(tok in lower_failure for tok in ["syntaxerror", "indentationerror", "unterminated", "invalid syntax"]):
                return "syntax_failure"
            return "test_failure"
        if compile_error.strip():
            return "compile_error_fix"
        combined = f"{task_description} {incremental_goal}".lower()
        if any(token in combined for token in ["feature", "new requirement", "implement", "incremental", "add"]):
            return "incremental_feature"
        return "bug_fix"

    @staticmethod
    def _extract_relevant_failure_block(failure_text: str, *, scenario: str) -> str:
        lines = failure_text.splitlines()
        if not lines:
            return failure_text

        def collect(keyword_set: Tuple[str, ...], radius: int) -> str:
            lowered = [line.lower() for line in lines]
            hits = [idx for idx, line in enumerate(lowered) if any(keyword in line for keyword in keyword_set)]
            if not hits:
                return failure_text
            ranges: List[Tuple[int, int]] = []
            for idx in hits:
                start = max(0, idx - radius)
                end = min(len(lines) - 1, idx + radius)
                if ranges and start <= ranges[-1][1] + 1:
                    ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
                else:
                    ranges.append((start, end))
            out: List[str] = []
            for block_idx, (start, end) in enumerate(ranges):
                if block_idx:
                    out.append("... [context omitted] ...")
                out.extend(lines[start:end + 1])
            return "\n".join(out)

        if scenario == "import_failure":
            return collect(("modulenotfounderror", "importerror", "cannot import name", "no module named"), 8)
        if scenario in {"syntax_failure", "compile_error_fix"}:
            return collect(("syntaxerror", "indentationerror", "unterminated", "invalid syntax", "unexpected indent"), 10)
        if scenario == "test_failure":
            return collect(("failed", "assert", "error collecting", "nameerror", "attributeerror", "typeerror"), 10)
        if scenario == "validation_failure":
            return collect(("placeholder", "notimplementederror", "contract", "detected issues"), 8)
        return failure_text

    @staticmethod
    def _classify_failure_context(
        *,
        scenario: str,
        failure_text: str,
        task_description: str,
        incremental_goal: str,
        failure_kind: str,
    ) -> Tuple[str, str]:
        kind = str(failure_kind or "").strip().lower()
        if kind == "test_failure":
            return "Test Failure Details", "Fix behavior to satisfy the failing test without over-editing unrelated files."
        if kind == "import_failure":
            return "Import Failure Details", "Repair import paths, module initialization, and import-time side effects."
        if kind == "syntax_failure":
            return "Syntax Error Details", "Repair syntax and keep changes minimal around the failing region."
        if kind == "validation_failure":
            return "Validation Failure Details", "Resolve placeholders or contract mismatches while preserving the planned API."
        if kind == "feature_update":
            return "Feature Update Details", "Implement the requested increment with the smallest coherent patch."

        text = f"{failure_text}\n{task_description}\n{incremental_goal}".lower()
        if "assert" in text or "pytest" in text or "expected" in text and "got" in text:
            return "Test Failure Details", "Fix behavior to satisfy the failing test without over-editing unrelated files."
        if any(tok in text for tok in ["importerror", "modulenotfounderror", "cannot import", "import_module", "traceback"]):
            return "Import Failure Details", "Repair import paths, module initialization, and import-time side effects."
        if any(tok in text for tok in ["syntaxerror", "indentationerror", "unterminated", "invalid syntax", "unmatched", "was never closed"]):
            return "Syntax Error Details", "Repair syntax and keep changes minimal around the failing region."
        if any(tok in text for tok in ["placeholder", "notimplementederror", "tdd placeholder", "responsibility", "contract"]):
            return "Validation Failure Details", "Resolve placeholders or contract mismatches while preserving the planned API."
        if scenario == "incremental_feature":
            return "Feature Update Details", "Implement the requested increment with the smallest coherent patch."
        return "Runtime Failure Details", "Repair the runtime failure indicated by the provided context."

    def _generate_patch_heuristic(
        self,
        task_description: str,
        related_files: Dict[str, str],
        compile_error: str,
        incremental_goal: str,
        failure_kind: str = "",
    ) -> Dict[str, Any]:
        updated_files = dict(related_files)
        touched: List[str] = []

        if compile_error:
            file_path, line_no = self._extract_file_and_line_from_error(compile_error)
            resolved_path = self._resolve_related_path(file_path, updated_files) if file_path else None
            if resolved_path:
                new_content = self._try_fix_compile_error(updated_files[resolved_path], line_no, compile_error)
                if new_content != updated_files[resolved_path]:
                    updated_files[resolved_path] = new_content
                    touched.append(resolved_path)

        if not touched and incremental_goal:
            target = self._choose_target_file_for_increment(related_files)
            if target:
                old = updated_files[target]
                new = self._apply_incremental_stub(old, incremental_goal, task_description)
                if new != old:
                    updated_files[target] = new
                    touched.append(target)

        patch_text = ""
        for path in sorted(set(touched)):
            old = related_files.get(path, "")
            new = updated_files.get(path, "")
            patch_text += self._build_unified_diff(path, old, new)

        return {
            "patch": patch_text,
            "updated_files": updated_files,
            "touched_files": sorted(set(touched)),
            "summary": "heuristic patch generated",
        }

    @staticmethod
    def _extract_file_and_line_from_error(compile_error: str) -> Tuple[Optional[str], Optional[int]]:
        # Python traceback pattern: File "path", line N
        m = re.search(r'File\s+"([^"]+)"\s*,\s*line\s+(\d+)', compile_error)
        if not m:
            return None, None
        path = m.group(1)
        # Keep only relative-looking suffix if traceback contains absolute paths.
        path = path.replace("\\", "/")
        line_no = int(m.group(2))
        return path, line_no

    def _try_fix_compile_error(self, content: str, line_no: Optional[int], compile_error: str) -> str:
        lines = content.splitlines()
        if not lines:
            return content

        lower_error = compile_error.lower()

        if line_no is not None and 1 <= line_no <= len(lines):
            idx = line_no - 1

            if "syntaxerror" in lower_error:
                current = lines[idx]
                if self._looks_like_missing_colon_line(current):
                    lines[idx] = current.rstrip() + ":"
                    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")

            if "indentationerror" in lower_error:
                indent = self._line_indent(lines[idx]) + "    "
                lines.insert(idx + 1, f"{indent}pass")
                return "\n".join(lines) + ("\n" if content.endswith("\n") else "")

        return content

    @staticmethod
    def _line_indent(line: str) -> str:
        return line[: len(line) - len(line.lstrip(" "))]

    @staticmethod
    def _looks_like_missing_colon_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            return False
        return bool(re.match(r"^(if|for|while|def|class|elif|else|try|except|finally|with)\b", stripped))

    @staticmethod
    def _choose_target_file_for_increment(related_files: Dict[str, str]) -> Optional[str]:
        if not related_files:
            return None
        py_files = [p for p in related_files.keys() if p.endswith(".py")]
        if py_files:
            return sorted(py_files)[0]
        return sorted(related_files.keys())[0]

    def _apply_incremental_stub(self, content: str, incremental_goal: str, task_description: str) -> str:
        func_name = self._infer_function_name(incremental_goal, task_description)
        if not func_name:
            return content

        if re.search(rf"def\s+{re.escape(func_name)}\s*\(", content):
            return content

        stub = (
            "\n\n"
            f"def {func_name}(*args, **kwargs):\n"
            f"    \"\"\"Incremental placeholder for: {incremental_goal or task_description}\"\"\"\n"
            "    raise NotImplementedError(\"TODO: implement incremental feature\")\n"
        )
        return content + stub

    @staticmethod
    def _infer_function_name(incremental_goal: str, task_description: str) -> Optional[str]:
        text = f"{incremental_goal} {task_description}".strip()
        if not text:
            return None

        patterns = [
            r"(?:add|implement)\s*(?:function|method)?\s*([A-Za-z_][A-Za-z0-9_]*)",
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:function|method)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _build_unified_diff(path: str, original: str, updated: str) -> str:
        if original == updated:
            return ""
        orig_lines = original.splitlines(keepends=True)
        updated_lines = updated.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines,
            updated_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
        return "\n".join(diff) + "\n"

    @staticmethod
    def _resolve_related_path(error_path: str, related_files: Dict[str, str]) -> Optional[str]:
        normalized = error_path.replace("\\", "/")
        if normalized in related_files:
            return normalized
        for candidate in related_files.keys():
            if normalized.endswith(candidate.replace("\\", "/")):
                return candidate
        return None

    @staticmethod
    def _normalize_diff_path(path: str) -> str:
        p = path.strip()
        if p in {"/dev/null", ""}:
            return p
        if p.startswith("a/") or p.startswith("b/"):
            p = p[2:]
        return p

    @staticmethod
    def _split_unified_patch_sections(patch_text: str) -> List[Tuple[str, str, List[str]]]:
        sections: List[Tuple[str, str, List[str]]] = []
        lines = patch_text.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i]
            if not line.startswith("--- "):
                i += 1
                continue

            old_path = line[4:].strip()
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                continue
            new_path = lines[i][4:].strip()
            i += 1

            hunks: List[str] = []
            while i < len(lines) and not lines[i].startswith("--- "):
                hunks.append(lines[i])
                i += 1

            sections.append((old_path, new_path, hunks))

        return sections

    @staticmethod
    def _is_strict_unified_diff(patch_text: str) -> bool:
        text = str(patch_text or "").strip()
        if not text:
            return False
        if "*** Begin Patch" in text or "*** Update File:" in text or "*** End Patch" in text:
            return False
        lines = text.splitlines()
        has_old = any(line.startswith("--- ") for line in lines)
        has_new = any(line.startswith("+++ ") for line in lines)
        has_hunk = any(line.startswith("@@") for line in lines)
        return has_old and has_new and has_hunk

    def _apply_hunks_to_text(self, original: str, hunk_lines: List[str]) -> str:
        if not hunk_lines:
            return original

        original_lines = original.splitlines()
        result_lines: List[str] = []
        src_index = 0
        i = 0

        while i < len(hunk_lines):
            line = hunk_lines[i]
            if not line.startswith("@@"):
                i += 1
                continue

            match = re.match(r"@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@", line)
            if not match:
                i += 1
                continue

            old_start = int(match.group(1))
            # Copy untouched source before this hunk.
            target_src_index = max(old_start - 1, 0)
            if target_src_index > src_index:
                result_lines.extend(original_lines[src_index:target_src_index])
                src_index = target_src_index

            i += 1
            while i < len(hunk_lines) and not hunk_lines[i].startswith("@@"):
                hline = hunk_lines[i]
                if not hline:
                    prefix = " "
                    payload = ""
                else:
                    prefix = hline[0]
                    payload = hline[1:] if len(hline) > 1 else ""

                if prefix == " ":
                    # Context line: advance in source if possible.
                    if src_index < len(original_lines):
                        src_index += 1
                    result_lines.append(payload)
                elif prefix == "-":
                    # Removal: skip one source line.
                    if src_index < len(original_lines):
                        src_index += 1
                elif prefix == "+":
                    result_lines.append(payload)
                elif prefix == "\\":
                    # "No newline at end of file" marker.
                    pass

                i += 1

        if src_index < len(original_lines):
            result_lines.extend(original_lines[src_index:])

        # Preserve trailing newline when any resulting lines exist.
        return "\n".join(result_lines) + ("\n" if result_lines else "")


__all__ = ["PatchAgent"]
