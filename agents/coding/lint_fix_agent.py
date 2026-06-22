"""Lint-and-fix agent for post-generation Python code cleanup."""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional

from agents.infra.llm_client import LLMClient

from .fix_agent import FixAgent
from .patch_agent import PatchAgent


class LintFixAgent:
    """Run lint checks and apply static/LLM-based repairs for generated Python files."""

    def __init__(self, api_config: Optional[Dict[str, Any]] = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.output_dir = output_dir
        self.fix_agent = FixAgent(max_rounds=int(self.api_config.get("lint_static_rounds", 2)))
        self.max_llm_rounds = max(5, int(self.api_config.get("lint_llm_rounds", 5)))
        configured_max_files = self.api_config.get("lint_llm_max_files")
        self.max_llm_files: Optional[int]
        if configured_max_files is None:
            # No default cap: allow all issue files into LLM queue.
            self.max_llm_files = None
        else:
            parsed_max_files = int(configured_max_files)
            self.max_llm_files = parsed_max_files if parsed_max_files > 0 else None
        self.llm_budget_ratio = max(0.0, float(self.api_config.get("lint_llm_budget_ratio", 1.0)))
        self.max_llm_chars = max(2000, int(self.api_config.get("lint_llm_max_chars", 120000)))
        self.enable_llm_fix = bool(self.api_config.get("enable_lint_llm_fix", True))
        self.patch_agent = PatchAgent(api_config={}, output_dir=output_dir)

        self.ruff_bin = shutil.which("ruff")
        self.llm_client = None
        if self.enable_llm_fix and self.api_config.get("api_key"):
            self.llm_client = LLMClient(self.api_config, output_dir, agent_name="lint_fix_agent")
            logging.info(
                "LintFixAgent LLM enabled: base_url=%s model=%s timeout=%s",
                self.api_config.get("base_url", ""),
                self.api_config.get("model", ""),
                self.api_config.get("lint_fix_timeout_seconds", 600),
            )

    def run_after_codegen(
        self,
        generated_root: str | Path,
        generated_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run lint checks and repairs after code generation stage."""
        root = Path(generated_root).resolve()
        targets = self._collect_python_files(root, generated_entries or [])
        report = self.run_on_files(targets, generated_root=root)
        try:
            from .static_preflight import run_static_preflight

            preflight = run_static_preflight(root, targets)
            report["static_preflight"] = preflight
            static_n = int(preflight.get("issue_count") or 0)
            compile_n = int(preflight.get("compile_failure_count") or 0)
            if static_n or compile_n:
                logging.warning(
                    "Static preflight: %d import/ruff note(s), %d compile error(s) "
                    "(see lint_fix_report.json static_preflight)",
                    static_n,
                    compile_n,
                )
            else:
                logging.info(
                    "Static preflight: clean for %s files (ruff_available=%s)",
                    preflight.get("files_checked"),
                    preflight.get("ruff_available"),
                )
        except Exception as exc:
            logging.warning("Static preflight failed and was skipped: %s", exc)
            report["static_preflight"] = {"error": str(exc)}
        return report

    def run_on_files(
        self,
        file_paths: List[str | Path],
        generated_root: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Run lint/fix on explicit file list."""
        normalized: List[Path] = []
        for item in file_paths:
            path = Path(item).resolve()
            if path.suffix == ".py" and path.exists():
                normalized.append(path)
        return self._run_for_targets(
            sorted({p for p in normalized}, key=lambda p: str(p)),
            generated_root=Path(generated_root).resolve() if generated_root else None,
        )

    def _run_for_targets(
        self,
        targets: List[Path],
        generated_root: Path | None = None,
    ) -> Dict[str, Any]:
        logging.info(
            "LintFixAgent start: targets=%d generated_root=%s ruff_available=%s llm_enabled=%s",
            len(targets),
            generated_root or "",
            bool(self.ruff_bin),
            bool(self.llm_client),
        )
        report: Dict[str, Any] = {
            "generated_root": str(generated_root) if generated_root else "",
            "ruff_available": bool(self.ruff_bin),
            "llm_enabled": bool(self.llm_client),
            "checked_files": len(targets),
            "files_with_issues": 0,
            "fixed_by_static": 0,
            "fixed_by_llm": 0,
            "unresolved": 0,
            "issue_categories": {},
            "final_issue_categories": {},
            "files": [],
        }

        states: List[Dict[str, Any]] = []
        for file_path in targets:
            logging.debug("Checking file: %s", file_path)
            initial = self._check_file(file_path)
            simple_case = self._is_simple_static_case(initial["diagnostics"])
            file_report: Dict[str, Any] = {
                "file": str(file_path),
                "initial_ok": initial["ok"],
                "initial_diagnostics": initial["diagnostics"],
                "initial_categories": self._categorize_diagnostics(initial["diagnostics"]),
                "simple_static_case": simple_case,
                "fixed_by": None,
                "final_ok": initial["ok"],
                "final_diagnostics": initial["diagnostics"],
                "final_categories": self._categorize_diagnostics(initial["diagnostics"]),
            }
            state: Dict[str, Any] = {
                "file_path": file_path,
                "current": initial,
                "file_report": file_report,
            }

            if initial["ok"]:
                logging.debug("File clean without fixes: %s", file_path)
                states.append(state)
                continue

            report["files_with_issues"] += 1
            logging.info(
                "Issues detected: file=%s simple_static_case=%s diagnostics_preview=%s",
                file_path,
                simple_case,
                (initial["diagnostics"] or "").splitlines()[0][:200] if initial["diagnostics"] else "",
            )

            if simple_case:
                logging.info("Trying static fix: %s", file_path)
                static_changed = self._apply_static_fix(file_path)
                after_static = self._check_file(file_path)
                state["current"] = after_static
                logging.info(
                    "Static fix result: file=%s changed=%s ok_after_static=%s",
                    file_path,
                    static_changed,
                    after_static["ok"],
                )
                if static_changed and after_static["ok"]:
                    file_report["fixed_by"] = "static"
                    file_report["final_ok"] = True
                    file_report["final_diagnostics"] = ""
                    report["fixed_by_static"] += 1
            states.append(state)

        llm_budget = self._compute_llm_budget(report["files_with_issues"])
        llm_files_used = 0
        llm_candidates = [
            state
            for state in states
            if not state["current"]["ok"] and state["file_report"]["fixed_by"] is None
        ]
        llm_candidates.sort(
            key=lambda state: (
                -self._diagnostic_severity(state["current"]["diagnostics"]),
                str(state["file_path"]),
            )
        )

        if llm_candidates:
            logging.info(
                "LLM budget planning: issue_files=%d, llm_budget=%d, configured_max=%s, ratio=%s",
                len(llm_candidates),
                llm_budget,
                self.max_llm_files if self.max_llm_files is not None else "unlimited",
                self.llm_budget_ratio,
            )

        for state in llm_candidates:
            file_path: Path = state["file_path"]
            file_report: Dict[str, Any] = state["file_report"]
            current = state["current"]

            if self.llm_client is None:
                logging.info("Skip LLM fix (client disabled): %s", file_path)
                continue
            if llm_files_used >= llm_budget:
                logging.info(
                    "Skip LLM fix (llm file budget exhausted %d/%d): %s",
                    llm_files_used,
                    llm_budget,
                    file_path,
                )
                continue

            llm_files_used += 1
            severity = self._diagnostic_severity(current["diagnostics"])
            logging.info(
                "Trying LLM fix: file=%s llm_file_index=%d/%d severity=%d rounds=%d",
                file_path,
                llm_files_used,
                llm_budget,
                severity,
                self.max_llm_rounds,
            )
            for round_idx in range(self.max_llm_rounds):
                if current["ok"]:
                    break
                attempt_no = round_idx + 1
                logging.info(
                    "LLM round %d/%d for %s",
                    attempt_no,
                    self.max_llm_rounds,
                    file_path,
                )
                changed = self._apply_llm_fix(
                    file_path,
                    current["diagnostics"],
                    attempt_no=attempt_no,
                )
                if not changed:
                    logging.info("LLM produced no change: %s", file_path)
                    continue
                current = self._check_file(file_path)
                state["current"] = current

            if current["ok"]:
                file_report["fixed_by"] = "llm"
                report["fixed_by_llm"] += 1
                logging.info("LLM fixed successfully: %s", file_path)

        for state in states:
            file_path = state["file_path"]
            file_report = state["file_report"]
            final_state = self._check_file(file_path)
            file_report["final_ok"] = final_state["ok"]
            file_report["final_diagnostics"] = final_state["diagnostics"]
            file_report["final_categories"] = self._categorize_diagnostics(final_state["diagnostics"])
            if not final_state["ok"] and file_report["fixed_by"] is None:
                report["unresolved"] += 1
                logging.info("Unresolved after fixes: %s", file_path)
            else:
                logging.debug(
                    "Final status ok=%s fixed_by=%s file=%s",
                    final_state["ok"],
                    file_report["fixed_by"],
                    file_path,
                )
            report["files"].append(file_report)

        initial_counter: Counter[str] = Counter()
        final_counter: Counter[str] = Counter()
        for file_report in report["files"]:
            for category in file_report.get("initial_categories", []):
                initial_counter[str(category)] += 1
            for category in file_report.get("final_categories", []):
                final_counter[str(category)] += 1
        report["issue_categories"] = dict(sorted(initial_counter.items()))
        report["final_issue_categories"] = dict(sorted(final_counter.items()))

        logging.info(
            "LintFixAgent done: checked=%s issues=%s static_fixed=%s llm_fixed=%s unresolved=%s categories=%s",
            report["checked_files"],
            report["files_with_issues"],
            report["fixed_by_static"],
            report["fixed_by_llm"],
            report["unresolved"],
            report["final_issue_categories"],
        )
        return report

    def _collect_python_files(
        self,
        generated_root: Path,
        generated_entries: List[Dict[str, Any]],
    ) -> List[Path]:
        candidates: Dict[str, Path] = {}

        for entry in generated_entries:
            if not isinstance(entry, dict):
                continue
            files = entry.get("files", {})
            if not isinstance(files, dict):
                continue
            for file_path in files.values():
                path = Path(str(file_path))
                if not path.is_absolute():
                    path = (generated_root / path).resolve()
                if path.suffix == ".py" and path.exists():
                    candidates[str(path)] = path

        if generated_root.exists():
            for path in generated_root.rglob("*.py"):
                if path.is_file():
                    candidates[str(path.resolve())] = path.resolve()

        return sorted(candidates.values(), key=lambda p: str(p))

    def _check_file(self, file_path: Path) -> Dict[str, Any]:
        diagnostics: List[str] = []
        content = file_path.read_text(encoding="utf-8")

        try:
            compile(content, str(file_path), "exec")
        except SyntaxError as exc:
            diagnostics.append(
                f"SyntaxError: {exc.msg} (line {exc.lineno}, col {exc.offset})"
            )

        if self.ruff_bin:
            proc = subprocess.run(
                [self.ruff_bin, "check", str(file_path)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
                diagnostics.append(output.strip())

        merged = "\n\n".join(d for d in diagnostics if d).strip()
        return {"ok": not bool(merged), "diagnostics": merged}

    def _apply_static_fix(self, file_path: Path) -> bool:
        changed = False
        updated = file_path.read_text(encoding="utf-8")

        fix_result = self.fix_agent.fix_python_content(updated, max_rounds=2)
        fixed_content = str(fix_result.get("fixed_content", updated))
        if bool(fix_result.get("fixed")) and fixed_content != updated:
            updated = fixed_content
            changed = True

        cleaned = self._basic_clean(updated)
        if cleaned != updated:
            updated = cleaned
            changed = True

        if changed:
            file_path.write_text(updated, encoding="utf-8")

        if self.ruff_bin:
            before_ruff = file_path.read_text(encoding="utf-8")
            subprocess.run(
                [self.ruff_bin, "check", "--fix", str(file_path)],
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [self.ruff_bin, "format", str(file_path)],
                capture_output=True,
                text=True,
            )
            after_ruff = file_path.read_text(encoding="utf-8")
            if after_ruff != before_ruff:
                changed = True

        return changed

    @staticmethod
    def _categorize_diagnostics(diagnostics: str) -> List[str]:
        text = (diagnostics or "").lower()
        if not text.strip():
            return []
        categories: List[str] = []
        if "syntaxerror" in text or "invalid syntax" in text:
            categories.append("syntax")
        if "indentationerror" in text or "unexpected indent" in text or "expected an indented block" in text:
            categories.append("indentation")
        if "unterminated string literal" in text or "unterminated triple-quoted string literal" in text:
            categories.append("string")
        if re.search(r"\bf401\b", text):
            categories.append("ruff_unused_import")
        if re.search(r"\bf821\b", text):
            categories.append("ruff_undefined_name")
        if re.search(r"\bf402\b", text):
            categories.append("ruff_shadow_import")
        if re.search(r"\be9\d*\b", text):
            categories.append("ruff_parse")
        if "import_path" in text or "relative import target" in text or ("module " in text and "not found under generated root" in text):
            categories.append("import_path")
        if "missing_symbol" in text or "cannot resolve" in text:
            categories.append("missing_symbol")
        if re.search(r"\bw29[13]\b", text):
            categories.append("whitespace")
        if not categories:
            categories.append("other")
        # preserve order and dedupe
        out: List[str] = []
        seen = set()
        for item in categories:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @staticmethod
    def _basic_clean(content: str) -> str:
        lines = [line.rstrip() for line in content.splitlines()]
        normalized = "\n".join(lines)
        if content.endswith("\n"):
            normalized += "\n"
        elif normalized:
            normalized += "\n"
        return normalized

    @staticmethod
    def _is_simple_static_case(diagnostics: str) -> bool:
        """
        Allow static fix only for clearly deterministic cases.

        Everything else should go to LLM fix.
        """
        text = (diagnostics or "").lower()
        if not text:
            return False

        simple_markers = [
            "unterminated string literal",
            "unterminated triple-quoted string literal",
            "eol while scanning string literal",
            "expected ':'",
            "expected an indented block",
            "indentationerror",
            "unexpected indent",
            "unexpected character after line continuation character",
            "invalid syntax",
        ]
        if any(marker in text for marker in simple_markers):
            return True

        # Keep a very conservative subset of auto-fixable ruff rules.
        if " f401 " in f" {text} ":
            return True  # unused import
        if " w291 " in f" {text} " or " w293 " in f" {text} ":
            return True  # trailing whitespace / blank-line whitespace

        return False

    def _compute_llm_budget(self, issue_count: int) -> int:
        if issue_count <= 0 or self.llm_client is None:
            return 0

        if self.max_llm_files is None:
            return issue_count

        proportional = int(math.ceil(issue_count * self.llm_budget_ratio))
        return min(issue_count, max(self.max_llm_files, proportional))

    @staticmethod
    def _diagnostic_severity(diagnostics: str) -> int:
        text = (diagnostics or "").lower()
        if not text:
            return 0

        score = 10
        if "syntaxerror" in text or "indentationerror" in text:
            score += 100
        if "unexpected character after line continuation character" in text:
            score += 12
        if "unterminated string literal" in text:
            score += 11
        if "unexpected indent" in text:
            score += 10
        if "invalid syntax" in text:
            score += 8

        if " e9" in f" {text} ":
            score += 60
        if " f821 " in f" {text} ":
            score += 40

        return score

    def _apply_llm_fix(self, file_path: Path, diagnostics: str, attempt_no: int) -> bool:
        if not self.llm_client:
            return False

        content = file_path.read_text(encoding="utf-8")
        if len(content) > self.max_llm_chars:
            logging.warning(
                "Skip LLM lint-fix for %s because file is too large (%d chars)",
                file_path,
                len(content),
            )
            return False

        # First two rounds request minimal patch. Starting from round 3, fall back to full-file output.
        if attempt_no >= 3:
            return self._apply_llm_fullfile_fix(file_path=file_path, diagnostics=diagnostics, content=content, attempt_no=attempt_no)
        return self._apply_llm_patch_fix(file_path=file_path, diagnostics=diagnostics, content=content, attempt_no=attempt_no)

    def _apply_llm_patch_fix(self, file_path: Path, diagnostics: str, content: str, attempt_no: int) -> bool:
        assert self.llm_client is not None
        virtual_path = "target.py"
        prompt = f"""You are fixing Python code to satisfy lint/syntax checks.

Attempt: {attempt_no}
Target file path for patch: {virtual_path}
Lint diagnostics:
{diagnostics or "(no diagnostics provided)"}

Return ONLY a unified diff patch for this single file.
Hard requirements:
1. The patch must only touch {virtual_path}.
2. Use headers exactly:
   --- a/{virtual_path}
   +++ b/{virtual_path}
3. Include valid @@ hunk lines.
4. No markdown fences, no extra text.

Current code:
{content}
"""
        try:
            response = self.llm_client.call(
                [
                    {
                        "role": "system",
                        "content": "You are a strict Python lint-fix assistant that returns valid unified diffs.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=32768,
                max_retry_times=5,
                timeout_seconds=float(self.api_config.get("lint_fix_timeout_seconds", 600)),
            )
        except Exception as exc:
            logging.warning(
                "LLM lint patch failed for %s: %s (base_url=%s, model=%s, timeout=%s)",
                file_path,
                exc,
                self.api_config.get("base_url", ""),
                self.api_config.get("model", ""),
                self.api_config.get("lint_fix_timeout_seconds", 600),
            )
            return False

        patch_text = self._strip_code_fence(str(response)).strip()
        if not patch_text or "--- " not in patch_text or "+++ " not in patch_text:
            return False

        try:
            apply_result = self.patch_agent.apply_patch_text(
                patch_text=patch_text,
                related_files={virtual_path: content},
            )
        except Exception as exc:
            logging.warning("Failed to apply LLM patch for %s: %s", file_path, exc)
            return False

        fixed = str(apply_result.get("updated_files", {}).get(virtual_path, ""))
        if not fixed:
            return False
        if fixed == content:
            return False
        if not fixed.endswith("\n"):
            fixed += "\n"
        file_path.write_text(fixed, encoding="utf-8")
        return True

    def _apply_llm_fullfile_fix(self, file_path: Path, diagnostics: str, content: str, attempt_no: int) -> bool:
        assert self.llm_client is not None
        prompt = f"""You are fixing Python code to satisfy lint/syntax checks.

Attempt: {attempt_no} (full-file fallback mode)

File path: {file_path}
Lint diagnostics:
{diagnostics or "(no diagnostics provided)"}

Requirements:
1. Return only corrected Python code.
2. Keep behavior the same unless required to fix errors.
3. Do not add markdown fences.

Current code:
{content}
"""
        try:
            response = self.llm_client.call(
                [
                    {
                        "role": "system",
                        "content": "You are a strict Python lint-fix assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=32768,
                max_retry_times=5,
                timeout_seconds=float(self.api_config.get("lint_fix_timeout_seconds", 600)),
            )
        except Exception as exc:
            logging.warning(
                "LLM lint-fix failed for %s: %s (base_url=%s, model=%s, timeout=%s)",
                file_path,
                exc,
                self.api_config.get("base_url", ""),
                self.api_config.get("model", ""),
                self.api_config.get("lint_fix_timeout_seconds", 600),
            )
            return False

        fixed = self._strip_code_fence(response).strip()
        if not fixed or fixed == content.strip():
            return False
        if not fixed.endswith("\n"):
            fixed += "\n"
        file_path.write_text(fixed, encoding="utf-8")
        return True

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) < 3:
            return stripped
        if lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1])
        return stripped


__all__ = ["LintFixAgent"]
