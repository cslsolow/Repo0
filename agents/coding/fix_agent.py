"""Fix agent for simple Python syntax repairs."""

from __future__ import annotations

import difflib
import keyword
import re
from typing import Any, Dict, List, Optional, Tuple


class FixAgent:
    """Heuristic fixer for common, simple Python syntax errors."""

    def __init__(self, max_rounds: int = 4, local_window_radius: int = 100) -> None:
        self.max_rounds = max(1, int(max_rounds))
        self.local_window_radius = max(20, int(local_window_radius))

    def fix_files(
        self,
        related_files: Dict[str, str],
        compile_error: str = "",
        preferred_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fix syntax errors in candidate Python files and return patch-like output."""
        updated_files = dict(related_files)
        touched_files: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}

        targets = self._choose_target_paths(
            related_files=updated_files,
            compile_error=compile_error,
            preferred_path=preferred_path,
        )

        for rel_path in targets:
            original = updated_files.get(rel_path, "")
            result = self.fix_python_content(original)
            if result["fixed"]:
                updated_files[rel_path] = str(result["fixed_content"])
                touched_files.append(rel_path)
            details[rel_path] = {
                "fixed": bool(result["fixed"]),
                "error_before": result["error_before"],
                "error_after": result["error_after"],
                "rounds": result["rounds"],
            }

        patch_text = ""
        for path in sorted(set(touched_files)):
            patch_text += self._build_unified_diff(
                path=path,
                original=related_files.get(path, ""),
                updated=updated_files.get(path, ""),
            )

        return {
            "updated_files": updated_files,
            "touched_files": sorted(set(touched_files)),
            "patch": patch_text,
            "summary": "heuristic syntax fixes applied" if touched_files else "no syntax fix applied",
            "details": details,
        }

    def fix_python_content(self, content: str, max_rounds: Optional[int] = None) -> Dict[str, Any]:
        """Attempt to repair syntax errors in a single Python source string."""
        rounds = max(1, int(max_rounds or self.max_rounds))
        current = self._apply_proactive_cleanup(content)
        error_before = self._get_syntax_error(current)
        if error_before is None:
            return {
                "fixed": current != content,
                "fixed_content": current,
                "error_before": None,
                "error_after": None,
                "rounds": 0,
            }

        completed_rounds = 0
        for _ in range(rounds):
            error = self._get_syntax_error(current)
            if error is None:
                break
            updated = self._apply_one_fix(current, error)
            if updated == current:
                break
            current = updated
            completed_rounds += 1

        error_after = self._get_syntax_error(current)
        return {
            "fixed": current != content and error_after is None,
            "fixed_content": current,
            "error_before": self._error_to_dict(error_before),
            "error_after": self._error_to_dict(error_after),
            "rounds": completed_rounds,
        }

    def _apply_proactive_cleanup(self, content: str) -> str:
        """Apply cheap structural cleanup before relying on exact error messages."""
        updated = content
        for cleaner in (
            self._strip_markdown_code_fences,
            self._fix_common_signature_typos,
            self._sanitize_declared_identifiers,
            self._fix_placeholder_name_typos,
            self._fix_empty_block_placeholders,
        ):
            candidate = cleaner(updated)
            if candidate != updated:
                updated = candidate
        return updated

    @staticmethod
    def _build_unified_diff(path: str, original: str, updated: str) -> str:
        if original == updated:
            return ""
        original_lines = original.splitlines(keepends=True)
        updated_lines = updated.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
        return "\n".join(diff) + "\n"

    @staticmethod
    def _error_to_dict(error: Optional[SyntaxError]) -> Optional[Dict[str, Any]]:
        if error is None:
            return None
        return {
            "msg": str(getattr(error, "msg", str(error))),
            "line": int(getattr(error, "lineno", 0) or 0),
            "offset": int(getattr(error, "offset", 0) or 0),
        }

    @staticmethod
    def _get_syntax_error(content: str) -> Optional[SyntaxError]:
        try:
            compile(content, "<fix-agent>", "exec")
            return None
        except SyntaxError as exc:
            return exc

    def _apply_one_fix(self, content: str, error: SyntaxError) -> str:
        proactive = self._apply_proactive_cleanup(content)
        if proactive != content and self._get_syntax_error(proactive) is None:
            return proactive

        local_fixed = self._apply_one_fix_local_window(content, error)
        if local_fixed != content:
            return local_fixed

        lines = content.splitlines()
        if not lines:
            return content

        msg = str(getattr(error, "msg", "") or str(error))
        lower_msg = msg.lower()
        idx = self._clamp_line_index(lines, getattr(error, "lineno", None))

        stripped_fences = self._strip_markdown_code_fences(content)
        if stripped_fences != content:
            return stripped_fences

        trial_signature_star = self._fix_bare_star_signature_by_trial(content, error)
        if trial_signature_star != content:
            return trial_signature_star

        trial_bare_star = self._fix_bare_star_argument_by_trial(content, error)
        if trial_bare_star != content:
            return trial_bare_star

        if "unexpected character after line continuation character" in lower_msg:
            if self._fix_line_continuation_artifact(lines, idx):
                return self._join_lines(lines, content)

        if "unexpected indent" in lower_msg:
            if self._normalize_tabs(lines):
                return self._join_lines(lines, content)
            if self._fix_unexpected_indent(lines, idx):
                return self._join_lines(lines, content)
            module_trial = self._fix_module_level_indented_block(content, error)
            if module_trial != content:
                return module_trial
            trial_dedent = self._fix_unexpected_indent_block_by_trial(content, error)
            if trial_dedent != content:
                return trial_dedent

        if "unterminated triple-quoted string literal" in lower_msg or "eof while scanning triple-quoted string literal" in lower_msg:
            if self._fix_unterminated_triple_quoted(lines, idx):
                return self._join_lines(lines, content)

        if "unterminated string literal" in lower_msg or "eol while scanning string literal" in lower_msg:
            if self._fix_unterminated_single_or_double(lines, idx):
                return self._join_lines(lines, content)

        if "expected ':'" in lower_msg:
            if self._fix_missing_colon(lines, idx):
                return self._join_lines(lines, content)

        if "named arguments must follow bare *" in lower_msg or "invalid star expression" in lower_msg:
            trial_signature = self._fix_bare_star_signature_by_trial(content, error)
            if trial_signature != content:
                return trial_signature
            trial = self._fix_bare_star_argument_by_trial(content, error)
            if trial != content:
                return trial

        if "expected '('" in lower_msg:
            if self._fix_missing_signature_parentheses(lines, idx):
                return self._join_lines(lines, content)

        if "expected an indented block" in lower_msg or "indentationerror" in lower_msg:
            if self._fix_missing_indented_block(lines, idx):
                return self._join_lines(lines, content)

        if "invalid syntax" in lower_msg:
            trial_assignment = self._fix_expression_assignment_by_trial(content, error)
            if trial_assignment != content:
                return trial_assignment
            trial_delimiter = self._fix_delimiter_near_error_by_trial(content, error)
            if trial_delimiter != content:
                return trial_delimiter
            if self._fix_missing_signature_parentheses(lines, idx):
                return self._join_lines(lines, content)
            if self._fix_missing_colon(lines, idx):
                return self._join_lines(lines, content)
            if self._fix_unclosed_opening_delimiter(lines, idx, error):
                return self._join_lines(lines, content)
            if self._fix_unmatched_closing_paren(lines, idx, error):
                return self._join_lines(lines, content)
            trial_removed = self._fix_unmatched_closing_paren_by_trial(content, error)
            if trial_removed != content:
                return trial_removed
            if self._fix_unbalanced_return_parentheses(lines, idx):
                return self._join_lines(lines, content)
            if self._fix_line_continuation_artifact(lines, idx):
                return self._join_lines(lines, content)
            if self._fix_unterminated_single_or_double(lines, idx):
                return self._join_lines(lines, content)
            if self._fix_unterminated_triple_quoted(lines, idx):
                return self._join_lines(lines, content)

        if "unmatched ')'" in lower_msg or "unmatched ']'" in lower_msg or "unmatched '}'" in lower_msg:
            if self._fix_unmatched_closing_paren(lines, idx, error):
                return self._join_lines(lines, content)
            trial_removed = self._fix_unmatched_closing_paren_by_trial(content, error)
            if trial_removed != content:
                return trial_removed

        if "was never closed" in lower_msg or "unexpected eof while parsing" in lower_msg:
            if self._fix_unclosed_opening_delimiter(lines, idx, error):
                return self._join_lines(lines, content)
            trial_delimiter = self._fix_delimiter_near_error_by_trial(content, error)
            if trial_delimiter != content:
                return trial_delimiter

        return content

    def _apply_one_fix_local_window(self, content: str, error: SyntaxError) -> str:
        lines = content.splitlines()
        if len(lines) <= (2 * self.local_window_radius + 40):
            return content

        idx = self._clamp_line_index(lines, getattr(error, "lineno", None))
        start = max(0, idx - self.local_window_radius)
        end = min(len(lines), idx + self.local_window_radius + 1)
        window_lines = list(lines[start:end])
        if not window_lines:
            return content

        local_error = self._rebase_error_for_window(error, start, len(window_lines))
        if local_error is None:
            return content

        updated_window = self._apply_one_fix_without_window(
            self._join_lines(window_lines, "\n"),
            local_error,
        )
        if updated_window == self._join_lines(window_lines, "\n"):
            return content

        updated_window_lines = updated_window.splitlines()
        merged_lines = list(lines)
        merged_lines[start:end] = updated_window_lines
        candidate = self._join_lines(merged_lines, content)
        if self._get_syntax_error(candidate) is None:
            return candidate
        return content

    def _apply_one_fix_without_window(self, content: str, error: SyntaxError) -> str:
        proactive = self._apply_proactive_cleanup(content)
        if proactive != content and self._get_syntax_error(proactive) is None:
            return proactive

        lines = content.splitlines()
        if not lines:
            return content

        msg = str(getattr(error, "msg", "") or str(error))
        lower_msg = msg.lower()
        idx = self._clamp_line_index(lines, getattr(error, "lineno", None))

        stripped_fences = self._strip_markdown_code_fences(content)
        if stripped_fences != content:
            return stripped_fences

        trial_signature_star = self._fix_bare_star_signature_by_trial(content, error)
        if trial_signature_star != content:
            return trial_signature_star

        trial_bare_star = self._fix_bare_star_argument_by_trial(content, error)
        if trial_bare_star != content:
            return trial_bare_star

        if "unexpected character after line continuation character" in lower_msg:
            if self._fix_line_continuation_artifact(lines, idx):
                return self._join_lines(lines, content)

        if "unexpected indent" in lower_msg:
            if self._normalize_tabs(lines):
                return self._join_lines(lines, content)
            if self._fix_unexpected_indent(lines, idx):
                return self._join_lines(lines, content)
            module_trial = self._fix_module_level_indented_block(content, error)
            if module_trial != content:
                return module_trial
            trial_dedent = self._fix_unexpected_indent_block_by_trial(content, error)
            if trial_dedent != content:
                return trial_dedent

        if "unterminated triple-quoted string literal" in lower_msg or "eof while scanning triple-quoted string literal" in lower_msg:
            if self._fix_unterminated_triple_quoted(lines, idx):
                return self._join_lines(lines, content)

        if "unterminated string literal" in lower_msg or "eol while scanning string literal" in lower_msg:
            if self._fix_unterminated_single_or_double(lines, idx):
                return self._join_lines(lines, content)

        if "expected ':'" in lower_msg:
            if self._fix_missing_colon(lines, idx):
                return self._join_lines(lines, content)

        if "named arguments must follow bare *" in lower_msg or "invalid star expression" in lower_msg:
            trial_signature = self._fix_bare_star_signature_by_trial(content, error)
            if trial_signature != content:
                return trial_signature
            trial = self._fix_bare_star_argument_by_trial(content, error)
            if trial != content:
                return trial

        if "expected '('" in lower_msg:
            if self._fix_missing_signature_parentheses(lines, idx):
                return self._join_lines(lines, content)

        if "expected an indented block" in lower_msg or "indentationerror" in lower_msg:
            if self._fix_missing_indented_block(lines, idx):
                return self._join_lines(lines, content)

        if "invalid syntax" in lower_msg:
            trial_assignment = self._fix_expression_assignment_by_trial(content, error)
            if trial_assignment != content:
                return trial_assignment
            trial_delimiter = self._fix_delimiter_near_error_by_trial(content, error)
            if trial_delimiter != content:
                return trial_delimiter
            if self._fix_missing_signature_parentheses(lines, idx):
                return self._join_lines(lines, content)
            if self._fix_missing_colon(lines, idx):
                return self._join_lines(lines, content)
            if self._fix_unclosed_opening_delimiter(lines, idx, error):
                return self._join_lines(lines, content)
            if self._fix_unmatched_closing_paren(lines, idx, error):
                return self._join_lines(lines, content)
            trial_removed = self._fix_unmatched_closing_paren_by_trial(content, error)
            if trial_removed != content:
                return trial_removed
            if self._fix_unbalanced_return_parentheses(lines, idx):
                return self._join_lines(lines, content)
            if self._fix_line_continuation_artifact(lines, idx):
                return self._join_lines(lines, content)
            if self._fix_unterminated_single_or_double(lines, idx):
                return self._join_lines(lines, content)
            if self._fix_unterminated_triple_quoted(lines, idx):
                return self._join_lines(lines, content)

        if "unmatched ')'" in lower_msg or "unmatched ']'" in lower_msg or "unmatched '}'" in lower_msg:
            if self._fix_unmatched_closing_paren(lines, idx, error):
                return self._join_lines(lines, content)
            trial_removed = self._fix_unmatched_closing_paren_by_trial(content, error)
            if trial_removed != content:
                return trial_removed

        if "was never closed" in lower_msg or "unexpected eof while parsing" in lower_msg:
            if self._fix_unclosed_opening_delimiter(lines, idx, error):
                return self._join_lines(lines, content)
            trial_delimiter = self._fix_delimiter_near_error_by_trial(content, error)
            if trial_delimiter != content:
                return trial_delimiter

        return content

    def _fix_bare_star_argument_by_trial(self, content: str, error: SyntaxError) -> str:
        lines = content.splitlines()
        if not lines:
            return content
        idx = self._clamp_line_index(lines, getattr(error, "lineno", None))
        patterns = (
            (r"\(\s*\*,\s*", "("),
            (r",\s*\*,\s*", ", "),
        )
        for candidate in [idx, idx - 1, idx + 1]:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            if "(*," not in line and ", *," not in line:
                continue
            for pattern, replacement in patterns:
                updated_line, count = re.subn(pattern, replacement, line, count=1)
                if count <= 0 or updated_line == line:
                    continue
                trial_lines = list(lines)
                trial_lines[candidate] = updated_line
                trial = self._join_lines(trial_lines, content)
                if self._get_syntax_error(trial) is None:
                    return trial
        return content

    def _fix_bare_star_signature_by_trial(self, content: str, error: SyntaxError) -> str:
        lines = content.splitlines()
        if not lines:
            return content
        idx = self._clamp_line_index(lines, getattr(error, "lineno", None))
        patterns = (
            (r"(\b(?:async\s+def|def)\s+\w+\([^#\n]*?),\s*\*,\s*(\*\*\w+)", r"\1, \2"),
            (r"(\b(?:async\s+def|def)\s+\w+\([^#\n]*?)\(\s*\*,\s*(\*\*\w+)", r"\1(\2"),
        )
        for candidate in [idx, idx - 1, idx + 1]:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            if "def " not in line or "**" not in line or "*" not in line:
                continue
            if "*, **" not in line and "(*, **" not in line:
                continue
            for pattern, replacement in patterns:
                updated_line, count = re.subn(pattern, replacement, line, count=1)
                if count <= 0 or updated_line == line:
                    continue
                trial_lines = list(lines)
                trial_lines[candidate] = updated_line
                trial = self._join_lines(trial_lines, content)
                if self._get_syntax_error(trial) is None:
                    return trial
        return content

    def _fix_expression_assignment_by_trial(self, content: str, error: SyntaxError) -> str:
        lines = content.splitlines()
        if not lines:
            return content
        idx = self._clamp_line_index(lines, getattr(error, "lineno", None))
        assign_re = re.compile(r"(?<![!<>=:])=(?!=)")
        for candidate in [idx, idx - 1, idx + 1]:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            for match in assign_re.finditer(line):
                start, end = match.span()
                updated_line = line[:start] + "==" + line[end:]
                if updated_line == line:
                    continue
                trial_lines = list(lines)
                trial_lines[candidate] = updated_line
                trial = self._join_lines(trial_lines, content)
                if self._get_syntax_error(trial) is None:
                    return trial
        return content

    @staticmethod
    def _rebase_error_for_window(
        error: SyntaxError,
        start_line_idx: int,
        window_len: int,
    ) -> Optional[SyntaxError]:
        lineno = int(getattr(error, "lineno", 0) or 0)
        if lineno <= 0:
            return None
        rebased_lineno = lineno - start_line_idx
        if rebased_lineno <= 0 or rebased_lineno > window_len:
            return None
        rebased = SyntaxError(getattr(error, "msg", str(error)))
        rebased.msg = getattr(error, "msg", str(error))
        rebased.lineno = rebased_lineno
        rebased.offset = int(getattr(error, "offset", 0) or 0)
        rebased.text = getattr(error, "text", None)
        return rebased

    @staticmethod
    def _clamp_line_index(lines: List[str], line_no: Optional[int]) -> int:
        if not lines:
            return 0
        if not line_no:
            return 0
        return min(max(int(line_no) - 1, 0), len(lines) - 1)

    @staticmethod
    def _join_lines(lines: List[str], original: str) -> str:
        joined = "\n".join(lines)
        if original.endswith("\n"):
            return joined + "\n"
        return joined

    @staticmethod
    def _strip_markdown_code_fences(content: str) -> str:
        if "```" not in content:
            return content
        lines = content.splitlines()
        kept = [line for line in lines if not line.strip().startswith("```")]
        if len(kept) == len(lines):
            return content
        joined = "\n".join(kept)
        if content.endswith("\n"):
            joined += "\n"
        return joined

    @staticmethod
    def _fix_common_signature_typos(content: str) -> str:
        """Repair common generation-time signature corruption before parsing."""
        updated = content
        patterns = (
            # def foo,(self, ...) -> def foo(self, ...)
            (
                re.compile(r"^(\s*(?:async\s+def|def)\s+[A-Za-z_][A-Za-z0-9_]*)\s*,\s*\(", re.MULTILINE),
                r"\1(",
            ),
            # class Foo,(Base): -> class Foo(Base):
            (
                re.compile(r"^(\s*class\s+[A-Za-z_][A-Za-z0-9_]*)\s*,\s*\(", re.MULTILINE),
                r"\1(",
            ),
            # class Foo,: -> class Foo:
            (
                re.compile(r"^(\s*class\s+[A-Za-z_][A-Za-z0-9_]*)\s*,\s*:", re.MULTILINE),
                r"\1:",
            ),
            # def foo,: -> def foo():
            (
                re.compile(r"^(\s*(?:async\s+def|def)\s+[A-Za-z_][A-Za-z0-9_]*)\s*,\s*:", re.MULTILINE),
                r"\1():",
            ),
            # def foo:(self, ...) -> def foo(self, ...)
            (
                re.compile(r"^(\s*(?:async\s+def|def)\s+[^\s(]+)\s*:\s*\(", re.MULTILINE),
                r"\1(",
            ),
        )
        for pattern, replacement in patterns:
            updated = pattern.sub(replacement, updated, count=0)
        return updated

    @staticmethod
    def _sanitize_declared_identifiers(content: str) -> str:
        """Normalize illegal class/function names into valid Python identifiers."""
        lines = content.splitlines()
        changed = False

        def _sanitize_name(name: str, *, fallback: str) -> str:
            cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
            cleaned = re.sub(r"_+", "_", cleaned).strip("_")
            if not cleaned:
                cleaned = fallback
            if cleaned[0].isdigit():
                cleaned = f"_{cleaned}"
            if keyword.iskeyword(cleaned):
                cleaned = f"{cleaned}_"
            return cleaned

        def _rewrite_decl(line: str, kind: str) -> str:
            if kind == "def":
                match = re.match(r"^(\s*(?:async\s+def|def)\s+)([^(:\s][^(:]*?)(\s*\(.*)$", line)
                if not match:
                    return line
                prefix, raw_name, suffix = match.groups()
                safe = _sanitize_name(raw_name, fallback="generated_function")
                return f"{prefix}{safe}{suffix}"
            match = re.match(r"^(\s*class\s+)([^(:\s][^(:]*?)(\s*(?:\(|:).*)$", line)
            if not match:
                return line
            prefix, raw_name, suffix = match.groups()
            safe = _sanitize_name(raw_name, fallback="GeneratedClass")
            return f"{prefix}{safe}{suffix}"

        for idx, line in enumerate(lines):
            stripped = line.strip()
            updated = line
            if stripped.startswith(("def ", "async def ")):
                updated = _rewrite_decl(line, "def")
            elif stripped.startswith("class "):
                updated = _rewrite_decl(line, "class")
            if updated != line:
                lines[idx] = updated
                changed = True

        if not changed:
            return content
        joined = "\n".join(lines)
        if content.endswith("\n"):
            joined += "\n"
        return joined

    @staticmethod
    def _fix_placeholder_name_typos(content: str) -> str:
        """Normalize placeholder-generated names that accidentally include punctuation."""
        updated = content
        patterns = (
            (
                re.compile(
                    r'raise\s+NotImplementedError\(\s*f?"([A-Za-z_][A-Za-z0-9_]*)\s*,\s+not yet implemented"\s*\)'
                ),
                r'raise NotImplementedError("\1 not yet implemented")',
            ),
            (
                re.compile(
                    r'raise\s+NotImplementedError\(\s*f?"([A-Za-z_][A-Za-z0-9_]*)\s*,\s+not implemented"\s*\)'
                ),
                r'raise NotImplementedError("\1 not implemented")',
            ),
        )
        for pattern, replacement in patterns:
            updated = pattern.sub(replacement, updated)
        return updated

    @staticmethod
    def _fix_empty_block_placeholders(content: str) -> str:
        """Convert bare TODO lines inside blocks into valid comments."""
        lines = content.splitlines()
        changed = False
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("TODO") and not stripped.startswith("#"):
                indent = line[: len(line) - len(line.lstrip())]
                lines[idx] = f"{indent}# {stripped}"
                changed = True
        if not changed:
            return content
        joined = "\n".join(lines)
        if content.endswith("\n"):
            joined += "\n"
        return joined

    @staticmethod
    def _normalize_tabs(lines: List[str]) -> bool:
        changed = False
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i] = line.expandtabs(4)
                changed = True
        return changed

    @staticmethod
    def _is_odd_unescaped_quote_count(line: str, quote_char: str) -> bool:
        if not line or line.lstrip().startswith("#"):
            return False
        pattern = rf"(?<!\\){re.escape(quote_char)}"
        return len(re.findall(pattern, line)) % 2 == 1

    def _fix_unterminated_single_or_double(self, lines: List[str], idx: int) -> bool:
        for candidate in [idx, idx - 1, idx + 1]:
            if candidate < 0 or candidate >= len(lines):
                continue
            for quote_char in ['"', "'"]:
                line = lines[candidate]
                stripped = line.strip()
                if stripped == quote_char:
                    indent = line[: len(line) - len(line.lstrip())]
                    lines[candidate] = f"{indent}{quote_char}{quote_char}"
                    return True
                if line.rstrip().endswith(quote_char):
                    continue
                if self._is_odd_unescaped_quote_count(line, quote_char):
                    lines[candidate] = f"{line}{quote_char}"
                    return True
        return False

    @staticmethod
    def _detect_unclosed_triple_delimiter(lines: List[str]) -> Optional[str]:
        for delimiter in ['"""', "'''"]:
            count = sum(line.count(delimiter) for line in lines)
            if count % 2 == 1:
                return delimiter
        return None

    def _fix_unterminated_triple_quoted(self, lines: List[str], idx: int) -> bool:
        delimiter = self._detect_unclosed_triple_delimiter(lines) or '"""'

        block_close = self._fix_unterminated_triple_block(lines, idx, delimiter)
        if block_close:
            return True

        for candidate in [idx, idx - 1, idx + 1]:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            if line.count(delimiter) % 2 == 1 and not line.rstrip().endswith(delimiter):
                lines[candidate] = f"{line}{delimiter}"
                return True

        for candidate in range(min(idx, len(lines) - 1), -1, -1):
            line = lines[candidate]
            if line.count(delimiter) % 2 == 1 and not line.rstrip().endswith(delimiter):
                lines[candidate] = f"{line}{delimiter}"
                return True

        lines.append(delimiter)
        return True

    def _fix_unterminated_triple_block(self, lines: List[str], idx: int, delimiter: str) -> bool:
        targeted_close = self._fix_docstring_block_before_code(lines, delimiter)
        if targeted_close:
            return True

        opener_idx = self._find_last_unmatched_triple_opener(lines, delimiter)
        if opener_idx < 0:
            return False

        insert_idx = None
        for candidate in range(opener_idx + 1, len(lines)):
            stripped = lines[candidate].strip()
            if not stripped:
                continue
            if stripped.startswith(("def ", "class ", "@")):
                insert_idx = candidate
                break
            if not lines[candidate].startswith((" ", "\t")) and stripped:
                insert_idx = candidate
                break
        if insert_idx is None:
            return False

        indent = lines[opener_idx][: len(lines[opener_idx]) - len(lines[opener_idx].lstrip())]
        lines.insert(insert_idx, indent + delimiter)
        return True

    def _fix_docstring_block_before_code(self, lines: List[str], delimiter: str) -> bool:
        for opener_idx, line in enumerate(lines):
            if not self._looks_like_docstring_opener(line, delimiter):
                continue
            indent = line[: len(line) - len(line.lstrip())]
            for candidate in range(opener_idx + 1, len(lines)):
                stripped = lines[candidate].strip()
                if not stripped:
                    continue
                if delimiter in stripped:
                    break
                if self._looks_like_docstring_code_boundary(lines[candidate], indent):
                    lines.insert(candidate, indent + delimiter)
                    return True
        return False

    @staticmethod
    def _looks_like_docstring_opener(line: str, delimiter: str) -> bool:
        stripped = line.strip()
        if stripped == delimiter:
            return False
        if not stripped.startswith(delimiter):
            return False
        return stripped.count(delimiter) % 2 == 1

    @staticmethod
    def _looks_like_docstring_code_boundary(line: str, opener_indent: str) -> bool:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return False

        code_prefixes = (
            "def ",
            "class ",
            "@",
            "return ",
            "raise ",
            "if ",
            "for ",
            "while ",
            "with ",
            "try:",
            "except ",
            "finally:",
            "pass",
        )
        if stripped.startswith(code_prefixes):
            return True

        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", stripped):
            return True
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped):
            return True
        return False

    @staticmethod
    def _find_last_unmatched_triple_opener(lines: List[str], delimiter: str) -> int:
        unmatched: List[int] = []
        for idx, line in enumerate(lines):
            count = line.count(delimiter)
            if count <= 0:
                continue
            for _ in range(count):
                if unmatched:
                    unmatched.pop()
                else:
                    unmatched.append(idx)
        return unmatched[-1] if unmatched else -1

    @staticmethod
    def _looks_like_missing_colon_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            return False
        return bool(
            re.match(
                r"^(if|for|while|def|class|elif|else|try|except|finally|with)\b",
                stripped,
            )
        )

    def _fix_missing_colon(self, lines: List[str], idx: int) -> bool:
        for candidate in [idx, idx - 1]:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            if self._looks_like_missing_colon_line(line):
                lines[candidate] = line.rstrip() + ":"
                return True
        return False

    @staticmethod
    def _looks_like_missing_signature_parentheses_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped.startswith(("def ", "async def ")):
            return False
        if "(" in stripped:
            return False
        return stripped.endswith(":") or "->" in stripped

    def _fix_missing_signature_parentheses(self, lines: List[str], idx: int) -> bool:
        for candidate in [idx, idx - 1]:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            if not self._looks_like_missing_signature_parentheses_line(line):
                continue
            indent = line[: len(line) - len(line.lstrip())]
            stripped = line.strip()
            if "->" in stripped:
                head, tail = stripped.split("->", 1)
                head = head.rstrip()
                if head.endswith(":"):
                    head = head[:-1].rstrip()
                lines[candidate] = f"{indent}{head}() ->{tail}"
            else:
                lines[candidate] = f"{line[:-1]}():"
            return True
        return False

    @staticmethod
    def _line_indent(line: str) -> str:
        return line[: len(line) - len(line.lstrip(" "))]

    def _fix_missing_indented_block(self, lines: List[str], idx: int) -> bool:
        if idx <= 0:
            return False
        parent_line = lines[idx - 1]
        parent_indent = self._line_indent(parent_line)
        pass_line = f"{parent_indent}    pass"
        if idx < len(lines) and lines[idx].strip() == "pass":
            return False
        lines.insert(idx, pass_line)
        return True

    @staticmethod
    def _matching_open_for(close_char: str) -> str:
        return {")": "(", "]": "[", "}": "{"}.get(close_char, "")

    def _fix_unmatched_closing_paren(
        self,
        lines: List[str],
        idx: int,
        error: SyntaxError,
    ) -> bool:
        if idx < 0 or idx >= len(lines):
            return False

        msg = str(getattr(error, "msg", "") or str(error))
        close_char = ")"
        if "unmatched ']'" in msg.lower():
            close_char = "]"
        elif "unmatched '}'" in msg.lower():
            close_char = "}"

        candidates = [idx, idx - 1, idx + 1]
        offset = int(getattr(error, "offset", 0) or 0)
        for candidate in candidates:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            if not line:
                continue

            search_positions: List[int] = []
            if offset > 0 and candidate == idx:
                pos = min(max(offset - 1, 0), max(len(line) - 1, 0))
                search_positions.append(pos)
            search_positions.extend(range(len(line) - 1, -1, -1))

            seen: set[int] = set()
            for pos in search_positions:
                if pos in seen:
                    continue
                seen.add(pos)
                if pos < 0 or pos >= len(line):
                    continue
                if line[pos] != close_char:
                    continue
                if self._can_remove_unmatched_closer(line, pos, close_char):
                    lines[candidate] = line[:pos] + line[pos + 1:]
                    return True
        return False

    def _fix_unmatched_closing_paren_by_trial(self, content: str, error: SyntaxError) -> str:
        lines = content.splitlines()
        if not lines:
            return content

        idx = self._clamp_line_index(lines, getattr(error, "lineno", None))
        msg = str(getattr(error, "msg", "") or str(error)).lower()
        close_char = ")"
        if "unmatched ']'" in msg:
            close_char = "]"
        elif "unmatched '}'" in msg:
            close_char = "}"

        candidates = [idx, idx - 1, idx + 1]
        offset = int(getattr(error, "offset", 0) or 0)
        for candidate in candidates:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            positions: List[int] = []
            if offset > 0 and candidate == idx:
                pos = min(max(offset - 1, 0), max(len(line) - 1, 0))
                positions.extend([pos, pos - 1, pos + 1])
            positions.extend(range(len(line) - 1, -1, -1))
            seen: set[int] = set()
            for pos in positions:
                if pos in seen or pos < 0 or pos >= len(line):
                    continue
                seen.add(pos)
                if line[pos] != close_char:
                    continue
                trial_lines = list(lines)
                trial_lines[candidate] = line[:pos] + line[pos + 1:]
                trial = self._join_lines(trial_lines, content)
                if self._get_syntax_error(trial) is None:
                    return trial
        return content

    def _fix_unclosed_opening_delimiter(
        self,
        lines: List[str],
        idx: int,
        error: SyntaxError,
    ) -> bool:
        msg = str(getattr(error, "msg", "") or str(error)).lower()
        close_char = None
        if "'(' was never closed" in msg:
            close_char = ")"
        elif "'[' was never closed" in msg:
            close_char = "]"
        elif "'{' was never closed" in msg:
            close_char = "}"
        if close_char is None:
            return False

        for candidate in range(len(lines) - 1, idx - 1, -1):
            line = lines[candidate]
            if not line.strip():
                continue
            stripped = line.rstrip()
            if stripped.endswith((",", "(", "[", "{", "\\", ":")):
                continue
            lines[candidate] = stripped + close_char
            return True
        for candidate in [idx, idx + 1, idx - 1]:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            if not line.strip():
                continue
            stripped = line.rstrip()
            if stripped.endswith((",", "(", "[", "{", "\\", ":")):
                continue
            lines[candidate] = stripped + close_char
            return True
        for candidate in range(len(lines) - 1, idx - 1, -1):
            line = lines[candidate]
            if not line.strip():
                continue
            stripped = line.rstrip()
            if stripped.endswith((",", "(", "[", "{", "\\", ":")):
                continue
            lines[candidate] = stripped + close_char
            return True
        return False

    def _fix_delimiter_near_error_by_trial(self, content: str, error: SyntaxError) -> str:
        lines = content.splitlines()
        if not lines:
            return content

        idx = self._clamp_line_index(lines, getattr(error, "lineno", None))
        offset = int(getattr(error, "offset", 0) or 0)
        candidate_lines = [idx, idx - 1, idx + 1]
        close_chars = [")", "]", "}"]

        for candidate in candidate_lines:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            if not line:
                continue

            insert_positions = []
            if candidate == idx and offset > 0:
                pos = min(max(offset - 1, 0), len(line))
                insert_positions.extend([pos, pos + 1])
            insert_positions.extend([len(line.rstrip()), len(line)])
            seen_insert = set()
            for pos in insert_positions:
                if pos in seen_insert or pos < 0 or pos > len(line):
                    continue
                seen_insert.add(pos)
                for close_char in close_chars:
                    trial_lines = list(lines)
                    trial_lines[candidate] = line[:pos] + close_char + line[pos:]
                    trial = self._join_lines(trial_lines, content)
                    if self._get_syntax_error(trial) is None:
                        return trial

            removal_positions = []
            if candidate == idx and offset > 0:
                pos = min(max(offset - 1, 0), max(len(line) - 1, 0))
                removal_positions.extend([pos, pos - 1, pos + 1])
            removal_positions.extend(range(len(line) - 1, -1, -1))
            seen_remove = set()
            for pos in removal_positions:
                if pos in seen_remove or pos < 0 or pos >= len(line):
                    continue
                seen_remove.add(pos)
                if line[pos] not in close_chars:
                    continue
                trial_lines = list(lines)
                trial_lines[candidate] = line[:pos] + line[pos + 1:]
                trial = self._join_lines(trial_lines, content)
                if self._get_syntax_error(trial) is None:
                    return trial
        return content

    def _can_remove_unmatched_closer(self, line: str, pos: int, close_char: str) -> bool:
        open_char = self._matching_open_for(close_char)
        if not open_char:
            return False
        before = line[:pos]
        after = line[pos + 1:]
        open_count = before.count(open_char)
        close_count = before.count(close_char)
        if close_count >= open_count:
            return True
        if line.count(close_char) > line.count(open_char):
            return True
        stripped_after = after.strip()
        if not stripped_after:
            return True
        if stripped_after and stripped_after[0] in ",:])}":
            return True
        return False

    def _fix_unexpected_indent(self, lines: List[str], idx: int) -> bool:
        # Common generation artifact: first line is indented at module level.
        if idx == 0 and lines[0].strip() and lines[0][:1].isspace():
            lines[0] = lines[0].lstrip()
            return True

        for candidate in [idx, idx - 1]:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            if not line.strip() or not line[:1].isspace():
                continue
            if candidate == 0:
                lines[candidate] = line.lstrip()
                return True
            prev = lines[candidate - 1].rstrip()
            if prev and not prev.endswith(":"):
                lines[candidate] = line.lstrip()
                return True
        return False

    def _fix_module_level_indented_block(self, content: str, error: SyntaxError) -> str:
        lines = content.splitlines()
        if not lines:
            return content

        idx = self._clamp_line_index(lines, getattr(error, "lineno", None))
        if idx < 0 or idx >= len(lines):
            return content

        start = idx
        while start > 0 and (lines[start - 1].startswith((" ", "\t")) or not lines[start - 1].strip()):
            start -= 1
        while start < len(lines) and not lines[start].startswith((" ", "\t")):
            start += 1
        if start >= len(lines) or not lines[start].startswith((" ", "\t")):
            return content

        end = start
        while end + 1 < len(lines):
            nxt = lines[end + 1]
            if not nxt.strip() or nxt.startswith((" ", "\t")):
                end += 1
                continue
            break

        block = lines[start : end + 1]
        indents = [
            len(line) - len(line.lstrip(" "))
            for line in block
            if line.strip() and line.startswith(" ")
        ]
        if not indents:
            return content

        min_indent = min(indents)
        if min_indent <= 0:
            return content

        trial_lines = list(lines)
        for i in range(start, end + 1):
            line = trial_lines[i]
            if line.strip():
                trial_lines[i] = line[min_indent:] if line.startswith(" " * min_indent) else line.lstrip()
        self._normalize_future_import_position(trial_lines)
        trial = self._join_lines(trial_lines, content)
        if self._get_syntax_error(trial) is None:
            return trial
        return content

    @staticmethod
    def _normalize_future_import_position(lines: List[str]) -> None:
        future_lines = [line.strip() for line in lines if line.strip().startswith("from __future__ import ")]
        if not future_lines:
            return

        first_code_idx = 0
        if lines and lines[0].strip().startswith(('"""', "'''")):
            delimiter = '"""' if lines[0].strip().startswith('"""') else "'''"
            stripped0 = lines[0].strip()
            if stripped0.count(delimiter) >= 2:
                first_code_idx = 1
            else:
                first_code_idx = 1
                while first_code_idx < len(lines):
                    if delimiter in lines[first_code_idx]:
                        first_code_idx += 1
                        break
                    first_code_idx += 1
        while first_code_idx < len(lines) and not lines[first_code_idx].strip():
            first_code_idx += 1

        filtered = [line for line in lines if not line.strip().startswith("from __future__ import ")]
        insert_idx = first_code_idx
        for future in reversed(future_lines):
            filtered.insert(insert_idx, future)
        lines[:] = filtered

    def _fix_unexpected_indent_block_by_trial(self, content: str, error: SyntaxError) -> str:
        lines = content.splitlines()
        if not lines:
            return content

        idx = self._clamp_line_index(lines, getattr(error, "lineno", None))
        if idx < 0 or idx >= len(lines):
            return content

        start = idx
        while start > 0 and lines[start - 1].strip():
            prev = lines[start - 1]
            if prev.rstrip().endswith(":"):
                break
            if not prev[:1].isspace():
                break
            start -= 1

        end = idx
        while end + 1 < len(lines) and lines[end + 1].strip():
            curr = lines[end]
            if curr.rstrip().endswith(":"):
                break
            end += 1

        candidates = [(idx, idx), (start, end), (start, idx), (idx, end)]
        for lo, hi in candidates:
            if lo < 0 or hi >= len(lines) or lo > hi:
                continue
            block = lines[lo : hi + 1]
            indents = [
                len(line) - len(line.lstrip(" "))
                for line in block
                if line.strip() and line.startswith(" ")
            ]
            if not indents:
                continue
            min_indent = min(indents)
            if min_indent <= 0:
                continue
            trial_lines = list(lines)
            for i in range(lo, hi + 1):
                line = trial_lines[i]
                if line.strip():
                    trial_lines[i] = line[min_indent:] if line.startswith(" " * min_indent) else line.lstrip()
            trial = self._join_lines(trial_lines, content)
            if self._get_syntax_error(trial) is None:
                return trial
        module_trial = self._fix_module_level_indented_block(content, error)
        if module_trial != content:
            return module_trial
        return content

    def _fix_line_continuation_artifact(self, lines: List[str], idx: int) -> bool:
        # Handle trailing "\" artifacts such as "expr\\            " or "expr\\  # ...".
        for candidate in [idx, idx - 1, idx + 1]:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]

            m_end = re.match(r"^(.*?)(?<!\\)\\\s*$", line)
            if m_end:
                lines[candidate] = m_end.group(1).rstrip()
                return True

            m_mid = re.match(r"^(.*?)(?<!\\)\\\s+(\S.*)$", line)
            if m_mid:
                left = m_mid.group(1).rstrip()
                right = m_mid.group(2).strip()
                lines[candidate] = f"{left} {right}".strip()
                return True
        return False

    def _fix_unbalanced_return_parentheses(self, lines: List[str], idx: int) -> bool:
        # Repair a frequent pattern:
        # return (...) /
        #        (... + 1)
        # where the denominator line is missing one or more closing ")".
        for candidate in [idx, idx - 1, idx - 2]:
            if candidate < 0 or candidate >= len(lines):
                continue
            line = lines[candidate]
            if "return" not in line or not line.rstrip().endswith("/"):
                continue

            next_line = candidate + 1
            while next_line < len(lines) and not lines[next_line].strip():
                next_line += 1
            if next_line >= len(lines):
                continue

            snippet = f"{line}\n{lines[next_line]}"
            missing = snippet.count("(") - snippet.count(")")
            if missing > 0:
                lines[next_line] = lines[next_line].rstrip() + (")" * missing)
                return True
        return False

    def _choose_target_paths(
        self,
        related_files: Dict[str, str],
        compile_error: str,
        preferred_path: Optional[str],
    ) -> List[str]:
        if preferred_path and preferred_path in related_files:
            return [preferred_path]

        extracted_path, _ = self._extract_file_and_line_from_error(compile_error)
        if extracted_path:
            resolved = self._resolve_related_path(extracted_path, related_files)
            if resolved:
                return [resolved]

        return sorted(path for path in related_files if path.endswith(".py"))

    @staticmethod
    def _extract_file_and_line_from_error(compile_error: str) -> Tuple[Optional[str], Optional[int]]:
        if not compile_error.strip():
            return None, None
        m = re.search(r'File\s+"([^"]+)"\s*,\s*line\s+(\d+)', compile_error)
        if m:
            return m.group(1).replace("\\", "/"), int(m.group(2))
        return None, None

    @staticmethod
    def _resolve_related_path(error_path: str, related_files: Dict[str, str]) -> Optional[str]:
        normalized = error_path.replace("\\", "/")
        if normalized in related_files:
            return normalized
        for candidate in related_files.keys():
            if normalized.endswith(candidate.replace("\\", "/")):
                return candidate
        return None


__all__ = ["FixAgent"]
