"""Utilities for calling LLM APIs from agents."""

from __future__ import annotations

import json
import os
import logging
import re
import time
import tempfile
import codecs
from datetime import datetime
from typing import Any, Dict, List

import fcntl

try:
    from openai import OpenAI
    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        _RETRYABLE_OPENAI_EXCEPTIONS: tuple[type[BaseException], ...] = (
            APIConnectionError,
            RateLimitError,
            InternalServerError,
            APITimeoutError,
        )
    except ImportError:  # pragma: no cover
        _RETRYABLE_OPENAI_EXCEPTIONS = ()
except ImportError:
    raise ImportError("OpenAI package is required. Install it with: pip install openai")


def _fix_missing_json_commas(text: str) -> str:
    """Insert commas between adjacent JSON values and keys when they're missing."""
    candidate_text = text
    candidate_text = re.sub(
        r'("([^"\\]|\\.)*")(\s*\n\s*)("[^"\\]*"\s*:)',
        r'\1,\3\4',
        candidate_text,
    )
    candidate_text = re.sub(
        r'(\b(?:true|false|null|-?\d+(?:\.\d+)?)\b)(\s*\n\s*)("[^"\\]*"\s*:)',
        r'\1,\2\3',
        candidate_text,
    )
    candidate_text = re.sub(r'}(\s*\n\s*){', r'},\1{', candidate_text)
    candidate_text = re.sub(r'](\s*\n\s*)\[', r'],\1[', candidate_text)
    return candidate_text


_VALUE_END_RE = re.compile(
    r'(true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|"(?:[^"\\]|\\.)*")\s*$'
)

# String keys for LLM JSON payloads where we apply quote/backslash/control-char fixes.
_JSON_LONG_TEXT_FIELD_NAMES: tuple[str, ...] = (
    "code",
    "content",
    "description",
    "rationale",
    "summary",
    "reasoning",
    "documentation",
    "integration_notes",
    "test_code",
    "implementation",
    "explanation",
    "justification",
    "message",
    "text",
    "body",
    "name",
    "reason",
    "output",
    "plan",
    "details",
    "error",
)


def _escape_invalid_backslashes_in_all_json_strings(text: str) -> str:
    """Escape invalid backslashes inside any JSON string, not only known fields."""
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                backslashes = 0
                j = i - 1
                while j >= 0 and text[j] == '\\':
                    backslashes += 1
                    j -= 1
                if backslashes % 2 == 0:
                    in_string = True
            i += 1
            continue

        if ch == '\\':
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt and nxt in {'"', '\\', '/', 'b', 'f', 'n', 'r', 't'}:
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            if nxt == "u" and i + 5 < len(text) and re.match(r"[0-9a-fA-F]{4}", text[i + 2:i + 6]):
                out.append(ch)
                out.append("u")
                out.extend(list(text[i + 2:i + 6]))
                i += 6
                continue
            out.append("\\\\")
            i += 1
            continue

        out.append(ch)
        if ch == '"':
            backslashes = 0
            j = len(out) - 2
            while j >= 0 and out[j] == '\\':
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                in_string = False
        i += 1
    return "".join(out)


def _json_context_stack(text: str, end_pos: int) -> tuple[list[str], bool]:
    """Return a stack of open JSON containers up to end_pos and string state."""
    stack: list[str] = []
    in_string = False
    escape_next = False

    for char in text[:end_pos]:
        if in_string:
            if escape_next:
                escape_next = False
            elif char == '\\':
                escape_next = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == '{':
            stack.append('{')
        elif char == '[':
            stack.append('[')
        elif char == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif char == ']' and stack and stack[-1] == '[':
            stack.pop()

    return stack, in_string


def _ends_with_json_value(text: str) -> bool:
    """Check if text ends with a JSON value token."""
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped[-1] in "]}":
        return True
    return _VALUE_END_RE.search(stripped) is not None


def _fix_missing_object_closers(text: str, error_pos: int) -> str | None:
    """Insert a missing object closer before a new object in arrays."""
    if error_pos < 0 or error_pos >= len(text):
        return None
    if text[error_pos] != "{":
        return None

    stack, in_string = _json_context_stack(text, error_pos)
    if in_string or not stack or stack[-1] != "{":
        return None
    if "[" not in stack:
        return None
    if not _ends_with_json_value(text[:error_pos]):
        return None

    empty_object = re.match(r"\{\s*\},", text[error_pos:])
    if empty_object:
        return text[:error_pos] + "}," + text[error_pos + empty_object.end():]

    return text[:error_pos] + "}," + text[error_pos:]


def _strip_bom_and_format_chars(text: str) -> str:
    """Remove UTF-8 BOM and common invisible Unicode characters LLMs sometimes emit."""
    stripped = text.lstrip("\ufeff")
    return stripped.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")


def _slice_first_complete_json(text: str) -> str | None:
    """Return the first top-level JSON object or array slice using string-aware bracket matching."""
    s = text.strip()
    start = -1
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start < 0:
        return None

    stack: list[str] = []
    in_string = False
    escape_next = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack or ch != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return s[start : i + 1]

    return None


def _strip_trailing_commas_outside_strings(text: str) -> str:
    """Remove JSON-invalid trailing commas before } or ] (outside quoted strings)."""
    chars = list(text)
    in_string = False
    escape_next = False
    i = 0
    while i < len(chars):
        ch = chars[i]
        if in_string:
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < len(chars) and chars[j] in " \t\n\r":
                j += 1
            if j < len(chars) and chars[j] in "}]":
                del chars[i]
                continue
        i += 1
    return "".join(chars)


def _collapse_duplicate_commas(text: str) -> str:
    """Turn `,,` and `, ,` patterns into a single comma outside strings."""
    chars = list(text)
    in_string = False
    escape_next = False
    i = 0
    while i + 1 < len(chars):
        ch = chars[i]
        if in_string:
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue

        if ch == ",":
            k = i + 1
            while k < len(chars) and chars[k] in " \t\n\r":
                k += 1
            if k < len(chars) and chars[k] == ",":
                del chars[k]
                continue
        i += 1
    return "".join(chars)


def _try_suffix_close_unterminated(text: str) -> str | None:
    """If the payload ends inside an open string or with unclosed containers, append closers."""
    stack, in_string = _json_context_stack(text, len(text))
    if not stack and not in_string:
        return None
    suffix = ""
    if in_string:
        suffix += '"'
    for opener in reversed(stack):
        suffix += "}" if opener == "{" else "]"
    return text.rstrip() + suffix


def _raw_decode_json_prefix(text: str) -> Any | None:
    """Parse the first valid JSON value and ignore trailing junk (`Extra data` cases)."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        decoder = json.JSONDecoder()
        value, end_idx = decoder.raw_decode(stripped)
        if end_idx < len(stripped):
            tail = stripped[end_idx:].strip()
            if tail:
                logging.info(
                    "JSON raw_decode consumed %d/%d chars; ignoring trailing: %r",
                    end_idx,
                    len(stripped),
                    tail[:120],
                )
        return value
    except json.JSONDecodeError:
        return None


def _fix_missing_container_closer(text: str, error_pos: int) -> str | None:
    """Insert a missing container closer when parser meets a mismatched closer.

    Typical malformed shape:
      "items": [
        "a",
        "b"
      }
    where `]` is missing before `}`.
    """
    if error_pos < 0 or error_pos >= len(text):
        return None

    current = text[error_pos]
    stack, in_string = _json_context_stack(text, error_pos)
    if in_string or not stack:
        return None

    top = stack[-1]
    if top == "[" and current == "}":
        return text[:error_pos] + "]" + text[error_pos:]
    if top == "{" and current == "]":
        return text[:error_pos] + "}" + text[error_pos:]

    return None


def _escape_unescaped_quotes_in_string_field(text: str, field_name: str) -> str:
    """Escape bare quotes inside a specific JSON string field.

    This targets malformed payloads like:
      {"code": "x = fn("a")"}
    where inner quotes are not escaped and break JSON parsing.

    Strategy:
    - locate `"field_name": "`
    - scan until the real end quote for that string field
    - if a non-escaped `"` is followed by a non-terminating character
      (i.e. not comma/object close after optional whitespace), treat it as
      inner content and escape it.
    """
    key_token = f'"{field_name}"'
    start = 0
    result = text

    while True:
        key_pos = result.find(key_token, start)
        if key_pos == -1:
            break

        colon_pos = result.find(":", key_pos + len(key_token))
        if colon_pos == -1:
            break

        quote_pos = result.find('"', colon_pos + 1)
        if quote_pos == -1:
            break

        i = quote_pos + 1
        escaped = False
        chars = list(result)
        changed = False

        while i < len(chars):
            ch = chars[i]
            if escaped:
                escaped = False
                i += 1
                continue

            if ch == "\\":
                escaped = True
                i += 1
                continue

            if ch == '"':
                j = i + 1
                while j < len(chars) and chars[j].isspace():
                    j += 1

                # Real field terminator should be followed by , or } or ]
                if j < len(chars) and chars[j] in {",", "}", "]"}:
                    break

                # Otherwise this is very likely an unescaped inner quote.
                chars.insert(i, "\\")
                changed = True
                i += 2
                continue

            i += 1

        if changed:
            result = "".join(chars)
            # resume after current key token region
            start = key_pos + len(key_token)
        else:
            start = quote_pos + 1

    return result


def _escape_invalid_backslashes_in_string_field(text: str, field_name: str) -> str:
    """Escape invalid backslash sequences inside a specific JSON string field."""
    key_token = f'"{field_name}"'
    start = 0
    result = text
    valid_simple = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}

    while True:
        key_pos = result.find(key_token, start)
        if key_pos == -1:
            break
        colon_pos = result.find(":", key_pos + len(key_token))
        if colon_pos == -1:
            break
        quote_pos = result.find('"', colon_pos + 1)
        if quote_pos == -1:
            break

        chars = list(result)
        i = quote_pos + 1
        escaped = False
        changed = False

        while i < len(chars):
            ch = chars[i]
            if escaped:
                escaped = False
                i += 1
                continue

            if ch == "\\":
                nxt = chars[i + 1] if i + 1 < len(chars) else ""
                if nxt not in valid_simple:
                    chars.insert(i, "\\")
                    changed = True
                    i += 2
                    continue
                if nxt == "u":
                    hex_part = "".join(chars[i + 2 : i + 6]) if i + 6 <= len(chars) else ""
                    if len(hex_part) != 4 or any(c not in "0123456789abcdefABCDEF" for c in hex_part):
                        chars.insert(i, "\\")
                        changed = True
                        i += 2
                        continue
                escaped = True
                i += 1
                continue

            if ch == '"':
                j = i + 1
                while j < len(chars) and chars[j].isspace():
                    j += 1
                if j < len(chars) and chars[j] in {",", "}", "]"}:
                    break
            i += 1

        if changed:
            result = "".join(chars)
            start = key_pos + len(key_token)
        else:
            start = quote_pos + 1

    return result


def _escape_control_chars_in_string_field(text: str, field_name: str) -> str:
    """Escape raw control chars (e.g., newlines) inside a specific JSON string field."""
    key_token = f'"{field_name}"'
    start = 0
    result = text

    while True:
        key_pos = result.find(key_token, start)
        if key_pos == -1:
            break
        colon_pos = result.find(":", key_pos + len(key_token))
        if colon_pos == -1:
            break
        quote_pos = result.find('"', colon_pos + 1)
        if quote_pos == -1:
            break

        chars = list(result)
        i = quote_pos + 1
        escaped = False
        changed = False

        while i < len(chars):
            ch = chars[i]
            if escaped:
                escaped = False
                i += 1
                continue
            if ch == "\\":
                escaped = True
                i += 1
                continue
            if ch == '"':
                j = i + 1
                while j < len(chars) and chars[j].isspace():
                    j += 1
                if j < len(chars) and chars[j] in {",", "}", "]"}:
                    break
                i += 1
                continue

            code = ord(ch)
            if code < 0x20:
                repl = {
                    "\n": "\\n",
                    "\r": "\\r",
                    "\t": "\\t",
                    "\b": "\\b",
                    "\f": "\\f",
                }.get(ch, f"\\u{code:04x}")
                chars[i : i + 1] = list(repl)
                changed = True
                i += len(repl)
                continue
            i += 1

        if changed:
            result = "".join(chars)
            start = key_pos + len(key_token)
        else:
            start = quote_pos + 1

    return result


def _decode_lenient_json_string_payload(raw: str) -> str:
    """Decode a string payload that came from malformed JSON.

    DeepSeek sometimes over-escapes code strings inside JSON, e.g. it emits
    ``\\\"`` where JSON needs ``\"``. If strict JSON decoding cannot handle the
    fragment, use a conservative escape decoder so Python source still gets
    real newlines and quotes.
    """
    candidates = [
        raw,
        raw.replace('\\\\"', '\\"'),
    ]
    for candidate in candidates:
        json_candidate = candidate
        json_candidate = json_candidate.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        json_candidate = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', json_candidate)
        json_candidate = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_candidate)
        try:
            return json.loads(f'"{json_candidate}"')
        except json.JSONDecodeError:
            continue

    try:
        return codecs.decode(raw, "unicode_escape")
    except Exception:
        return raw.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t").replace('\\"', '"')


def _find_malformed_string_end(text: str, start: int, patterns: tuple[str, ...]) -> int | None:
    """Find the closing quote of a malformed long JSON string by schema boundary."""
    best: int | None = None
    tail = text[start:]
    for pattern in patterns:
        match = re.search(pattern, tail, flags=re.DOTALL)
        if match is None:
            continue
        candidate = start + match.start()
        if best is None or candidate < best:
            best = candidate
    return best


def _trim_truncated_code_payload_tail(text: str) -> str:
    """Trim common non-code tails added after a truncated code payload.

    DeepSeek occasionally emits a long code string, then appends markdown fences or
    explanatory prose such as "Rest of content remains unchanged..." after the useful
    payload. When schema delimiters are missing, we salvage the remainder by dropping
    these common tails before lenient string decoding.
    """
    trimmed = text.strip()

    fence_pos = trimmed.find("\n```")
    if fence_pos != -1:
        trimmed = trimmed[:fence_pos].rstrip()

    # Observed explanatory suffix in malformed patch responses.
    truncation_markers = (
        "\nRest of content remains unchanged until the end, just keeping the provided content truncated.",
        "\n... Rest of content remains unchanged until the end, just keeping the provided content truncated.",
        "\nThe rest of the content remains unchanged",
    )
    for marker in truncation_markers:
        marker_pos = trimmed.find(marker)
        if marker_pos != -1:
            trimmed = trimmed[:marker_pos].rstrip()

    return trimmed


def _repair_updated_files_json_payload(text: str) -> Dict[str, Any] | None:
    """Recover PatchAgent JSON with malformed ``updated_files[*].content`` strings."""
    if '"updated_files"' not in text or '"content"' not in text or '"path"' not in text:
        return None

    files: list[dict[str, str]] = []
    cursor = text.find('"updated_files"')
    while cursor != -1:
        path_match = re.search(r'"path"\s*:\s*"((?:[^"\\]|\\.)*)"', text[cursor:], flags=re.DOTALL)
        if path_match is None:
            break
        path_start = cursor + path_match.start()
        path_end = cursor + path_match.end()
        path = _decode_lenient_json_string_payload(path_match.group(1))

        content_key = re.search(r'"content"\s*:\s*"', text[path_end:], flags=re.DOTALL)
        if content_key is None:
            break
        content_start = path_end + content_key.end()
        content_end = _find_malformed_string_end(
            text,
            content_start,
            (
                r'"\s*}\s*,\s*{\s*"path"\s*:',
                r'"\s*}\s*]\s*,\s*"(?:touched_files|summary)"\s*:',
                r'"\s*}\s*]\s*}',
            ),
        )
        if content_end is None:
            # Fallback: salvage the remainder for truncated single-file payloads.
            # This avoids re-requesting when the only broken part is the closing
            # JSON structure around a very large content string.
            remainder = _trim_truncated_code_payload_tail(text[content_start:])
            if not remainder:
                break
            content = _decode_lenient_json_string_payload(remainder)
            files.append({"path": path, "content": content})
            cursor = len(text)
            break

        content = _decode_lenient_json_string_payload(text[content_start:content_end])
        files.append({"path": path, "content": content})
        cursor = content_end + 1

    if not files:
        return None

    summary = ""
    summary_match = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text[cursor:], flags=re.DOTALL)
    if summary_match:
        summary = _decode_lenient_json_string_payload(summary_match.group(1))

    touched_files = [item["path"] for item in files]
    return {
        "updated_files": files,
        "touched_files": touched_files,
        "summary": summary,
    }


def _repair_single_code_json_payload(text: str) -> Dict[str, Any] | None:
    """Recover simple codegen JSON schemas with one long code string field."""
    if not text.lstrip().startswith("{"):
        return None

    for code_field, path_field in (("code", "file_path"), ("test_code", "test_file_path")):
        key_match = re.search(rf'"{re.escape(code_field)}"\s*:\s*"', text, flags=re.DOTALL)
        if key_match is None:
            continue

        code_start = key_match.end()
        code_end = _find_malformed_string_end(
            text,
            code_start,
            (
                r'"\s*}\s*$',
                r'"\s*,\s*"(?:summary|rationale|metadata|notes)"\s*:',
            ),
        )
        if code_end is None:
            continue

        result: Dict[str, Any] = {
            code_field: _decode_lenient_json_string_payload(text[code_start:code_end])
        }
        path_match = re.search(rf'"{re.escape(path_field)}"\s*:\s*"((?:[^"\\]|\\.)*)"', text[: key_match.start()], flags=re.DOTALL)
        if path_match:
            result[path_field] = _decode_lenient_json_string_payload(path_match.group(1))
        return result

    return None


def _repair_known_code_json_payload(text: str) -> Dict[str, Any] | None:
    """Recover known code-output JSON schemas when long source strings break JSON."""
    return _repair_updated_files_json_payload(text) or _repair_single_code_json_payload(text)


class TokenTracker:
    """Tracks token usage across all LLM calls."""

    def __init__(self, output_dir: str = ".", agent_name: str = "unknown"):
        self.output_dir = output_dir
        self.agent_name = agent_name
        self.agent_log_file = os.path.join(output_dir, f"token_usage_{agent_name}.json")
        self.global_log_file = os.path.join(output_dir, "token_usage.json")
        self.agent_lock_file = os.path.join(output_dir, f"token_usage_{agent_name}.lock")
        self.global_lock_file = os.path.join(output_dir, "token_usage.lock")
        self.usage_records = []
        self.global_summary = {}
        self._load_existing_records()
        self._load_global_summary()

    @staticmethod
    def _load_json_file(path: str, default: Any) -> Any:
        """Load JSON content from file, returning default on parse or existence errors."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return default

    @staticmethod
    def _atomic_write_json(path: str, data: Any) -> None:
        """Write JSON atomically to avoid partial/truncated files under concurrent writers."""
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_token_usage_", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
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

    def _with_lock(self, lock_path: str, callback):
        """Run callback under an exclusive filesystem lock."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(lock_path, 'a+', encoding='utf-8') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                return callback()
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _summarize_agent_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collapse raw per-call records into the global per-agent summary shape."""
        summary = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_cached_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost": 0.0,
            "model": "",
        }
        for record in records:
            summary["total_calls"] += 1
            summary["total_tokens"] += int(record.get("total_tokens", 0) or 0)
            summary["total_prompt_tokens"] += int(record.get("prompt_tokens", 0) or 0)
            summary["total_cached_prompt_tokens"] += int(record.get("cached_prompt_tokens", 0) or 0)
            summary["total_completion_tokens"] += int(record.get("completion_tokens", 0) or 0)
            summary["total_cost"] += float(record.get("estimated_cost", 0.0) or 0.0)
            if record.get("model"):
                summary["model"] = str(record["model"])
        return summary

    def _append_agent_record(self, record: Dict[str, Any]) -> None:
        """Persist one record under a per-agent lock so same-agent workers do not clobber each other."""
        def _append() -> None:
            existing = self._load_json_file(self.agent_log_file, [])
            if not isinstance(existing, list):
                logging.warning(
                    "Agent usage file %s has unexpected format %s; resetting to empty list",
                    self.agent_log_file,
                    type(existing),
                )
                existing = []
            existing.append(record)
            self._atomic_write_json(self.agent_log_file, existing)
            self.usage_records = existing

        self._with_lock(self.agent_lock_file, _append)

    def _rebuild_global_summary_from_agent_logs(self) -> Dict[str, Any]:
        """Rebuild the global summary from all per-agent token usage files."""
        summary: Dict[str, Any] = {}
        if not os.path.isdir(self.output_dir):
            return summary

        for file_name in sorted(os.listdir(self.output_dir)):
            if not file_name.startswith("token_usage_") or not file_name.endswith(".json"):
                continue
            if file_name == "token_usage.json":
                continue
            if file_name.startswith("token_usage_.tmp"):
                continue
            agent_name = file_name[len("token_usage_") : -len(".json")]
            file_path = os.path.join(self.output_dir, file_name)
            records = self._load_json_file(file_path, [])
            if not isinstance(records, list):
                logging.warning(
                    "Agent usage file %s has unexpected format %s; skipping from global summary",
                    file_path,
                    type(records),
                )
                continue
            summary[agent_name] = self._summarize_agent_records(records)
        return summary

    def _load_existing_records(self):
        """Load existing agent-specific token usage records from file."""
        loaded = self._load_json_file(self.agent_log_file, [])
        self.usage_records = loaded if isinstance(loaded, list) else []

    def _load_global_summary(self):
        """Load existing global token usage summary from file."""
        loaded_data = self._load_json_file(self.global_log_file, {})
        if isinstance(loaded_data, dict):
            self.global_summary = loaded_data
        else:
            logging.warning(
                "Global summary file has unexpected format (type: %s), resetting to empty dict",
                type(loaded_data),
            )
            self.global_summary = {}

    def record_usage(
        self,
        agent_name: str,
        model: str,
        usage_info: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
    ):
        """Record token usage for an agent call."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "model": model,
            "prompt_tokens": usage_info.get("prompt_tokens", 0),
            "cached_prompt_tokens": usage_info.get("cached_prompt_tokens", 0),
            "completion_tokens": usage_info.get("completion_tokens", 0),
            "total_tokens": usage_info.get("total_tokens", 0),
            "estimated_cost": usage_info.get("estimated_cost", 0)
        }
        if metadata:
            record["metadata"] = dict(metadata)

        self._append_agent_record(record)
        self._save_global_summary()

        return record

    def _save_agent_records(self):
        """Save agent-specific records to file."""
        self._atomic_write_json(self.agent_log_file, self.usage_records)

    def _update_global_summary(self, record: Dict[str, Any]):
        """Update global summary with new record."""
        # Ensure global_summary is a dict
        if not isinstance(self.global_summary, dict):
            logging.warning(f"Global summary is not a dict (type: {type(self.global_summary)}), resetting")
            self.global_summary = {}
        
        agent = record["agent"]
        if agent not in self.global_summary:
            self.global_summary[agent] = {
                "total_calls": 0,
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_cached_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost": 0.0,
                "model": record["model"]
            }

        summary = self.global_summary[agent]
        summary["total_calls"] += 1
        summary["total_tokens"] += record["total_tokens"]
        summary["total_prompt_tokens"] += record["prompt_tokens"]
        summary["total_cached_prompt_tokens"] += record.get("cached_prompt_tokens", 0)
        summary["total_completion_tokens"] += record["completion_tokens"]
        summary["total_cost"] += record.get("estimated_cost", 0)

    def _save_global_summary(self):
        """Save a globally consistent summary rebuilt from all per-agent logs under a global lock."""
        def _rebuild_and_write() -> None:
            self.global_summary = self._rebuild_global_summary_from_agent_logs()
            self._atomic_write_json(self.global_log_file, self.global_summary)

        self._with_lock(self.global_lock_file, _rebuild_and_write)
    
    def get_agent_usage(self, agent_name: str) -> Dict[str, Any]:
        """Get usage statistics for a specific agent from global summary."""
        return self.global_summary.get(agent_name, {
            "total_calls": 0,
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_cached_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost": 0.0
        })


class LLMClient:
    """OpenAI client wrapper for calling LLM APIs with the configured endpoint."""

    def __init__(self, api_config: Dict[str, Any], output_dir: str, agent_name: str = "unknown") -> None:
        self.base_url = api_config.get("base_url", "")
        self.api_key = api_config.get("api_key", "")
        self.model = api_config.get("model", "")
        self.reasoning_effort = str(api_config.get("reasoning_effort", "") or "").strip()
        self.agent_name = agent_name
        self.request_timeout = float(api_config.get("request_timeout", 600))
        self.retry_backoff_seconds = float(api_config.get("retry_backoff_seconds", 2.0))
        # Extra attempts after the first HTTP/SDK try (total = 1 + this value).
        _mr = api_config.get("max_retry_times", api_config.get("max_retries", 3))
        self.default_max_retry_times = max(int(_mr), 0)
        # Extra full completion rounds when JSON repair still fails (each round re-calls the LLM).
        _jr = api_config.get("json_parse_retries", 2)
        self.default_json_parse_retries = max(int(_jr), 0)
        self.enable_output_token_routing = bool(api_config.get("enable_output_token_routing", False))
        self.short_output_model = str(api_config.get("short_output_model", "deepseek-chat") or "deepseek-chat")
        self.short_output_max_tokens = max(int(api_config.get("short_output_max_tokens", 8192)), 1)
        self.long_output_model = str(api_config.get("long_output_model", self.model) or self.model)
        self.long_output_max_tokens = max(int(api_config.get("long_output_max_tokens", 0) or 0), 0)
        self.output_token_rerun_margin = max(int(api_config.get("output_token_rerun_margin", 32)), 0)

        if not self.api_key:
            raise ValueError("API key is required for LLM calls")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        # Initialize token tracker
        parent_dir = os.path.dirname(output_dir)
        self.token_tracker = TokenTracker(parent_dir, self.agent_name)

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 32768,
        max_retry_times: int | None = None,
        timeout_seconds: float | None = None,
        operation_name: str | None = None,
        usage_metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Call the LLM API and return the response text."""
        effective_timeout = self.request_timeout if timeout_seconds is None else float(timeout_seconds)
        cfg_retries = (
            self.default_max_retry_times if max_retry_times is None else max(0, int(max_retry_times))
        )
        max_attempts = cfg_retries + 1
        op_name = str(operation_name or self.agent_name or "llm_call").strip()

        def _route_for_request(
            requested_max_tokens: int,
            metadata: Dict[str, Any] | None,
        ) -> tuple[str, int, str, str]:
            if not self.enable_output_token_routing:
                return self.model, requested_max_tokens, "default", ""
            if requested_max_tokens <= self.short_output_max_tokens:
                return self.short_output_model, requested_max_tokens, "short", ""
            metadata = metadata or {}
            for key in ("largest_file_chars", "raw_chars", "prompt_chars", "failure_chars"):
                try:
                    value = int(metadata.get(key, 0) or 0)
                except Exception:
                    value = 0
                if value > self.short_output_max_tokens:
                    return self.long_output_model, self.long_output_max_tokens or requested_max_tokens, "long_direct", key
            return self.short_output_model, self.short_output_max_tokens, "short", ""

        def _record_response_usage(response: Any, model_used: str, metadata: Dict[str, Any]) -> int | None:
            if not (hasattr(response, 'usage') and response.usage):
                return None
            prompt_tokens_details = getattr(response.usage, "prompt_tokens_details", None)
            cached_prompt_tokens = 0
            if prompt_tokens_details is not None:
                cached_prompt_tokens = getattr(prompt_tokens_details, "cached_tokens", 0) or 0
                if not cached_prompt_tokens and isinstance(prompt_tokens_details, dict):
                    cached_prompt_tokens = prompt_tokens_details.get("cached_tokens", 0) or 0
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "cached_prompt_tokens": cached_prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            self.token_tracker.record_usage(
                self.agent_name,
                model_used,
                usage_info,
                metadata=metadata,
            )
            return int(response.usage.completion_tokens or 0)

        def _extract_response_text(response: Any) -> tuple[str, Dict[str, Any]]:
            reply_text = ""
            response_debug: Dict[str, Any] = {}
            if hasattr(response, 'choices'):
                choices = response.choices
                if isinstance(choices, list) and len(choices) > 0:
                    first_choice = choices[0]
                    msg_obj = getattr(first_choice, "message", None)
                    raw = getattr(msg_obj, "content", None) if msg_obj is not None else None
                    reply_text = str(raw).strip() if raw is not None else ""
                    response_debug = {
                        "finish_reason": getattr(first_choice, "finish_reason", None),
                        "message_role": getattr(msg_obj, "role", None) if msg_obj is not None else None,
                        "content_type": type(raw).__name__ if raw is not None else "None",
                        "has_tool_calls": bool(getattr(msg_obj, "tool_calls", None)) if msg_obj is not None else False,
                        "has_refusal": bool(getattr(msg_obj, "refusal", None)) if msg_obj is not None else False,
                    }
                elif isinstance(choices, dict):
                    logging.warning(
                        "Response.choices is a dict, not a list. Keys: %s",
                        list(choices.keys()),
                    )
                    first_choice = next(iter(choices.values())) if choices else None
                    if first_choice and hasattr(first_choice, 'message'):
                        raw = getattr(first_choice.message, "content", None)
                        reply_text = str(raw).strip() if raw is not None else ""
                        response_debug = {
                            "finish_reason": getattr(first_choice, "finish_reason", None),
                            "message_role": getattr(first_choice.message, "role", None),
                            "content_type": type(raw).__name__ if raw is not None else "None",
                            "has_tool_calls": bool(getattr(first_choice.message, "tool_calls", None)),
                            "has_refusal": bool(getattr(first_choice.message, "refusal", None)),
                        }
            return reply_text, response_debug

        def _should_rerun_long(
            route_name: str,
            requested_max_tokens: int,
            routed_max_tokens: int,
            completion_tokens: int | None,
            response_debug: Dict[str, Any],
        ) -> bool:
            if route_name != "short" or routed_max_tokens >= requested_max_tokens:
                return False
            if response_debug.get("finish_reason") == "length":
                return True
            if completion_tokens is None:
                return False
            return completion_tokens >= max(routed_max_tokens - self.output_token_rerun_margin, 1)

        for attempt in range(1, max_attempts + 1):
            try:
                metadata_base = dict(usage_metadata or {})
                model_used, routed_max_tokens, route_name, route_reason = _route_for_request(max_tokens, metadata_base)
                metadata_base.update(
                    {
                        "operation_name": op_name,
                        "output_token_route": route_name,
                        "output_token_route_reason": route_reason,
                        "requested_max_tokens": max_tokens,
                        "actual_max_tokens": routed_max_tokens,
                    }
                )
                request_kwargs = {
                    "model": model_used,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": routed_max_tokens,
                    "timeout": effective_timeout,
                }
                if self.reasoning_effort:
                    request_kwargs["reasoning_effort"] = self.reasoning_effort
                response = self.client.chat.completions.create(**request_kwargs)

                completion_tokens: int | None = None
                reply_text, response_debug = _extract_response_text(response)
                metadata_base["finish_reason"] = response_debug.get("finish_reason")
                # Record token usage (do not fail request if usage logging fails)
                try:
                    completion_tokens = _record_response_usage(response, model_used, metadata_base)
                except Exception as token_err:
                    logging.warning(f"Failed to record token usage: {token_err}")

                if _should_rerun_long(route_name, max_tokens, routed_max_tokens, completion_tokens, response_debug):
                    long_max_tokens = self.long_output_max_tokens or max_tokens
                    long_metadata = dict(usage_metadata or {})
                    long_metadata.update(
                        {
                            "operation_name": op_name,
                            "output_token_route": "long_rerun",
                            "requested_max_tokens": max_tokens,
                            "actual_max_tokens": long_max_tokens,
                            "rerun_from_model": model_used,
                            "rerun_from_max_tokens": routed_max_tokens,
                            "rerun_from_finish_reason": response_debug.get("finish_reason"),
                            "rerun_from_completion_tokens": completion_tokens,
                        }
                    )
                    logging.info(
                        "Re-running LLM call with long output config: agent=%s operation=%s short_model=%s short_max_tokens=%s short_completion_tokens=%s short_finish_reason=%s long_model=%s long_max_tokens=%s",
                        self.agent_name,
                        op_name,
                        model_used,
                        routed_max_tokens,
                        completion_tokens,
                        response_debug.get("finish_reason"),
                        self.long_output_model,
                        long_max_tokens,
                    )
                    long_kwargs = dict(request_kwargs)
                    long_kwargs["model"] = self.long_output_model
                    long_kwargs["max_tokens"] = long_max_tokens
                    response = self.client.chat.completions.create(**long_kwargs)
                    reply_text, response_debug = _extract_response_text(response)
                    long_metadata["finish_reason"] = response_debug.get("finish_reason")
                    try:
                        completion_tokens = _record_response_usage(response, self.long_output_model, long_metadata)
                    except Exception as token_err:
                        logging.warning(f"Failed to record long-rerun token usage: {token_err}")

                if reply_text:
                    return reply_text

                if attempt < max_attempts:
                    delay_seconds = max(self.retry_backoff_seconds * (2 ** (attempt - 1)), 0.0)
                    logging.warning(
                        "LLM returned empty completion text: agent=%s operation=%s model=%s reasoning_effort=%s timeout=%.1fs attempt=%d/%d debug=%s. Retrying in %.1fs",
                        self.agent_name,
                        op_name,
                        model_used,
                        self.reasoning_effort,
                        effective_timeout,
                        attempt,
                        max_attempts,
                        response_debug,
                        delay_seconds,
                    )
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    continue

                logging.warning(
                    "Empty response from LLM after %d attempt(s): agent=%s operation=%s model=%s reasoning_effort=%s timeout=%.1fs debug=%s",
                    max_attempts,
                    self.agent_name,
                    op_name,
                    model_used,
                    self.reasoning_effort,
                    effective_timeout,
                    response_debug,
                )
                return ""

            except Exception as e:
                error_type = type(e).__name__
                is_timeout = error_type == "APITimeoutError"
                retryable_sdk = isinstance(e, _RETRYABLE_OPENAI_EXCEPTIONS)

                # Retry on common transient HTTP/API failures (e.g., 429/5xx),
                # even when the exception type isn't a timeout.
                err_text = str(e)
                status_code: int | None = None
                status_match = re.search(
                    r"(?i)(?:error code|status code)\s*[:=]?\s*(\d{3})",
                    err_text,
                )
                if status_match:
                    try:
                        status_code = int(status_match.group(1))
                    except Exception:
                        status_code = None

                retryable_http = status_code in {429, 500, 502, 503, 504}
                retryable = is_timeout or retryable_http or retryable_sdk

                if retryable and attempt < max_attempts:
                    delay_seconds = max(self.retry_backoff_seconds * (2 ** (attempt - 1)), 0.0)
                    logging.warning(
                        "LLM API call failed (agent=%s operation=%s model=%s reasoning_effort=%s timeout=%.1fs retryable=%s status_code=%s error_type=%s) attempt %d/%d. Retrying in %.1f seconds. err=%s",
                        self.agent_name,
                        op_name,
                        self.model,
                        self.reasoning_effort,
                        effective_timeout,
                        retryable,
                        status_code,
                        error_type,
                        attempt,
                        max_attempts,
                        delay_seconds,
                        err_text,
                    )
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    continue

                if is_timeout:
                    logging.error(
                        "LLM API call timed out after %.1f seconds: agent=%s operation=%s model=%s reasoning_effort=%s err=%s",
                        effective_timeout,
                        self.agent_name,
                        op_name,
                        self.model,
                        self.reasoning_effort,
                        e,
                    )
                logging.error(
                    "LLM API call failed: agent=%s operation=%s model=%s reasoning_effort=%s error_type=%s err=%s",
                    self.agent_name,
                    op_name,
                    self.model,
                    self.reasoning_effort,
                    error_type,
                    e,
                )
                raise RuntimeError(f"LLM API call failed: {e}") from e

        return ""

    def call_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 32768,
        max_retry_times: int | None = None,
        timeout_seconds: float | None = None,
        json_parse_retries: int | None = None,
        operation_name: str | None = None,
        usage_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | List[Any]:
        """Call the LLM and parse the response as JSON (can return dict or list)."""
        llm_retries = (
            self.default_max_retry_times if max_retry_times is None else max(0, int(max_retry_times))
        )
        extra_parse = (
            self.default_json_parse_retries
            if json_parse_retries is None
            else max(0, int(json_parse_retries))
        )
        max_parse_rounds = 1 + extra_parse
        last_exc: BaseException | None = None

        for parse_round in range(1, max_parse_rounds + 1):
            response_text = self.call(
                messages,
                temperature,
                max_tokens,
                max_retry_times=llm_retries,
                timeout_seconds=timeout_seconds,
                operation_name=operation_name,
                usage_metadata=usage_metadata,
            )
            try:
                return self._parse_llm_json_response(response_text)
            except (RuntimeError, ValueError) as e:
                last_exc = e
                if parse_round < max_parse_rounds:
                    delay_seconds = max(self.retry_backoff_seconds * (2 ** (parse_round - 1)), 0.0)
                    logging.warning(
                        "call_json: parse failed (agent=%s operation=%s model=%s reasoning_effort=%s round %d/%d): %s; re-requesting LLM in %.1fs",
                        self.agent_name,
                        str(operation_name or self.agent_name or "llm_call").strip(),
                        self.model,
                        self.reasoning_effort or "",
                        parse_round,
                        max_parse_rounds,
                        e,
                        delay_seconds,
                    )
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    continue
                raise
        raise RuntimeError("call_json: exhausted parse retries") from last_exc

    def _parse_llm_json_response(self, response_text: str) -> Dict[str, Any] | List[Any]:
        """Parse model output into JSON; raises RuntimeError/ValueError on failure."""
        original_response = response_text
        fixed_text = None

        if not response_text or not response_text.strip():
            logging.error("Empty response from LLM; cannot parse JSON")
            raise RuntimeError("LLM response is empty")
        
        try:
            response_text = _strip_bom_and_format_chars(response_text)
            # Try to extract JSON from markdown code blocks if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()

            response_text = response_text.strip()
            if not response_text.startswith(("{", "[")):
                json_start = -1
                for char in ["{", "["]:
                    pos = response_text.find(char)
                    if pos != -1 and (json_start == -1 or pos < json_start):
                        json_start = pos
                if json_start != -1:
                    response_text = response_text[json_start:]

            complete = _slice_first_complete_json(response_text)
            if complete:
                response_text = complete

            normalized = _collapse_duplicate_commas(
                _strip_trailing_commas_outside_strings(response_text)
            )
            if normalized != response_text:
                response_text = normalized
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            logging.error(f"Response text: {original_response}")
            logging.error(f"Response: {response_text}")

            raise ValueError(f"Error parsing response: {e}") from e

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raw_parsed = _raw_decode_json_prefix(response_text)
            if raw_parsed is not None:
                logging.info("Parsed JSON via raw_decode (trailing non-JSON or partial consume)")
                return raw_parsed

            suffix_closed = _try_suffix_close_unterminated(response_text)
            if suffix_closed:
                try:
                    result = json.loads(suffix_closed)
                    logging.info(
                        "LLM JSON parse repair succeeded: suffix close (unterminated); parse OK"
                    )
                    return result
                except json.JSONDecodeError:
                    raw2 = _raw_decode_json_prefix(suffix_closed)
                    if raw2 is not None:
                        logging.info("Parsed JSON after suffix close via raw_decode")
                        return raw2

            # Try to fix common JSON issues
            logging.warning(f"Initial JSON parsing failed at position {e.pos}: {e.msg}")
            fixed_text = None

            def _repair_return(result_obj: Any, detail: str):
                logging.info("LLM JSON parse repair succeeded: %s; parse OK", detail)
                return result_obj

            # Attempt -1: recover known code-output schemas whose long source
            # strings broke JSON escaping. This avoids wasting another LLM call
            # when the structure is still recoverable.
            schema_repaired = _repair_known_code_json_payload(response_text)
            if schema_repaired is not None:
                return _repair_return(schema_repaired, "recovered malformed code-output JSON schema")

            # Attempt 0: insert missing commas between values and the next key/object
            try:
                candidate_text = _fix_missing_json_commas(response_text)
                if candidate_text != response_text:
                    fixed_text = candidate_text
                    result = json.loads(candidate_text)
                    logging.info("Successfully fixed JSON by inserting missing commas")
                    return _repair_return(result, "inserted missing commas")
            except json.JSONDecodeError:
                pass

            # Attempt 0b: close missing object before a new object in arrays
            if "Expecting ',' delimiter" in e.msg:
                candidate_text = _fix_missing_object_closers(response_text, e.pos)
                if candidate_text:
                    try:
                        fixed_text = candidate_text
                        result = json.loads(candidate_text)
                        logging.info("Successfully fixed JSON by closing missing object")
                        return _repair_return(result, "closed missing object")
                    except json.JSONDecodeError:
                        pass

                # Attempt 0b-2: close missing array/object delimiter before a mismatched closer.
                candidate_text = _fix_missing_container_closer(response_text, e.pos)
                if candidate_text:
                    try:
                        fixed_text = candidate_text
                        result = json.loads(candidate_text)
                        logging.info("Successfully fixed JSON by closing missing container delimiter")
                        return _repair_return(result, "closed missing container delimiter")
                    except json.JSONDecodeError:
                        pass

                # Attempt 0c: escape bare quotes inside known large string payload fields.
                for string_field in ("code", "content"):
                    candidate_text = _escape_unescaped_quotes_in_string_field(response_text, string_field)
                    if candidate_text != response_text:
                        try:
                            fixed_text = candidate_text
                            result = json.loads(candidate_text)
                            logging.info(
                                "Successfully fixed JSON by escaping unescaped quotes in '%s' field",
                                string_field,
                            )
                            return _repair_return(
                                result, f"escaped unescaped quotes in '{string_field}' field"
                            )
                        except json.JSONDecodeError:
                            continue

            # Attempt 0d: sanitize malformed string payloads in common long-text fields.
            if any(err in e.msg for err in ("Invalid \\escape", "Invalid control character", "Expecting ',' delimiter")):
                candidate_text = response_text
                candidate_text = _escape_invalid_backslashes_in_all_json_strings(candidate_text)
                for string_field in _JSON_LONG_TEXT_FIELD_NAMES:
                    candidate_text = _escape_unescaped_quotes_in_string_field(candidate_text, string_field)
                    candidate_text = _escape_invalid_backslashes_in_string_field(candidate_text, string_field)
                    candidate_text = _escape_control_chars_in_string_field(candidate_text, string_field)
                if candidate_text != response_text:
                    try:
                        fixed_text = candidate_text
                        result = json.loads(candidate_text)
                        logging.info("Successfully fixed JSON by sanitizing malformed string fields")
                        return _repair_return(result, "sanitized malformed string fields")
                    except json.JSONDecodeError:
                        pass
            
            # Attempt 1: Try to fix unterminated strings by finding the last valid position
            if "Unterminated string" in e.msg or "Expecting property name" in e.msg:
                logging.info("Attempting to fix unterminated JSON strings...")
                
                # Find the position of the error
                error_pos = e.pos
                
                # Try to find the last complete key-value pair before the error
                # Work backwards from error position to find valid JSON structure
                truncated_text = response_text[:error_pos]
                
                # Try to close any open strings and objects
                _max_unterm_scan = 32000
                for attempt_pos in range(error_pos, max(0, error_pos - _max_unterm_scan), -1):
                    test_text = response_text[:attempt_pos].rstrip()
                    
                    # Count unclosed braces and brackets
                    brace_depth = 0
                    bracket_depth = 0
                    in_string = False
                    escape_next = False
                    
                    for char in test_text:
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                            continue
                        if char == '"' and not in_string:
                            in_string = True
                        elif char == '"' and in_string:
                            in_string = False
                        elif not in_string:
                            if char == '{':
                                brace_depth += 1
                            elif char == '}':
                                brace_depth -= 1
                            elif char == '[':
                                bracket_depth += 1
                            elif char == ']':
                                bracket_depth -= 1
                    
                    # Try to close the structure
                    closing = ''
                    if in_string:
                        closing += '"'
                    
                    # Close any trailing comma or incomplete key-value
                    if test_text.rstrip().endswith(','):
                        test_text = test_text.rstrip()[:-1]
                    elif test_text.rstrip().endswith(':'):
                        test_text = test_text.rstrip()[:-1]
                    
                    # Add closing brackets
                    closing += '}' * brace_depth
                    closing += ']' * bracket_depth
                    
                    fixed_text = test_text + closing
                    
                    try:
                        result = json.loads(fixed_text)
                        logging.info(f"Successfully fixed JSON by truncating at position {attempt_pos}")
                        return _repair_return(
                            result, f"truncated unterminated string near position {attempt_pos}"
                        )
                    except json.JSONDecodeError:
                        continue
            
            # If all attempts fail, log and raise
            logging.error(f"JSON parsing failed at position {e.pos}")
            logging.error(f"Error: {e.msg}")
            logging.error(f"Response length: {len(original_response)} chars")
            logging.error(f"Response_processed: {fixed_text if fixed_text is not None else '<unset>'}")
            logging.error(f"Response: {original_response}")
            logging.error(f"Extracted JSON preview (first 500 chars): {response_text[:500]}")
            logging.error(f"Extracted JSON preview (last 500 chars): {response_text[-500:]}")
            raise RuntimeError(f"LLM response is not valid JSON: {e.msg} at position {e.pos}") from e
