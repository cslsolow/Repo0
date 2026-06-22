import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.llm_client import (  # noqa: E402
    LLMClient,
    TokenTracker,
    _escape_unescaped_quotes_in_string_field,
    _escape_invalid_backslashes_in_string_field,
    _escape_control_chars_in_string_field,
    _fix_missing_json_commas,
    _fix_missing_object_closers,
    _fix_missing_container_closer,
)


def test_fix_missing_json_commas_between_string_and_key():
    raw = '[{"description": "x"\n "order": 1}]'
    fixed = _fix_missing_json_commas(raw)
    print(fixed)
    assert fixed != raw
    assert json.loads(fixed) == [{"description": "x", "order": 1}]


def test_fix_missing_json_commas_between_objects():
    raw = '[{"a": 1}\n{"b": 2}]'
    fixed = _fix_missing_json_commas(raw)
    assert fixed != raw
    assert json.loads(fixed) == [{"a": 1}, {"b": 2}]


def test_fix_missing_json_commas_no_change():
    raw = '{"a": 1, "b": 2}'
    assert _fix_missing_json_commas(raw) == raw

def test_fix_missing_json_commas_real_like_payload():
    raw = """
[
  {
    "name": "runtime-execution",
    "description": "desc A",
    "order": 0
  },
  {
    "name": "server-adapter-mount",
    "description": "desc B"
    "order": 1
  }
]
""".strip()
    fixed = _fix_missing_json_commas(raw)
    print(fixed)
    assert fixed != raw
    assert json.loads(fixed)[1]["order"] == 1


def test_fix_missing_object_closer_with_empty_object():
    raw = """
{
  "requirements": [
    {
      "name": "A",
      "description": "desc"
    {
    },
    {
      "name": "B",
      "description": "desc2"
    }
  ]
}
""".strip()
    with pytest.raises(json.JSONDecodeError) as excinfo:
        json.loads(raw)
    fixed = _fix_missing_object_closers(raw, excinfo.value.pos)
    assert fixed is not None
    parsed = json.loads(fixed)
    assert [item.get("name") for item in parsed["requirements"] if item] == ["A", "B"]


def test_fix_missing_object_closer_without_empty_object():
    raw = """
{
  "requirements": [
    {
      "name": "A",
      "description": "desc"
    {
      "name": "B",
      "description": "desc2"
    }
  ]
}
""".strip()
    with pytest.raises(json.JSONDecodeError) as excinfo:
        json.loads(raw)
    fixed = _fix_missing_object_closers(raw, excinfo.value.pos)
    assert fixed is not None
    parsed = json.loads(fixed)
    assert [item.get("name") for item in parsed["requirements"]] == ["A", "B"]


def test_fix_missing_container_closer_array_before_object_end():
    raw = """
{
  "components": [
    {
      "name": "PlotCoreAPI",
      "serves_subrequirements": [
        "a",
        "b",
        "c"
    }
  ]
}
""".strip()
    with pytest.raises(json.JSONDecodeError) as excinfo:
        json.loads(raw)

    fixed = _fix_missing_container_closer(raw, excinfo.value.pos)
    assert fixed is not None
    parsed = json.loads(fixed)
    assert parsed["components"][0]["serves_subrequirements"] == ["a", "b", "c"]


def test_call_json_repairs_missing_container_closer(tmp_path):
    raw = """
{
  "components": [
    {
      "name": "PlotCoreAPI",
      "serves_subrequirements": [
        "a",
        "b",
        "c"
    }
  ]
}
""".strip()

    client = LLMClient(
        {
            "api_key": "test-key",
            "base_url": "http://example.com",
            "model": "test-model",
        },
        str(tmp_path),
        agent_name="test",
    )
    client.call = lambda *_args, **_kwargs: raw  # type: ignore[assignment]

    parsed = client.call_json([{"role": "user", "content": "x"}])
    assert parsed["components"][0]["name"] == "PlotCoreAPI"
    assert parsed["components"][0]["serves_subrequirements"] == ["a", "b", "c"]


def test_escape_unescaped_quotes_in_code_field():
    raw = (
        '{'
        '"file_path":"x.py",'
        '"code":"value = data.encode("utf-8")\\nprint(\\"ok\\")"'
        '}'
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)

    fixed = _escape_unescaped_quotes_in_string_field(raw, "code")
    parsed = json.loads(fixed)
    assert parsed["file_path"] == "x.py"
    assert 'encode("utf-8")' in parsed["code"]


def test_escape_invalid_backslashes_in_code_field():
    raw = '{"file_path":"x.py","code":"path = C:\\new\\tmp\\x.py\\q"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)

    fixed = _escape_invalid_backslashes_in_string_field(raw, "code")
    parsed = json.loads(fixed)
    assert "path = C:" in parsed["code"]


def test_escape_control_chars_in_description_field():
    raw = '{\n  "description": "line1\nline2",\n  "order": 1\n}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)

    fixed = _escape_control_chars_in_string_field(raw, "description")
    parsed = json.loads(fixed)
    assert parsed["description"] == "line1\nline2"


def test_call_retries_once_after_timeout(tmp_path):
    class APITimeoutError(Exception):
        pass

    class _FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise APITimeoutError("Request timed out.")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

    completions = _FakeCompletions()
    client = LLMClient(
        {
            "api_key": "test-key",
            "base_url": "http://example.com",
            "model": "test-model",
            "request_timeout": 1,
            "max_retries": 1,
            "retry_backoff_seconds": 0,
        },
        str(tmp_path),
        agent_name="test",
    )
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = client.call([{"role": "user", "content": "hello"}])
    assert result == "ok"
    assert completions.calls == 2


def test_call_json_raw_decode_strips_trailing_noise(tmp_path):
    payload = '{"a": 1, "b": "ok"}\n\nNotes: not JSON'
    client = LLMClient(
        {
            "api_key": "test-key",
            "base_url": "http://example.com",
            "model": "test-model",
            "json_parse_retries": 0,
        },
        str(tmp_path),
        agent_name="test",
    )
    client.call = lambda *_a, **_k: payload  # type: ignore[assignment]

    parsed = client.call_json([{"role": "user", "content": "x"}])
    assert parsed == {"a": 1, "b": "ok"}


def test_call_json_trailing_commas_repaired(tmp_path):
    payload = '{"items": [1, 2, ], "x": "y",}'
    client = LLMClient(
        {
            "api_key": "test-key",
            "base_url": "http://example.com",
            "model": "test-model",
            "json_parse_retries": 0,
        },
        str(tmp_path),
        agent_name="test",
    )
    client.call = lambda *_a, **_k: payload  # type: ignore[assignment]

    parsed = client.call_json([{"role": "user", "content": "x"}])
    assert parsed["items"] == [1, 2]
    assert parsed["x"] == "y"


def test_token_tracker_rebuilds_global_summary_from_all_agent_logs(tmp_path):
    TokenTracker._atomic_write_json(  # type: ignore[attr-defined]
        str(tmp_path / "token_usage_code_generator.json"),
        [
            {
                "timestamp": "2026-01-01T00:00:00",
                "agent": "code_generator",
                "model": "gpt-5-mini",
                "prompt_tokens": 10,
                "cached_prompt_tokens": 2,
                "completion_tokens": 20,
                "total_tokens": 30,
                "estimated_cost": 0.1,
            },
            {
                "timestamp": "2026-01-01T00:00:01",
                "agent": "code_generator",
                "model": "gpt-5-mini",
                "prompt_tokens": 5,
                "cached_prompt_tokens": 0,
                "completion_tokens": 7,
                "total_tokens": 12,
                "estimated_cost": 0.05,
            },
        ],
    )
    TokenTracker._atomic_write_json(  # type: ignore[attr-defined]
        str(tmp_path / "token_usage_patch_agent.json"),
        [
            {
                "timestamp": "2026-01-01T00:00:02",
                "agent": "patch_agent",
                "model": "gpt-5-mini",
                "prompt_tokens": 3,
                "cached_prompt_tokens": 1,
                "completion_tokens": 9,
                "total_tokens": 12,
                "estimated_cost": 0.02,
            }
        ],
    )

    tracker = TokenTracker(str(tmp_path), "architect")
    tracker._save_global_summary()
    merged = json.loads((tmp_path / "token_usage.json").read_text(encoding="utf-8"))

    assert merged["code_generator"]["total_calls"] == 2
    assert merged["code_generator"]["total_prompt_tokens"] == 15
    assert merged["code_generator"]["total_cached_prompt_tokens"] == 2
    assert merged["code_generator"]["total_completion_tokens"] == 27
    assert merged["patch_agent"]["total_calls"] == 1
    assert merged["patch_agent"]["total_tokens"] == 12


def test_token_tracker_record_usage_does_not_clobber_other_agent_summary(tmp_path):
    tracker_a = TokenTracker(str(tmp_path), "code_generator")
    tracker_b = TokenTracker(str(tmp_path), "patch_agent")

    tracker_a.record_usage(
        "code_generator",
        "gpt-5-mini",
        {
            "prompt_tokens": 11,
            "cached_prompt_tokens": 1,
            "completion_tokens": 13,
            "total_tokens": 24,
            "estimated_cost": 0.1,
        },
    )
    tracker_b.record_usage(
        "patch_agent",
        "gpt-5-mini",
        {
            "prompt_tokens": 17,
            "cached_prompt_tokens": 0,
            "completion_tokens": 19,
            "total_tokens": 36,
            "estimated_cost": 0.2,
        },
    )

    merged = json.loads((tmp_path / "token_usage.json").read_text(encoding="utf-8"))
    assert set(merged) == {"code_generator", "patch_agent"}
    assert merged["code_generator"]["total_calls"] == 1
    assert merged["patch_agent"]["total_calls"] == 1
    assert merged["code_generator"]["total_prompt_tokens"] == 11
    assert merged["patch_agent"]["total_completion_tokens"] == 19


def test_call_json_brace_inside_string_does_not_truncate_early(tmp_path):
    payload = (
        'noise prefix {"k": "has } brace inside", "n": 2} trailing junk that is not }} valid'
    )
    client = LLMClient(
        {
            "api_key": "test-key",
            "base_url": "http://example.com",
            "model": "test-model",
            "json_parse_retries": 0,
        },
        str(tmp_path),
        agent_name="test",
    )
    client.call = lambda *_a, **_k: payload  # type: ignore[assignment]

    parsed = client.call_json([{"role": "user", "content": "x"}])
    assert parsed["n"] == 2
    assert "brace" in parsed["k"]
