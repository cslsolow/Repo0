import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_AGENT_PATH = ROOT / "agents" / "coding" / "patch_agent.py"

spec = importlib.util.spec_from_file_location("patch_agent", PATCH_AGENT_PATH)
assert spec is not None and spec.loader is not None
patch_agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch_agent_mod)
PatchAgent = patch_agent_mod.PatchAgent


def test_apply_patch_text_updates_existing_file():
    agent = PatchAgent(api_config={})
    related = {
        "pkg/mod.py": "def add(a, b):\n    return a+b\n",
    }
    patch = """--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a+b
+    return a + b
"""

    result = agent.apply_patch_text(patch, related)

    print(result)
    # assert False
    assert "pkg/mod.py" in result["touched_files"]
    assert result["updated_files"]["pkg/mod.py"] == "def add(a, b):\n    return a + b\n"


def test_apply_patch_text_creates_new_file():
    agent = PatchAgent(api_config={})
    related = {}
    patch = """--- /dev/null
+++ b/new_feature.py
@@ -0,0 +1,2 @@
+def ping():
+    return \"pong\"
"""

    result = agent.apply_patch_text(patch, related)

    assert "new_feature.py" in result["created_files"]
    assert result["updated_files"]["new_feature.py"].startswith("def ping()")


def test_apply_patch_text_rejects_apply_patch_style_format():
    agent = PatchAgent(api_config={})
    related = {"pkg/mod.py": "def add(a, b):\n    return a+b\n"}
    patch = """*** Begin Patch
*** Update File: pkg/mod.py
@@
-def add(a, b):
+def add(a, b):
*** End Patch
"""

    try:
        agent.apply_patch_text(patch, related)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "standard unified diff" in str(exc)


def test_is_strict_unified_diff_accepts_only_real_unified_diff():
    patch = """--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1 +1 @@
-x = 1
+x = 2
"""
    assert PatchAgent._is_strict_unified_diff(patch) is True
    assert PatchAgent._is_strict_unified_diff("*** Begin Patch\n*** End Patch\n") is False


def test_generate_patch_with_llm_uses_split_diff_and_full_file_requests():
    class StubLLMClient:
        def __init__(self):
            self.calls = []

        def call_json(self, messages, **kwargs):
            self.calls.append({"messages": messages, "kwargs": kwargs})
            op = kwargs.get("operation_name")
            if str(op).endswith(".diff"):
                return {
                    "patch": "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a+b\n+    return a + b\n",
                    "touched_files": ["pkg/mod.py"],
                    "summary": "diff",
                }
            if str(op).endswith(".full_file"):
                return {
                    "updated_files": [{"path": "pkg/mod.py", "content": "def add(a, b):\n    return a + b\n"}],
                    "touched_files": ["pkg/mod.py"],
                    "summary": "full",
                }
            raise AssertionError(f"unexpected operation name: {op}")

    agent = PatchAgent(api_config={})
    agent.llm_client = StubLLMClient()
    telemetry = {"scenario": "syntax_failure"}

    result = agent._generate_patch_with_llm(
        task_description="fix spacing",
        related_files={"pkg/mod.py": "def add(a, b):\n    return a+b\n"},
        compile_error="",
        incremental_goal="",
        failure_kind="syntax_failure",
        telemetry=telemetry,
    )

    assert result["patch"].startswith("--- a/pkg/mod.py")
    assert result["updated_files"]["pkg/mod.py"] == "def add(a, b):\n    return a + b\n"
    assert result["diff_updated_files"]["pkg/mod.py"] == "def add(a, b):\n    return a + b\n"
    assert result["full_file_updated_files"]["pkg/mod.py"] == "def add(a, b):\n    return a + b\n"
    assert len(agent.llm_client.calls) == 2
    assert str(agent.llm_client.calls[0]["kwargs"]["operation_name"]).endswith(".diff")
    assert str(agent.llm_client.calls[1]["kwargs"]["operation_name"]).endswith(".full_file")


def test_generate_patch_heuristic_fixes_missing_colon_from_compile_error():
    agent = PatchAgent(api_config={})
    related = {
        "src/demo.py": "def run(x)\n    return x\n",
    }
    error = 'Traceback ...\\n  File "/tmp/project/src/demo.py", line 1\\n    def run(x)\\nSyntaxError: expected \':\''

    result = agent.generate_patch(
        task_description="fix syntax error",
        related_files=related,
        compile_error=error,
        incremental_goal="",
    )

    assert "src/demo.py" in result["touched_files"]
    assert "def run(x):" in result["updated_files"]["src/demo.py"]
    assert "--- a/src/demo.py" in result["patch"]


def test_generate_patch_heuristic_supports_incremental_stub():
    agent = PatchAgent(api_config={})
    related = {
        "core/service.py": "class Service:\n    pass\n",
    }

    result = agent.generate_patch(
        task_description="增量开发",
        related_files=related,
        compile_error="",
        incremental_goal="新增函数 calculate_total",
    )

    assert "core/service.py" in result["touched_files"]
    assert "def calculate_total" in result["updated_files"]["core/service.py"]


def test_extract_focus_lines_matches_traceback_paths():
    agent = PatchAgent(api_config={})
    related = {
        "pkg/mod.py": "a\nb\nc\n",
        "tests/test_mod.py": "x\ny\nz\n",
    }
    failure = (
        'Traceback (most recent call last):\n'
        '  File "/tmp/work/pkg/mod.py", line 2, in <module>\n'
        "    boom()\n"
        'tests/test_mod.py:3: AssertionError\n'
    )

    focus = agent._extract_focus_lines(failure, related)

    assert focus["pkg/mod.py"] == {2}
    assert focus["tests/test_mod.py"] == {3}


def test_build_related_files_prompt_payload_keeps_large_files_full():
    agent = PatchAgent(api_config={})
    large = "\n".join(f"line {i}" for i in range(1, 4000))
    related = {"pkg/mod.py": large}
    failure = 'File "/tmp/work/pkg/mod.py", line 250, in <module>'

    payload = agent._build_related_files_prompt_payload(
        related_files=related,
        failure_text=failure,
        scenario="bug_fix",
        max_full_chars=100,
    )

    assert payload[0]["content_mode"] == "full"
    assert payload[0]["focus_lines"] == [250]
    assert "line 3999" in payload[0]["content"]


def test_build_related_files_prompt_payload_keeps_primary_file_full_for_compile_error():
    agent = PatchAgent(api_config={})
    large = "\n".join(f"line {i}" for i in range(1, 4000))
    related = {
        "pkg/mod.py": large,
        "tests/test_mod.py": "def test_x():\n    assert True\n",
    }
    failure = 'File "/tmp/work/pkg/mod.py", line 250, in <module>'

    payload = agent._build_related_files_prompt_payload(
        related_files=related,
        failure_text=failure,
        scenario="compile_error_fix",
        max_full_chars=100,
    )

    assert payload[0]["path"] == "pkg/mod.py"
    assert payload[0]["content_mode"] == "full"
    assert "line 3999" in payload[0]["content"]


def test_build_related_files_prompt_payload_uses_import_policy_for_missing_module():
    agent = PatchAgent(api_config={})
    large = "\n".join(f"line {i}" for i in range(1, 120))
    related = {
        "pkg/mod.py": large,
        "tests/test_mod.py": "import pkg.mod\n\ndef test_x():\n    assert True\n",
    }
    failure = (
        "tests/test_mod.py:1: in <module>\n"
        "    import pkg.mod\n"
        "pkg/mod.py:20: in <module>\n"
        "    from pkg.missing import X\n"
        "E   ModuleNotFoundError: No module named 'pkg.missing'\n"
    )

    payload = agent._build_related_files_prompt_payload(
        related_files=related,
        failure_text=failure,
        scenario="import_failure",
        max_full_chars=100,
    )

    assert payload[0]["path"] == "pkg/mod.py"
    assert payload[0]["content_mode"] == "full"
    assert payload[1]["path"] == "tests/test_mod.py"
    assert payload[1]["content_mode"] == "full"


def test_detect_patch_scenario_promotes_missing_module_test_failure_to_import_failure():
    agent = PatchAgent(api_config={})
    scenario = agent._detect_patch_scenario(
        "Pytest failed for component X",
        "E   ModuleNotFoundError: No module named 'pkg.missing'",
        "Fix implementation and/or tests so tests pass.",
        failure_kind="test_failure",
    )

    assert scenario == "import_failure"


def test_generate_patch_records_analysis_for_heuristic_calls():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "agents_output"
        out.mkdir(parents=True, exist_ok=True)
        agent = PatchAgent(api_config={}, output_dir=str(out))
        related = {"src/demo.py": "def run(x)\n    return x\n"}
        error = 'Traceback ...\n  File "/tmp/project/src/demo.py", line 1\n    def run(x)\nSyntaxError: expected \':\''

        result = agent.generate_patch(
            task_description="fix syntax error",
            related_files=related,
            compile_error=error,
            incremental_goal="",
            failure_kind="syntax_failure",
            telemetry_context={"component_name": "DemoComponent", "stage": "unit_test"},
        )

        assert "src/demo.py" in result["touched_files"]
        events_path = Path(tmp) / "patch_agent_events.json"
        summary_path = Path(tmp) / "patch_agent_analysis.json"
        optimization_path = Path(tmp) / "patch_agent_optimization_report.json"
        assert events_path.exists()
        assert summary_path.exists()
        assert optimization_path.exists()

        events = json.loads(events_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        optimization = json.loads(optimization_path.read_text(encoding="utf-8"))
        assert len(events) == 1
        assert events[0]["component_name"] == "DemoComponent"
        assert events[0]["scenario"] == "syntax_failure"
        assert events[0]["mode"] == "heuristic"
        assert summary["total_calls"] == 1
        assert summary["heuristic_calls"] == 1
        assert summary["by_component"]["DemoComponent"]["calls"] == 1
        assert summary["by_scenario"]["syntax_failure"]["calls"] == 1
        assert optimization["overview"]["total_calls"] == 1
        assert optimization["scenario_decisions"][0]["scenario"] == "syntax_failure"
        assert optimization["scenario_decisions"][0]["effective_update_rate"] == 1.0
