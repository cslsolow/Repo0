import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = ROOT / "agents" / "coding" / "test_rewrite_agent.py"

spec = importlib.util.spec_from_file_location("test_rewrite_agent", AGENT_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
RewriteAgent = mod.TestRewriteAgent

RUNNER_PATH = ROOT / "scripts" / "run_test_rewrite_agent.py"
runner_spec = importlib.util.spec_from_file_location("run_test_rewrite_agent", RUNNER_PATH)
assert runner_spec is not None and runner_spec.loader is not None
runner_mod = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner_mod)


class FakeLLMClient:
    def call(self, messages, temperature=0.1, max_tokens=1024):  # noqa: ANN001, ANN201
        return "from newpkg.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


class RaisingLLMClient:
    def call(self, messages, temperature=0.1, max_tokens=1024):  # noqa: ANN001, ANN201
        raise RuntimeError("boom")


def test_parse_pytest_summary_counts_basic_fields():
    output = "================ 3 passed, 1 failed, 2 skipped in 0.12s ================"
    summary = RewriteAgent.parse_pytest_summary(output)

    assert summary["passed"] == 3
    assert summary["failed"] == 1
    assert summary["skipped"] == 2
    assert summary["total"] == 6


def test_rewrite_fails_when_llm_not_configured(tmp_path):
    original_repo = tmp_path / "original"
    generated_repo = tmp_path / "generated"

    (original_repo / "tests").mkdir(parents=True)
    (generated_repo / "newpkg").mkdir(parents=True)
    (generated_repo / "newpkg" / "__init__.py").write_text("", encoding="utf-8")
    (generated_repo / "newpkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (original_repo / "tests" / "test_math.py").write_text(
        "from legacy.math_api import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )

    agent = RewriteAgent(api_config={}, output_dir=str(tmp_path / "out"))
    result = agent.rewrite_tests_and_evaluate(original_repo, generated_repo)

    assert result["rewrite_success_count"] == 0
    assert result["rewrite_failure_count"] == 1
    assert result["failed_list"][0]["reason"] == "llm_not_configured"
    assert result["pytest"]["returncode"] is None
    assert result["remaining_pass_rate"] == 0.0


def test_rewrite_fails_when_api_mapping_missing(tmp_path):
    original_repo = tmp_path / "original"
    generated_repo = tmp_path / "generated"

    (original_repo / "tests").mkdir(parents=True)
    (generated_repo / "newpkg").mkdir(parents=True)
    (generated_repo / "newpkg" / "__init__.py").write_text("", encoding="utf-8")
    (generated_repo / "newpkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (original_repo / "tests" / "test_math.py").write_text(
        "from legacy.math_api import subtract\n\n\ndef test_sub():\n    assert subtract(2, 1) == 1\n",
        encoding="utf-8",
    )

    agent = RewriteAgent(api_config={}, output_dir=str(tmp_path / "out"))
    agent.llm_client = FakeLLMClient()

    result = agent.rewrite_tests_and_evaluate(original_repo, generated_repo)

    assert result["rewrite_success_count"] == 0
    assert result["rewrite_failure_count"] == 1
    assert "api_mapping_not_found" in result["failed_list"][0]["reason"]
    assert result["pytest"]["returncode"] is None


def test_rewrite_fails_when_llm_request_fails(tmp_path):
    original_repo = tmp_path / "original"
    generated_repo = tmp_path / "generated"

    (original_repo / "tests").mkdir(parents=True)
    (generated_repo / "newpkg").mkdir(parents=True)
    (generated_repo / "newpkg" / "__init__.py").write_text("", encoding="utf-8")
    (generated_repo / "newpkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (original_repo / "tests" / "test_math.py").write_text(
        "from legacy.math_api import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )

    agent = RewriteAgent(api_config={}, output_dir=str(tmp_path / "out"))
    agent.llm_client = RaisingLLMClient()

    result = agent.rewrite_tests_and_evaluate(original_repo, generated_repo)

    assert result["rewrite_success_count"] == 0
    assert result["rewrite_failure_count"] == 1
    assert result["failed_list"][0]["reason"].startswith("llm_request_failed:")


def test_rewrite_and_evaluate_counts_remaining_subset(tmp_path):
    original_repo = tmp_path / "original"
    generated_repo = tmp_path / "generated"

    (original_repo / "tests").mkdir(parents=True)
    (generated_repo / "newpkg").mkdir(parents=True)

    (generated_repo / "newpkg" / "__init__.py").write_text("", encoding="utf-8")
    (generated_repo / "newpkg" / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )

    (original_repo / "tests" / "test_ok.py").write_text(
        "from legacy.math_api import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (original_repo / "tests" / "test_missing.py").write_text(
        "from legacy.math_api import subtract\n\n\ndef test_sub():\n    assert subtract(2, 1) == 1\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    agent = RewriteAgent(api_config={}, output_dir=str(output_dir))
    agent.llm_client = FakeLLMClient()

    result = agent.rewrite_tests_and_evaluate(
        original_repo_root=original_repo,
        generated_repo_root=generated_repo,
    )

    assert result["test_files_count"] == 2
    assert result["rewrite_success_count"] == 1
    assert result["rewrite_failure_count"] == 1
    assert result["rewrite_success_rate"] == 0.5

    assert result["pytest"]["returncode"] == 0
    assert result["pytest"]["summary"]["passed"] == 1
    assert result["estimated_original_test_cases"] == 2
    assert result["executed_test_cases"] == 1
    assert result["passed_test_cases"] == 1
    assert result["all_tests_pass_rate"] == 0.5
    assert result["remaining_pass_rate"] == 1.0
    assert result["pass_rate"] == 1.0


def test_write_rewrite_manifest_persists_paths_and_flags(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    original_repo = tmp_path / "original"
    generated_repo = tmp_path / "generated"
    tests_root = original_repo / "tests"
    rewritten_root = output_dir / "rewritten_tests"
    result_json = output_dir / "test_rewrite_result.json"

    manifest_path = runner_mod.write_rewrite_manifest(
        output_dir=output_dir,
        original_repo=original_repo,
        generated_repo=generated_repo,
        tests_root=tests_root,
        rewritten_root=rewritten_root,
        result_json=result_json,
        pytest_args=["-q", "--maxfail=1"],
        llm_config={
            "base_url": "http://example.test/v1",
            "model": "gpt-test",
            "reasoning_effort": "medium",
            "api_key": "secret",
        },
        max_fix_rounds=2,
        strict_api_mapping=True,
        evaluate_only=False,
        skip_rewrite_if_exists=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["original_repo_root"] == str(original_repo)
    assert manifest["generated_repo_root"] == str(generated_repo)
    assert manifest["rewritten_tests_root"] == str(rewritten_root)
    assert manifest["pytest_args"] == ["-q", "--maxfail=1"]
    assert manifest["llm_config"]["api_key_present"] is True
    assert manifest["strict_api_mapping"] is True
