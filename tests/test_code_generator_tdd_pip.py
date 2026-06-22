"""Unit tests for TDD dependency preparation (pip / editable install) in CodeGeneratorAgent."""

from __future__ import annotations

import subprocess
import tempfile
import json
from pathlib import Path

from agents.coding.code_generator import CodeGeneratorAgent


def test_tdd_has_project_metadata_detects_packaging_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert CodeGeneratorAgent._tdd_has_project_metadata(root) is False
        (root / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0"\n', encoding="utf-8")
        assert CodeGeneratorAgent._tdd_has_project_metadata(root) is True


def test_tdd_needs_pip_prepare_editable_when_setup_present() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "setup.py").write_text(
            "from setuptools import setup; setup(name='t', packages=[])\n",
            encoding="utf-8",
        )
        agent = CodeGeneratorAgent({"tdd_pip_project_root": str(root)})
        assert agent._tdd_needs_pip_prepare() is True


def test_run_pytest_in_docker_reuses_component_venv_and_markers(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp_root, tempfile.TemporaryDirectory() as tmp_proj:
        sandbox_root = Path(tmp_root)
        project_root = Path(tmp_proj)
        (project_root / "setup.py").write_text(
            "from setuptools import setup; setup(name='t', packages=[])\n",
            encoding="utf-8",
        )
        seen = {}

        def fake_run(cmd, capture_output, text, timeout):
            seen["cmd"] = cmd
            seen["timeout"] = timeout
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr("agents.coding.code_generator.shutil.which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr("agents.coding.code_generator.subprocess.run", fake_run)

        agent = CodeGeneratorAgent(
            {
                "tdd_docker_image": "repo0-codegen-tdd:latest",
                "tdd_pip_project_root": str(project_root),
                "tdd_pytest_timeout": 30,
                "tdd_pip_timeout": 40,
            }
        )
        rc, out = agent._run_pytest_in_docker(
            sandbox_root,
            "tests/test_demo.py",
            heuristic_pip_specs=["numpy", "pandas>=2"],
        )

        assert rc == 0
        assert out == "ok\n"
        inner = seen["cmd"][-1]
        assert "python -m venv --system-site-packages /tmp/.tdd_venv" in inner
        assert ". /tmp/.tdd_venv/bin/activate" in inner
        assert "PIP_ROOT_USER_ACTION=ignore" in inner
        assert "mkdir -p /tmp/.tdd_state" in inner
        assert "project_editable_ready" in inner
        assert "/tmp/.tdd_state/project_editable_ready" in inner
        assert "pip_numpy_ready" in inner
        assert "pip_pandas_2_ready" in inner
        assert "python -m pytest -p no:cacheprovider tests/test_demo.py -q --tb=short" in inner


def test_write_sandbox_context_sources_copies_known_dependency_files() -> None:
    with tempfile.TemporaryDirectory() as tmp_root, tempfile.TemporaryDirectory() as tmp_proj:
        sandbox_root = Path(tmp_root)
        project_root = Path(tmp_proj)
        dep_rel = "statsmodels/model_selection/design_matrix_data_validation.py"
        dep_src = project_root / dep_rel
        dep_src.parent.mkdir(parents=True, exist_ok=True)
        dep_src.write_text("VALUE = 1\n", encoding="utf-8")

        agent = CodeGeneratorAgent({"tdd_pip_project_root": str(project_root)})
        copied = agent._write_sandbox_context_sources(
            sandbox_root,
            "=== IMPLEMENTED COMPONENTS (Available for reuse) ===\n"
            "[Design-Matrix & Data Validation] (Parent: Input Validation)\n"
            f"  File: {dep_rel}\n",
        )

        assert copied == 1
        assert (sandbox_root / dep_rel).read_text(encoding="utf-8") == "VALUE = 1\n"
        assert (sandbox_root / "statsmodels" / "__init__.py").exists()
        assert (sandbox_root / "statsmodels" / "model_selection" / "__init__.py").exists()


def test_normalize_test_file_path_sanitizes_dotted_names() -> None:
    agent = CodeGeneratorAgent({})
    path = agent._normalize_test_file_path(
        "tests/test_bayesian_mixed_glm.model_spec.py",
        component_name="BayesianMixedGLM.ModelSpec",
    )

    assert path == "tests/test_bayesian_mixed_glm_model_spec.py"


def test_tdd_fix_loop_normalizes_dotted_test_file_paths(monkeypatch) -> None:
    seen = {}
    agent = CodeGeneratorAgent({"tdd_disable_docker": True, "tdd_max_fix_retries": 0})

    monkeypatch.setattr(agent, "_tdd_needs_pip_prepare", lambda: False)
    monkeypatch.setattr(agent, "_autofix_pair_before_pytest", lambda **kwargs: (kwargs["impl_body"], kwargs["test_body"]))
    monkeypatch.setattr(agent, "_write_sandbox_sources", lambda root, sources: None)
    monkeypatch.setattr(agent, "_write_sandbox_context_sources", lambda root, context: 0)

    def fake_run_pytest(root, rel_test, heuristic_pip_specs=None):
        seen["rel_test"] = rel_test
        return 0, "1 passed in 0.01s"

    monkeypatch.setattr(agent, "_run_pytest_in_sandbox", fake_run_pytest)

    _, _, meta = agent._tdd_fix_loop(
        rel_impl="statsmodels/bayesian_mixed/mod.py",
        impl_code="def ok():\n    return 1\n",
        rel_test="tests/test_bayesian_mixed_glm.postproc_and_diagnostics.py",
        test_code="def test_ok():\n    assert True\n",
        component_name="BayesianMixedGLM.PostprocAndDiagnostics",
    )

    assert seen["rel_test"] == "tests/test_bayesian_mixed_glm_postproc_and_diagnostics.py"
    assert meta["final_pytest_rc"] == 0


def test_placeholder_detection_ignores_try_except_internal_pass() -> None:
    agent = CodeGeneratorAgent({})
    code = """
def sample(x):
    try:
        return 1 / x
    except ZeroDivisionError:
        pass
    return None
"""
    issues = agent._find_python_placeholder_issues(code, "demo.py")
    assert issues == []


def test_placeholder_detection_ignores_tdd_in_comments() -> None:
    agent = CodeGeneratorAgent({})
    code = """
def sample():
    # early TDD phase compatibility note
    return 1
"""
    issues = agent._find_python_placeholder_issues(code, "demo.py")
    assert issues == []


def test_placeholder_detection_treats_protocol_stub_as_soft_warning() -> None:
    agent = CodeGeneratorAgent({})
    code = """
from typing import Protocol

class AdapterProtocol(Protocol):
    def call(self, value):
        ...
"""
    assert agent._find_python_placeholder_issues(code, "demo.py") == []
    warnings = agent._find_python_placeholder_warnings(code, "demo.py")
    assert warnings == ["demo.py:6 function 'call' has only `...`"]


def test_placeholder_detection_treats_concrete_test_pass_as_hard_issue() -> None:
    agent = CodeGeneratorAgent({})
    code = """
class FakePlugin:
    def register_plugin(self):
        pass
"""
    issues = agent._find_python_placeholder_issues(code, "tests/test_demo.py")
    assert issues == ["tests/test_demo.py:4 function 'register_plugin' has only `pass`"]


def test_save_generated_code_allows_soft_placeholder_warnings(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = CodeGeneratorAgent({})
        monkeypatch.setattr(agent, "_file_level_peer_repo_postcheck_and_repair", lambda **kwargs: {"passed": True})
        monkeypatch.setattr(
            agent,
            "_compile_postcheck_and_repair_file",
            lambda **kwargs: {"syntax_postcheck": {"passed": True}, "compile_postcheck": {"passed": True}},
        )
        code_result = {
            "component_name": "AdapterProtocol",
            "file_path": "statsmodels/demo_protocol.py",
            "code": (
                "from typing import Protocol\n\n"
                "class AdapterProtocol(Protocol):\n"
                "    def call(self, value):\n"
                "        ...\n"
            ),
            "tests": {},
            "documentation": "",
            "integration_notes": "",
            "language": "python",
            "skeleton_fill_tdd": {"final_pytest_rc": 0},
        }
        created = agent.save_generated_code(code_result, tmp)
        assert "code" in created
        assert code_result["save_succeeded"] is True
        assert code_result["placeholder_warnings"]["code"] == [
            "statsmodels/demo_protocol.py:5 function 'call' has only `...`"
        ]


def test_save_generated_code_removes_stale_non_normalized_test_file(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = CodeGeneratorAgent({})
        monkeypatch.setattr(agent, "_file_level_peer_repo_postcheck_and_repair", lambda **kwargs: {"passed": True})
        monkeypatch.setattr(
            agent,
            "_compile_postcheck_and_repair_file",
            lambda **kwargs: {"syntax_postcheck": {"passed": True}, "compile_postcheck": {"passed": True}},
        )
        stale = Path(tmp) / "tests/test_bayesian_mixed_glm.model_spec.py"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("def test_old():\n    assert True\n", encoding="utf-8")
        code_result = {
            "component_name": "BayesianMixedGLM.ModelSpec",
            "file_path": "statsmodels/bayesian_mixed/model_spec.py",
            "code": "def ok():\n    return 1\n",
            "tests": {
                "test_file_path": "tests/test_bayesian_mixed_glm.model_spec.py",
                "test_code": "def test_new():\n    assert True\n",
            },
            "documentation": "",
            "integration_notes": "",
            "language": "python",
            "skeleton_fill_tdd": {"final_pytest_rc": 0},
        }
        created = agent.save_generated_code(code_result, tmp)
        assert "test" in created
        assert not stale.exists()
        assert (Path(tmp) / "tests/test_bayesian_mixed_glm_model_spec.py").exists()


def test_save_generated_code_skips_hard_placeholder_test_but_keeps_code(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = CodeGeneratorAgent({})
        monkeypatch.setattr(agent, "_file_level_peer_repo_postcheck_and_repair", lambda **kwargs: {"passed": True})
        monkeypatch.setattr(
            agent,
            "_compile_postcheck_and_repair_file",
            lambda **kwargs: {"syntax_postcheck": {"passed": True}, "compile_postcheck": {"passed": True}},
        )
        code_result = {
            "component_name": "FormulaTermEngine",
            "file_path": "statsmodels/formula/term_engine.py",
            "code": "def ok():\n    return 1\n",
            "tests": {
                "test_file_path": "tests/test_formula_term_engine.py",
                "test_code": (
                    "class FakePlugin:\n"
                    "    def register_plugin(self):\n"
                    "        pass\n"
                ),
            },
            "documentation": "",
            "integration_notes": "",
            "language": "python",
            "skeleton_fill_tdd": {"final_pytest_rc": 1},
        }

        created = agent.save_generated_code(code_result, tmp)

        assert "code" in created
        assert "test" not in created
        assert code_result["save_succeeded"] is True
        assert code_result["test_save_succeeded"] is False
        assert "unresolved placeholders" in code_result["test_save_error"]
        assert code_result["skipped_test_files"][0]["path"] == "tests/test_formula_term_engine.py"
        assert (Path(tmp) / "statsmodels/formula/term_engine.py").exists()
        assert not (Path(tmp) / "tests/test_formula_term_engine.py").exists()


def test_local_validation_repairs_normalize_invalid_import_module_paths() -> None:
    agent = CodeGeneratorAgent({})
    impl = (
        "from statsmodels.tsa.api.py.arima import ARIMA\n\n"
        "def build():\n"
        "    return ARIMA\n"
    )
    test = (
        "from statsmodels.tsa.api.py.arima import ARIMA\n\n"
        "def test_build():\n"
        "    assert ARIMA is not None\n"
    )

    patched_impl, patched_test, meta = agent._apply_local_validation_repairs(
        component_name="TimeSeriesComponent",
        responsibilities=[],
        rel_impl="statsmodels/tsa/demo.py",
        impl_code=impl,
        rel_test="tests/test_demo.py",
        test_code=test,
        implemented_components_context="",
        stage="unit_test",
    )

    assert "statsmodels.tsa.api.py.arima" not in patched_impl
    assert "statsmodels.tsa.api.arima" in patched_impl
    assert "statsmodels.tsa.api.py.arima" not in patched_test
    assert meta["attempted"] is True
    assert meta["changed"] is True


def test_codegen_timing_report_records_events() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = CodeGeneratorAgent({}, output_dir=tmp)
        started = __import__("time").perf_counter()
        agent._record_codegen_timing_event(
            component_name="DemoComponent",
            stage="unit_stage",
            started_at_perf=started,
            meta={"x": 1},
        )
        events = json.loads((Path(tmp) / "codegen_timing_events.json").read_text(encoding="utf-8"))
        report = json.loads((Path(tmp) / "codegen_timing_report.json").read_text(encoding="utf-8"))
        assert len(events) == 1
        assert events[0]["component_name"] == "DemoComponent"
        assert events[0]["stage"] == "unit_stage"
        assert report["total_events"] == 1
        assert report["by_stage"]["unit_stage"]["events"] == 1
        assert report["by_component"]["DemoComponent"]["events"] == 1
