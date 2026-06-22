from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.coding.setup_py_agent import SetupPyAgent  # noqa: E402


def test_setup_py_agent_skips_tests_and_local_runtime_dependencies(tmp_path: Path):
    pkg = tmp_path / "scikit_learn"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import numpy\n", encoding="utf-8")
    (pkg / "core.py").write_text("import numpy\n", encoding="utf-8")

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_core.py").write_text(
        "import pytest\nimport statsmodels\nimport scikit_learn\n",
        encoding="utf-8",
    )

    agent = SetupPyAgent(api_config={}, output_dir=str(tmp_path))
    report = agent.run(
        project_root=tmp_path,
        setup_py_path=tmp_path / "generated_code" / "setup.py",
        package_name="scikit_learn",
        skip_llm=True,
        enable_postcheck=False,
    )

    assert report["install_requires"] == ["numpy"]


def test_setup_py_agent_sanitizes_llm_runtime_dependency_output(tmp_path: Path):
    pkg = tmp_path / "statsmodels"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import numpy\n", encoding="utf-8")
    (pkg / "core.py").write_text("import numpy\n", encoding="utf-8")

    class FakeLLM:
        def call_json(self, *_args, **_kwargs):
            return {
                "install_requires": ["numpy", "pytest", "src", "statsmodels"],
                "mapping_changes": [],
                "dropped_import_roots": [],
                "conflict_resolutions": [],
                "notes": "test",
            }

    agent = SetupPyAgent(api_config={}, output_dir=str(tmp_path))
    agent.llm_client = FakeLLM()
    report = agent.run(
        project_root=tmp_path,
        setup_py_path=tmp_path / "generated_code" / "setup.py",
        package_name="statsmodels",
        skip_llm=False,
        enable_postcheck=False,
    )

    assert report["install_requires"] == ["numpy"]
    setup_text = (tmp_path / "generated_code" / "setup.py").read_text(encoding="utf-8")
    assert "pytest" not in setup_text
    assert "'src'" not in setup_text
