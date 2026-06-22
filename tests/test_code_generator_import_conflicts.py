from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.coding.code_generator import CodeGeneratorAgent


def test_extract_repo_python_paths_from_postcheck_output_includes_conflict_modules_and_inits(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "sympy" / "symbolic_expression").mkdir(parents=True)
    (repo_root / "sympy" / "code_generation").mkdir(parents=True)
    (repo_root / "sympy" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "sympy" / "symbolic_expression" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "sympy" / "code_generation" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "sympy" / "symbolic_expression" / "metadata_extension_api.py").write_text("", encoding="utf-8")
    (repo_root / "sympy" / "code_generation" / "probabilistic_execution_and_uncertainty_engine.py").write_text("", encoding="utf-8")

    output = """
Traceback (most recent call last):
  File "/tmp/workspace/test_file.py", line 4, in <module>
    from sympy.symbolic_expression.expression_model_core import Expr
  File "/repo/sympy/symbolic_expression/__init__.py", line 6, in <module>
    from . import metadata_extension_api
  File "/repo/sympy/symbolic_expression/metadata_extension_api.py", line 52, in <module>
    from sympy.code_generation.extension_integration_runtime_orchestrator import (
  File "/repo/sympy/code_generation/__init__.py", line 9, in <module>
    from . import probabilistic_execution_and_uncertainty_engine
  File "/repo/sympy/code_generation/probabilistic_execution_and_uncertainty_engine.py", line 136, in <module>
    from sympy.symbolic_expression.metadata_extension_api import MetadataService
ImportError: cannot import name 'MetadataService' from partially initialized module 'sympy.symbolic_expression.metadata_extension_api' (most likely due to a circular import) (/repo/sympy/symbolic_expression/metadata_extension_api.py)
"""

    rel_paths = CodeGeneratorAgent._extract_repo_python_paths_from_postcheck_output(output, repo_root)

    assert "sympy/symbolic_expression/__init__.py" in rel_paths
    assert "sympy/code_generation/__init__.py" in rel_paths
    assert "sympy/symbolic_expression/metadata_extension_api.py" in rel_paths
    assert "sympy/code_generation/probabilistic_execution_and_uncertainty_engine.py" in rel_paths


def test_postcheck_saved_component_downgrades_on_import_conflict(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "sympy").mkdir(parents=True)
    (repo_root / "sympy" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "sympy" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")

    agent = CodeGeneratorAgent(api_config={})
    conflict_output = (
        "ImportError: cannot import name 'MetadataService' from partially initialized module "
        "'sympy.symbolic_expression.metadata_extension_api' (most likely due to a circular import)"
    )
    agent._run_saved_python_import_postcheck = lambda **kwargs: (False, conflict_output)  # type: ignore[method-assign]

    report = agent.postcheck_saved_component(
        code_result={"component_name": "Demo", "file_path": "sympy/demo.py"},
        repo_root=repo_root,
        created_files={},
        max_fix_attempts=0,
    )

    assert report["passed"] is False
    assert report["downgraded"] is True
    assert report["reason"] == "import_conflict"


def test_postcheck_package_modules_downgrades_on_import_conflict(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "sympy").mkdir(parents=True)
    (repo_root / "sympy" / "__init__.py").write_text("", encoding="utf-8")

    agent = CodeGeneratorAgent(api_config={})
    conflict_output = (
        "ImportError: cannot import name 'MetadataService' from partially initialized module "
        "'sympy.symbolic_expression.metadata_extension_api' (most likely due to a circular import)"
    )
    agent._run_saved_python_import_postcheck = lambda **kwargs: (False, conflict_output)  # type: ignore[method-assign]

    report = agent.postcheck_package_modules(
        package_modules=["sympy"],
        repo_root=repo_root,
        max_fix_attempts=0,
    )

    assert report["passed"] is False
    assert report["modules"][0]["passed"] is False
    assert report["modules"][0]["downgraded"] is True
    assert report["modules"][0]["reason"] == "import_conflict"


def test_hyphenated_repo_uses_importable_package_root() -> None:
    agent = CodeGeneratorAgent(api_config={"repo": "a-b"})

    assert agent._primary_python_package_root() == "a_b"
    assert agent.path_allowed_roots[0] == "a_b"
    assert CodeGeneratorAgent(api_config={"repo": "a-b", "path_allowed_roots": ["a-b"]}).path_allowed_roots == ["a_b"]
    assert agent.normalize_file_path("feature/demo.py") == "a_b/feature/demo.py"
    assert agent.normalize_file_path("a-b/feature/demo.py") == "a_b/feature/demo.py"
