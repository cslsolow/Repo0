"""Tests for TDD pip heuristic (import scan + pytest log parsing)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agents.coding.tdd_pip_heuristic import (
    collect_top_level_import_roots,
    missing_import_roots_from_pytest_log,
    sandbox_top_level_names,
    specs_from_project_import_closure,
    specs_from_missing_roots,
    specs_from_sources_and_sandbox,
)


def test_collect_skips_relative_imports() -> None:
    src = """
from .mypkg import x
from other import y
"""
    roots = collect_top_level_import_roots(src)
    assert roots == {"other"}


def test_collect_import_and_from() -> None:
    src = "import numpy as np\nfrom scipy.stats import norm\n"
    roots = collect_top_level_import_roots(src)
    assert roots == {"numpy", "scipy"}


def test_specs_skip_stdlib_and_local_package() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "mypkg").mkdir()
        (root / "mypkg" / "__init__.py").write_text("", encoding="utf-8")
        impl = "import json\nimport mypkg\nimport pandas\n"
        test = "import os\n"
        rel_impl = "mypkg/impl.py"
        rel_test = "tests/test_x.py"
        specs = specs_from_sources_and_sandbox(impl, test, root, rel_impl, rel_test)
        assert "pandas" in specs
        assert "json" not in specs
        assert "os" not in specs
        assert "mypkg" not in specs


def test_pypy_alias_sklearn() -> None:
    impl = "import sklearn"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        specs = specs_from_sources_and_sandbox(impl, "", root, "a.py", "b.py")
        assert any("scikit" in s for s in specs)


def test_missing_modules_from_pytest_log() -> None:
    log = """
E   ModuleNotFoundError: No module named 'foo.bar'
E   ImportError: No module named 'baz'
"""
    got = missing_import_roots_from_pytest_log(log)
    assert got == ["foo", "baz"]


def test_specs_from_missing_skips_local() -> None:
    miss = ["pandas", "mypkg"]
    local = {"mypkg", "tests"}
    specs = specs_from_missing_roots(miss, local=local)
    assert specs == ["pandas"]


def test_sandbox_top_level_from_rel_paths_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        names = sandbox_top_level_names(root, ("pkg/mod.py", "other.py"))
        assert {"pkg", "other"}.issubset(names)


def test_project_import_closure_collects_transitive_third_party() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "statsmodels" / "formula_design").mkdir(parents=True)
        (root / "tests").mkdir()
        (root / "statsmodels" / "__init__.py").write_text("", encoding="utf-8")
        (root / "statsmodels" / "formula_design" / "__init__.py").write_text("", encoding="utf-8")
        (root / "statsmodels" / "formula_design" / "data_integration_and_cache_manager.py").write_text(
            "import pandas as pd\n",
            encoding="utf-8",
        )
        (root / "statsmodels" / "core_impl.py").write_text(
            "from statsmodels.formula_design.data_integration_and_cache_manager import pd\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_core_impl.py").write_text(
            "from statsmodels.core_impl import pd\n",
            encoding="utf-8",
        )
        specs = specs_from_project_import_closure(
            root,
            ("statsmodels/core_impl.py", "tests/test_core_impl.py"),
        )
        assert "pandas" in specs
