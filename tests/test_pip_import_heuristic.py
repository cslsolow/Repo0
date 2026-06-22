"""Tests for shared pip/import heuristics (SetupPyAgent + TDD)."""

from __future__ import annotations

from agents.coding.pip_import_heuristic import (
    distribution_for_import_root,
    filter_third_party_import_roots,
    stdlib_names,
    third_party_roots_to_heuristic_rows,
    third_party_roots_to_pip_install_specs,
)


def test_distribution_sklearn() -> None:
    assert distribution_for_import_root("sklearn") == "scikit-learn"


def test_filter_matches_stdlib_and_local() -> None:
    ext, skip = filter_third_party_import_roots(
        ["os", "pandas", "mypkg"],
        local_top_level={"mypkg"},
        stdlib=stdlib_names(),
    )
    assert "pandas" in ext
    assert "os" in skip
    assert "mypkg" in skip


def test_heuristic_rows_align_with_pip_specs() -> None:
    third = ["yaml", "numpy"]
    rows = third_party_roots_to_heuristic_rows(third)
    specs = third_party_roots_to_pip_install_specs(third)
    dists = {r["suggested_distribution"] for r in rows}
    assert set(specs) == dists
