"""TDD pytest pip heuristics — re-exports shared logic from ``pip_import_heuristic``."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set

from .pip_import_heuristic import (
    collect_import_roots_from_sources,
    collect_transitive_project_import_roots,
    import_roots_to_pip_specs,
    missing_import_roots_from_pytest_log,
    sandbox_top_level_names,
    stdlib_names,
)

# Back-compat names for tests and older imports
collect_top_level_import_roots = collect_import_roots_from_sources
stdlib_top_level_names = stdlib_names


def specs_from_sources_and_sandbox(
    impl: str,
    test: str,
    sandbox_root: Path,
    rel_impl: str,
    rel_test: str,
) -> List[str]:
    roots = collect_import_roots_from_sources(impl, test)
    local = sandbox_top_level_names(sandbox_root, (rel_impl, rel_test))
    return import_roots_to_pip_specs(roots, stdlib=stdlib_names(), local=local)


def specs_from_missing_roots(missing: Iterable[str], *, local: Set[str]) -> List[str]:
    roots = {str(m).split(".")[0].strip() for m in missing if str(m).strip()}
    return import_roots_to_pip_specs(roots, stdlib=stdlib_names(), local=local)


def specs_from_project_import_closure(
    project_root: Path,
    rel_paths: Iterable[str],
) -> List[str]:
    local = sandbox_top_level_names(project_root, rel_paths)
    roots = collect_transitive_project_import_roots(project_root, rel_paths)
    return import_roots_to_pip_specs(roots, stdlib=stdlib_names(), local=local)
