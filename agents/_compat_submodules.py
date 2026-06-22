"""Register legacy ``agents.<name>`` module aliases (same objects as new locations).

Loading the ``agents`` package runs :func:`register_legacy_submodule_names` first
so ``from agents.llm_client import …`` and other **flat** legacy module paths
keep working without per-name ``.py`` stubs.

The three CLI entrypoints ``python -m agents.{rqmts_paser,graph_parser,pr_rqmts_paser}``
use small **subpackages** (``__init__.py`` + ``__main__.py``) because ``runpy`` /
``importlib`` cannot treat a plain ``sys.modules`` alias as a runnable main module.
"""

from __future__ import annotations

import importlib
import sys

_LEGACY_TO_TARGET: tuple[tuple[str, str], ...] = (
    ("agents.llm_client", "agents.infra.llm_client"),
    ("agents.dag", "agents.rqmts.dag"),
    ("agents.hierarchical_dag", "agents.rqmts.hierarchical_dag"),
    ("agents.graph", "agents.rqmts.graph"),
    ("agents.dag_evolution", "agents.rqmts.dag_evolution"),
    ("agents.hierarchical_evolution", "agents.rqmts.hierarchical_evolution"),
    ("agents.architect", "agents.cognitive.architect"),
    ("agents.planner", "agents.cognitive.planner"),
    ("agents.strategist", "agents.cognitive.strategist"),
    ("agents.generator", "agents.cognitive.generator"),
    ("agents.module_assignment", "agents.cognitive.module_assignment"),
    ("agents.module_planner", "agents.cognitive.module_planner"),
    ("agents.memory", "agents.cognitive.memory"),
    ("agents.code_generator", "agents.coding.code_generator"),
    ("agents.fix_agent", "agents.coding.fix_agent"),
    ("agents.patch_agent", "agents.coding.patch_agent"),
    ("agents.lint_fix_agent", "agents.coding.lint_fix_agent"),
    ("agents.static_preflight", "agents.coding.static_preflight"),
    ("agents.setup_py_agent", "agents.coding.setup_py_agent"),
    ("agents.test_rewrite_agent", "agents.coding.test_rewrite_agent"),
    ("agents.requirement_merge_agent", "agents.merge.requirement_merge_agent"),
    ("agents.component_merge_agent", "agents.merge.component_merge_agent"),
    ("agents.dependency_graph_agent", "agents.analysis.dependency_graph_agent"),
    ("agents.localization_pipeline_agent", "agents.analysis.localization_pipeline_agent"),
    ("agents.depth_action_builder", "agents.analysis.depth_action_builder"),
)

_registered = False


def register_legacy_submodule_names() -> None:
    global _registered
    if _registered:
        return
    for legacy, target in _LEGACY_TO_TARGET:
        if legacy not in sys.modules:
            sys.modules[legacy] = importlib.import_module(target)
    _registered = True
