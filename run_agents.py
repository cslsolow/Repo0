"""Simple orchestrator for the Repo0 multi-agent prototype."""

from __future__ import annotations

import atexit
import argparse
import ast
import hashlib
import importlib
import json
import logging
import os
import py_compile
import re
import shlex
import shutil
import subprocess
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Any, Dict, Set

from package_api_plan_builder import (
    build_canonical_package_grouping as _external_build_canonical_package_grouping,
    build_package_api_plan as _external_build_package_api_plan,
    derive_component_export_symbols as _external_derive_component_export_symbols,
)
from agents import (
    ArchitectAgent,
    MemoryAgent,
    PlannerAgent,
    RequirementDAG,
    RequirementNode,
    DAGEvolutionAgent,
    StrategistAgent,
    DependencyGraphAgent,
    ModuleAssignmentAgent,
    ModulePlanningAgent,
    RequirementMergeAgent,
    ComponentMergeAgent,
    ComponentSplitAgent,
    SetupPyAgent,
)
from agents.rqmts_paser import generate_and_save_one_requirements
from agents.graph_parser import generate_and_save_edges
from agents.coding.structured_contracts import find_structured_contract_issues
from agents.package_root import normalize_python_package_root
from agents.analysis.gap_addition import (
    GapAdditionDecision,
    GapAdditionJudge,
    apply_gap_addition_decision,
    propose_gap_candidate_for_parent,
    run_local_gap_cleanup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Repo0 agent pipeline")
    parser.add_argument(
        "--requirements-file",
        type=Path,
        required=True,
        help="Path to a text file with high-level requirements.",
    )
    parser.add_argument(
        "--req-path",
        type=Path,
        default=None,
        help="Optional path to a README.req file to override the repository default.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Name of the repository under the repos/ directory to focus on.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).parent,
        help="Workspace root (defaults to the script directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory where agent artifacts will be stored.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://aihubmix.com/v1",
        help="Base URL for API calls.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("API_KEY", ""),
        help="API key for external service calls. Defaults to API_KEY from the environment.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="",
        help="Optional reasoning_effort to pass through to the LLM API.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DeepSeek-V3.2-Exp",
        help="Model name to use for API calls.",
    )
    parser.add_argument(
        "--enable-output-token-routing",
        action="store_true",
        help="Use a short-output model first, then rerun clipped responses with the long-output model.",
    )
    parser.add_argument(
        "--short-output-model",
        type=str,
        default="deepseek-chat",
        help="Model used first when --enable-output-token-routing is set.",
    )
    parser.add_argument(
        "--short-output-max-tokens",
        type=int,
        default=8192,
        help="max_tokens for the short-output model.",
    )
    parser.add_argument(
        "--long-output-model",
        type=str,
        default="",
        help="Long-output model for requests that cannot fit the short-output cap. Defaults to --model.",
    )
    parser.add_argument(
        "--long-output-max-tokens",
        type=int,
        default=0,
        help="Long-output max_tokens. 0 means use the original request max_tokens.",
    )
    parser.add_argument(
        "--output-token-rerun-margin",
        type=int,
        default=32,
        help="Rerun with the long model when completion tokens are within this margin of the short cap.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of all artifacts, ignoring existing files.",
    )
    parser.add_argument(
        "--resume-rerun-retained-tdd-failures",
        action="store_true",
        help=(
            "When resuming, rerun components retained after TDD failure. "
            "Default: reuse retained artifacts if their files pass syntax/compile/import checks."
        ),
    )
    parser.add_argument(
        "--retry-empty-generated-components",
        action="store_true",
        help=(
            "When resuming, regenerate only components whose generated_files.json entry has empty files. "
            "Other existing generated entries are kept as-is."
        ),
    )
    parser.add_argument(
        "--evolve-requirements-file",
        type=Path,
        default=None,
        help="Optional path to a JSON or text file with new requirements for DAG evolution.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--postcheck-max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for conflict-free component import postchecks within a parent (default: 4).",
    )
    parser.add_argument(
        "--use-processes",
        action="store_true",
        help="Use multiprocessing instead of multithreading (better for CPU-bound tasks).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--session-loop",
        action="store_true",
        help="Run an interactive multi-round session (incremental evolve or restart from scratch each round).",
    )
    parser.add_argument(
        "--session-max-rounds",
        type=int,
        default=0,
        help="Maximum rounds in session loop; 0 means unlimited.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the requirement merge step and use original requirements for DAG building.",
    )
    parser.add_argument(
        "--parent-codegen-dag-source",
        type=str,
        choices=("requirement", "dependency", "none"),
        default="dependency",
        help="Choose the parent-level DAG source for codegen ordering/context: requirement, dependency, or none.",
    )
    parser.add_argument(
        "--disable-decomposition",
        action="store_true",
        help="Disable requirement DAG decomposition and plan directly on the original requirement DAG.",
    )
    parser.add_argument(
        "--disable-graph-module",
        action="store_true",
        help="Disable the graph module entirely and use graph-free planning/grouping/scheduling.",
    )
    parser.add_argument(
        "--disable-dependency-graph",
        action="store_true",
        help="Disable implementation dependency graph generation/usage while keeping requirement-graph planning enabled.",
    )
    parser.add_argument(
        "--no-graph-seed",
        type=int,
        default=42,
        help="Seed used to pseudo-randomize parent order when the graph module is disabled.",
    )
    parser.add_argument(
        "--disable-layout-schema",
        action="store_true",
        help="Disable strict file-layout schema enforcement during planning/generation.",
    )
    parser.add_argument(
        "--layout-root",
        type=str,
        default=None,
        help="Optional package root for generated Python modules (default: --repo).",
    )
    parser.add_argument(
        "--layout-whitelist",
        type=str,
        default="",
        help="Comma-separated top-level directory whitelist. Defaults to '<repo>,docs,tests,tools,examples'.",
    )
    parser.add_argument(
        "--layout-alias-map",
        type=str,
        default="",
        help="Optional alias map for first subpackage segment, e.g. 'timeseries=tsa,toolkit=tools'.",
    )
    parser.add_argument(
        "--disable-component-merge",
        action="store_true",
        help="Disable LLM-based component merge stage after architecture generation.",
    )
    parser.add_argument(
        "--enable-cross-requirement-component-merge",
        action="store_true",
        help="Allow component merge to deduplicate components across different requirement parents.",
    )
    parser.add_argument(
        "--component-merge-input",
        type=Path,
        default=None,
        help=(
            "Optional architectures JSON snapshot to use as the fixed input for the first "
            "component merge stage. This forces component merge to be recomputed from the snapshot."
        ),
    )
    parser.add_argument(
        "--disable-structure-refinement",
        action="store_true",
        help="Disable pre-codegen structure refinement, including component merge and action-guided split.",
    )
    parser.add_argument(
        "--disable-strategist",
        action="store_true",
        help="Disable strategist action selection and run downstream planning with empty action hints.",
    )
    parser.add_argument(
        "--enable-component-metric-actions",
        action="store_true",
        help=(
            "Augment strategist actions with conservative component-level metric triggers. "
            "Default off preserves baseline behavior."
        ),
    )
    parser.add_argument(
        "--component-metric-split-cohesion-threshold",
        type=float,
        default=2.0 / 3.0,
        help="Cohesion threshold used by metric split augmentation (default: 2/3).",
    )
    parser.add_argument(
        "--component-metric-split-min-subrequirements",
        type=int,
        default=3,
        help="Minimum served subrequirements before metric split may upgrade a save action (default: 3).",
    )
    parser.add_argument(
        "--component-split-min-confidence",
        type=float,
        default=0.70,
        help="Minimum confidence required to accept an LLM-proposed split (default: 0.70).",
    )
    parser.add_argument(
        "--enable-component-metric-merge-judge",
        action="store_true",
        help=(
            "Allow conservative metric merge candidates to be reviewed by an LLM judge. "
            "Requires --enable-component-metric-actions and an API key."
        ),
    )
    parser.add_argument(
        "--component-metric-merge-max-small-subrequirements",
        type=int,
        default=1,
        help="Maximum served subrequirements for the smaller side of a metric merge candidate (default: 1).",
    )
    parser.add_argument(
        "--tdd-revise-failure-threshold",
        type=int,
        default=2,
        help="Number of repeated retained TDD failures before emitting a revise action report (default: 2).",
    )
    parser.add_argument(
        "--action-refinement-rounds",
        type=int,
        default=1,
        help=(
            "Number of strategist/action-guided component refinement feedback rounds before module planning "
            "(default: 1, matching the original pipeline)."
        ),
    )
    parser.add_argument(
        "--action-refinement-stop-on-stable",
        action="store_true",
        help="Stop multi-round action refinement early when a round applies no merge/split changes.",
    )
    parser.add_argument(
        "--action-refinement-save-stops-component",
        action="store_true",
        help="Treat strategist action 'save' as a per-component stop signal in multi-round action refinement.",
    )
    parser.add_argument(
        "--enable-gap-add-actions",
        action="store_true",
        help="Run gap-driven add actions after structural refinement and before module planning.",
    )
    parser.add_argument(
        "--gap-add-proposal-threshold",
        type=float,
        default=0.55,
        help="Minimum heuristic proposer score required before judging a gap-add candidate.",
    )
    parser.add_argument(
        "--gap-add-component-threshold",
        type=float,
        default=0.74,
        help="Acceptance threshold for add_component decisions.",
    )
    parser.add_argument(
        "--gap-add-requirement-threshold",
        type=float,
        default=0.82,
        help="Acceptance threshold for add_requirement_and_component decisions.",
    )
    parser.add_argument(
        "--stop-after-architecture-refinement",
        action="store_true",
        help="Persist refined architectures and stop before module planning and code generation.",
    )
    parser.add_argument(
        "--enable-component-merge-embedding-analysis",
        action="store_true",
        help="Enable optional embedding score/clustering analysis for component merge diagnostics.",
    )
    parser.add_argument(
        "--component-merge-embedding-threshold",
        type=float,
        default=0.78,
        help="Weighted score threshold for embedding diagnostic merge candidate edges.",
    )
    parser.add_argument(
        "--component-merge-dominance-gap",
        type=float,
        default=0.12,
        help="Score gap used to break transitive chain merges in embedding diagnostics.",
    )
    parser.add_argument(
        "--component-merge-admission-mode",
        type=str,
        choices=("strict", "llm_review_relaxed"),
        default="strict",
        help="Validation mode for applying component merge groups. Default strict preserves existing behavior.",
    )
    parser.add_argument(
        "--component-merge-relaxed-best",
        type=float,
        default=0.30,
        help="Relaxed best-pair threshold used only by llm_review_relaxed admission.",
    )
    parser.add_argument(
        "--component-merge-relaxed-avg",
        type=float,
        default=0.26,
        help="Relaxed average-pair threshold used only by llm_review_relaxed admission.",
    )
    parser.add_argument(
        "--component-merge-relaxed-min-pair",
        type=float,
        default=0.20,
        help="Relaxed worst-pair threshold used only by llm_review_relaxed admission.",
    )
    parser.add_argument(
        "--component-merge-relaxed-dominance-gap",
        type=float,
        default=0.28,
        help="Relaxed chain dominance gap used only by llm_review_relaxed admission.",
    )
    parser.add_argument(
        "--component-merge-name-weight",
        type=float,
        default=0.5,
        help="Name similarity weight in embedding diagnostics.",
    )
    parser.add_argument(
        "--component-merge-resp-weight",
        type=float,
        default=0.35,
        help="Responsibility similarity weight in embedding diagnostics.",
    )
    parser.add_argument(
        "--component-merge-subreq-weight",
        type=float,
        default=0.15,
        help="Serves-subrequirements overlap weight in embedding diagnostics.",
    )
    parser.add_argument(
        "--setup-py-use-llm",
        action="store_true",
        help="Use LLM in SetupPyAgent to correct install_requires mappings (requires --api-key).",
    )
    parser.add_argument(
        "--setup-py-postcheck",
        action="store_true",
        help="After generating setup.py, run venv pip-install + import presence checks.",
    )
    parser.add_argument(
        "--setup-py-path",
        type=Path,
        default=None,
        help="Optional path for generated setup.py (default: <output>/generated_code/setup.py).",
    )
    parser.add_argument(
        "--setup-py-package-name",
        type=str,
        default=None,
        help="Optional python package name to put into setup.py name= (default: derived from layout_root).",
    )
    parser.add_argument(
        "--codegen-tdd-max-fix-retries",
        type=int,
        default=3,
        help="Max PatchAgent rounds after pytest failure in Python TDD codegen (default: 3).",
    )
    parser.add_argument(
        "--codegen-tdd-disable-docker",
        action="store_true",
        help=(
            "Run TDD pytest on the host Python. Default: Docker image repo0-codegen-tdd:latest "
            "(auto-built once if missing; see docker/codegen-tdd/)."
        ),
    )
    parser.add_argument(
        "--codegen-tdd-docker-network-host",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "TDD ``docker build`` (auto image) and ``docker run`` (pytest) use ``--network host`` when on "
            "(default: on). Disable with --no-codegen-tdd-docker-network-host if unsupported."
        ),
    )
    parser.add_argument(
        "--codegen-tdd-pip-timeout",
        type=int,
        default=600,
        help="Timeout seconds for each pip subprocess during TDD dependency prep (default: 600).",
    )
    parser.add_argument(
        "--codegen-tdd-missing-module-pip-retries",
        type=int,
        default=3,
        help=(
            "Max pytest re-runs after ModuleNotFoundError-driven pip installs per patch round (default: 3)."
        ),
    )
    parser.add_argument(
        "--import-postcheck-max-fix-attempts",
        type=int,
        default=10,
        help="Max fix attempts for file-level import postchecks (default: 10).",
    )
    parser.add_argument(
        "--package-postcheck-max-fix-attempts",
        type=int,
        default=10,
        help="Max fix attempts for package/top-level import postchecks (default: 10).",
    )
    parser.add_argument(
        "--init-export-lazy-imports",
        action="store_true",
        help="Generate lazy package __init__.py exports instead of eager top-level imports.",
    )
    return parser.parse_args()


def _to_snake_case(name: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or "").strip())
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = text.strip("_").lower()
    return text or "unnamed_component"


def _to_pascal_case(name: str) -> str:
    snake = _to_snake_case(name)
    parts = [part for part in snake.split("_") if part]
    if not parts:
        return "UnnamedComponent"
    return "".join(part.capitalize() for part in parts)


def _parse_layout_alias_map(raw_value: str) -> Dict[str, str]:
    text = str(raw_value or "").strip()
    if not text:
        return {}
    # Accept JSON object or CSV style "a=b,c=d".
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return {
                    _to_snake_case(str(k)): _to_snake_case(str(v))
                    for k, v in payload.items()
                    if str(k).strip() and str(v).strip()
                }
        except Exception:
            logging.warning("Invalid --layout-alias-map JSON, fallback to CSV parsing.")

    mapping: Dict[str, str] = {}
    for pair in text.split(","):
        part = pair.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        src, dst = part.split("=", 1)
        src_norm = _to_snake_case(src)
        dst_norm = _to_snake_case(dst)
        if src_norm and dst_norm:
            mapping[src_norm] = dst_norm
    return mapping


def _build_layout_policy(args: argparse.Namespace) -> Dict[str, Any]:
    layout_root = normalize_python_package_root(args.layout_root or args.repo)
    if not layout_root:
        layout_root = "src"

    user_whitelist = [
        item.strip().strip("/")
        for item in str(args.layout_whitelist or "").split(",")
        if item.strip()
    ]
    default_whitelist = [layout_root, "docs", "tests", "tools", "examples"]
    whitelist = user_whitelist or default_whitelist
    whitelist = list(dict.fromkeys(whitelist))

    alias_map = _parse_layout_alias_map(args.layout_alias_map)
    return {
        "enabled": not bool(args.disable_layout_schema),
        "layout_root": layout_root,
        "top_whitelist": whitelist,
        "alias_map": alias_map,
        "canonical_packages": [],
        "default_subpackage": "core",
        "component_package_index": {},
        "component_package_path_index": {},
    }


def _canonical_subpackage_candidates(policy: Dict[str, Any]) -> List[str]:
    raw = policy.get("canonical_packages", [])
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        token = _to_snake_case(str(item or ""))
        if token and token not in out:
            out.append(token)
    return out


def _best_canonical_subpackage(token: str, candidates: List[str]) -> str:
    token_norm = _to_snake_case(token)
    if not token_norm or not candidates:
        return ""
    if token_norm in candidates:
        return token_norm
    token_parts = set(token_norm.split("_"))
    best_name = ""
    best_score = 0.0
    for cand in candidates:
        cand_parts = set(cand.split("_"))
        if not cand_parts:
            continue
        overlap = len(token_parts & cand_parts) / len(token_parts | cand_parts)
        if overlap > best_score:
            best_score = overlap
            best_name = cand
    return best_name


def _normalize_layout_file_path(
    raw_path: str,
    policy: Dict[str, Any],
    *,
    fallback_rel_path: str | None = None,
) -> str:
    layout_root = str(policy.get("layout_root") or "src")
    whitelist = set(policy.get("top_whitelist") or [layout_root, "docs", "tests", "tools", "examples"])
    alias_map = policy.get("alias_map") or {}
    canonical_packages = _canonical_subpackage_candidates(policy)
    default_subpackage = str(policy.get("default_subpackage") or "core").strip() or "core"
    default_subpackage = _to_snake_case(default_subpackage)

    path = str(raw_path or "").strip().replace("\\", "/")
    path = path.lstrip("./").lstrip("/")
    if not path:
        path = str(fallback_rel_path or f"{layout_root}/generated/unnamed_component.py")
    if not path.endswith(".py"):
        path = f"{path}.py"

    parts = [part for part in path.split("/") if part and part != "."]
    if not parts:
        parts = [layout_root, "generated", "unnamed_component.py"]
    if normalize_python_package_root(parts[0], default="") == layout_root:
        parts[0] = layout_root

    if parts[0] not in whitelist:
        parts = [layout_root] + parts

    if len(parts) >= 2 and parts[0] == layout_root and parts[1] == layout_root:
        parts = parts[1:]

    if len(parts) >= 3 and parts[0] == layout_root:
        mapped = alias_map.get(parts[1])
        if mapped:
            parts[1] = mapped

    if parts and parts[0] == layout_root:
        if len(parts) == 2 and parts[1].endswith(".py"):
            parts = [layout_root, default_subpackage, parts[1]]
        if len(parts) >= 3 and canonical_packages:
            cand = _to_snake_case(parts[1])
            if cand not in canonical_packages:
                mapped = _best_canonical_subpackage(cand, canonical_packages)
                parts[1] = mapped or default_subpackage

    return "/".join(parts)


def _select_layout_subpackage(
    component: Dict[str, Any],
    requirement: Dict[str, Any],
    policy: Dict[str, Any],
) -> str:
    alias_map = policy.get("alias_map") or {}
    component_name = str(component.get("name", "")).strip()
    req_name = str(requirement.get("name", "")).strip()
    package_path_index = policy.get("component_package_path_index") or {}
    if req_name and component_name and isinstance(package_path_index, dict):
        direct_path = str(package_path_index.get(f"{req_name}::{component_name}", "")).strip().strip("/")
        if direct_path:
            return direct_path
    package_index = policy.get("component_package_index") or {}
    if req_name and component_name and isinstance(package_index, dict):
        direct = str(package_index.get(f"{req_name}::{component_name}", "")).strip()
        if direct:
            return _to_snake_case(direct)

    text_parts: List[str] = [
        req_name,
        str(requirement.get("description", "")),
        component_name,
    ]
    responsibilities = component.get("responsibilities", [])
    if isinstance(responsibilities, list):
        text_parts.extend(str(item) for item in responsibilities)
    bag = " ".join(text_parts).lower()

    for key in sorted(alias_map.keys(), key=len, reverse=True):
        if key in bag:
            return str(alias_map[key])

    canonical_packages = _canonical_subpackage_candidates(policy)
    if canonical_packages:
        req_slug = _to_snake_case(req_name)
        req_tokens = [tok for tok in req_slug.split("_") if tok]
        for tok in req_tokens:
            if tok in canonical_packages:
                return tok
        best = _best_canonical_subpackage(req_slug, canonical_packages)
        if best:
            return best

    # Generic fallback: derive a stable namespace from requirement name first.
    requirement_slug = _to_snake_case(str(requirement.get("name", "")))
    req_tokens = [token for token in requirement_slug.split("_") if token]
    stopwords = {
        "feature", "features", "task", "tasks", "requirement", "requirements",
        "component", "components", "module", "modules", "system", "implementation",
        "support", "manager", "engine", "service", "layer",
    }
    informative = [tok for tok in req_tokens if len(tok) >= 3 and tok not in stopwords]
    if informative:
        chosen = "_".join(informative[:2])
        if canonical_packages:
            mapped = _best_canonical_subpackage(chosen, canonical_packages)
            if mapped:
                return mapped
        return chosen

    component_slug = _to_snake_case(str(component.get("name", "")))
    comp_tokens = [tok for tok in component_slug.split("_") if len(tok) >= 3 and tok not in stopwords]
    if comp_tokens:
        chosen = "_".join(comp_tokens[:2])
        if canonical_packages:
            mapped = _best_canonical_subpackage(chosen, canonical_packages)
            if mapped:
                return mapped
        return chosen
    if canonical_packages:
        return canonical_packages[0]
    return str(policy.get("default_subpackage") or "core")


def _build_component_file_plan(
    architecture: Dict[str, Any],
    requirement: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, str]:
    components = architecture.get("components", []) if isinstance(architecture, dict) else []
    if not isinstance(components, list):
        return {}

    layout_root = str(policy.get("layout_root") or "src")
    plan: Dict[str, str] = {}
    for component in components:
        if not isinstance(component, dict):
            continue
        component_name = str(component.get("name", "")).strip()
        if not component_name:
            continue
        subpackage = _select_layout_subpackage(component, requirement, policy)
        snake_name = _to_snake_case(component_name)
        rel_path = f"{layout_root}/{subpackage}/{snake_name}.py"
        normalized = _normalize_layout_file_path(rel_path, policy, fallback_rel_path=rel_path)
        plan[component_name] = normalized
    return plan


def _derive_component_export_symbols(
    component_name: str,
    responsibilities: Any,
    planned_file_path: str,
) -> List[str]:
    return _external_derive_component_export_symbols(
        component_name=component_name,
        responsibilities=responsibilities,
        planned_file_path=planned_file_path,
    )


def _build_package_api_plan(
    architectures: List[dict],
    layout_policy: Dict[str, Any],
) -> Dict[str, Any]:
    return _external_build_package_api_plan(
        architectures=architectures,
        layout_policy=layout_policy,
        build_component_file_plan=_build_component_file_plan,
    )


def _build_canonical_package_grouping(
    architectures: List[dict],
    layout_policy: Dict[str, Any],
) -> Dict[str, Any]:
    return _external_build_canonical_package_grouping(
        architectures=architectures,
        layout_policy=layout_policy,
    )


def _is_oov_layout_path(rel_path: str, policy: Dict[str, Any]) -> bool:
    path = str(rel_path or "").strip().replace("\\", "/").lstrip("./").lstrip("/")
    if not path:
        return True
    parts = [p for p in path.split("/") if p and p != "."]
    if not parts:
        return True
    layout_root = str(policy.get("layout_root") or "").strip().strip("/")
    if not layout_root:
        return True
    if parts[0] != layout_root:
        whitelist = set(policy.get("top_whitelist") or [layout_root, "docs", "tests", "tools", "examples"])
        return parts[0] not in whitelist
    canonical_packages = _canonical_subpackage_candidates(policy)
    if not canonical_packages:
        return False
    if len(parts) < 3:
        return True
    return _to_snake_case(parts[1]) not in canonical_packages


def _default_component_rel_path(component_name: str, policy: Dict[str, Any]) -> str:
    layout_root = str(policy.get("layout_root") or "src").strip().strip("/") or "src"
    default_pkg = str(policy.get("default_subpackage") or "core").strip() or "core"
    default_pkg = _to_snake_case(default_pkg)
    return f"{layout_root}/{default_pkg}/{_to_snake_case(component_name)}.py"


def _enforce_layout_with_oov_retry(
    *,
    code_generator: Any,
    code_result: Dict[str, Any],
    component: Dict[str, Any],
    unified_task: Dict[str, Any],
    architecture: Dict[str, Any],
    implemented_context: str,
    layout_policy: Dict[str, Any],
    planned_rel_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Enforce canonical layout; retry once when path falls outside allowed canonical packages."""
    layout_meta = {
        "oov_detected": False,
        "retried": False,
        "forced_fallback": False,
        "final_path": "",
    }
    if not layout_policy.get("enabled"):
        layout_meta["final_path"] = str(code_result.get("file_path", ""))
        return code_result, layout_meta

    component_name = str(component.get("name", "")).strip() or str(code_result.get("component_name", "component")).strip()
    fallback_path = _default_component_rel_path(component_name or "component", layout_policy)
    preferred_path = planned_rel_path or str(code_result.get("file_path", "")).strip() or fallback_path
    normalized_preferred = _normalize_layout_file_path(
        preferred_path,
        layout_policy,
        fallback_rel_path=fallback_path,
    )
    code_result["file_path"] = normalized_preferred

    if not _is_oov_layout_path(code_result.get("file_path", ""), layout_policy):
        layout_meta["final_path"] = str(code_result.get("file_path", ""))
        return code_result, layout_meta

    layout_meta["oov_detected"] = True
    logging.warning(
        "OOV package path detected for component '%s': %s. Retrying with canonical fallback path.",
        component_name,
        code_result.get("file_path", ""),
    )

    retry_path = normalized_preferred
    if _is_oov_layout_path(retry_path, layout_policy):
        retry_path = _normalize_layout_file_path(
            fallback_path,
            layout_policy,
            fallback_rel_path=fallback_path,
        )
    retry_context = (
        f"{implemented_context}\n\n"
        "=== STRICT LAYOUT RETRY ===\n"
        f"You MUST generate this component at exact path: {retry_path}\n"
        "Do not invent a new package root or subpackage."
    )
    try:
        layout_meta["retried"] = True
        retry_result = code_generator.generate_code(
            component,
            unified_task,
            architecture,
            language="python",
            implemented_components_context=retry_context,
            planned_file_path=retry_path,
        )
        retry_result["file_path"] = _normalize_layout_file_path(
            retry_path or retry_result.get("file_path", ""),
            layout_policy,
            fallback_rel_path=retry_path or fallback_path,
        )
        code_result = retry_result
    except Exception as exc:
        logging.warning(
            "Layout retry failed for component '%s': %s. Falling back to canonical path.",
            component_name,
            exc,
        )

    if _is_oov_layout_path(code_result.get("file_path", ""), layout_policy):
        layout_meta["forced_fallback"] = True
        forced_path = _normalize_layout_file_path(
            fallback_path,
            layout_policy,
            fallback_rel_path=fallback_path,
        )
        code_result["file_path"] = forced_path
        logging.warning(
            "Forced canonical fallback path for component '%s': %s",
            component_name,
            forced_path,
        )

    layout_meta["final_path"] = str(code_result.get("file_path", ""))
    return code_result, layout_meta


def _module_from_relative_py_path(rel_path: str) -> str:
    path = str(rel_path or "").strip().replace("\\", "/").lstrip("./").lstrip("/")
    if not path.endswith(".py"):
        return ""
    module = path[:-3]
    if module.endswith("/__init__"):
        module = module[: -len("/__init__")]
    module = module.strip("/")
    return module.replace("/", ".")


def _discover_top_level_package_modules(generated_root: Path) -> List[str]:
    modules: List[str] = []
    if not generated_root.is_dir():
        return modules
    for child in sorted(generated_root.iterdir()):
        if child.is_dir() and (child / "__init__.py").is_file():
            modules.append(child.name)
    return modules


def _ensure_package_inits(code_file: Path, generated_root: Path, layout_root: str) -> List[str]:
    created: List[str] = []
    try:
        rel = code_file.resolve().relative_to(generated_root.resolve())
    except Exception:
        return created

    rel_parts = list(rel.parts)
    if not rel_parts or rel_parts[0] != layout_root:
        return created

    current = generated_root / layout_root
    init_file = current / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")
        created.append(str(init_file))

    for part in rel_parts[1:-1]:
        current = current / part
        if not current.exists() or not current.is_dir():
            break
        init_file = current / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
            created.append(str(init_file))
    return created


_INIT_EXPORT_MARKER = "# Auto-generated by Repo0 init-export stage."


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_string_list_literal(node: ast.AST) -> List[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return []
    values: List[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            values.append(elt.value.strip())
    return _dedupe_keep_order(values)


def _extract_module_export_symbols(module_file: Path) -> Dict[str, Any]:
    text = module_file.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(module_file))

    public_symbols: List[str] = []
    explicit_symbols: List[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                public_symbols.append(node.name)
            continue

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    explicit_symbols = _extract_string_list_literal(node.value)
                    break
            continue

        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "__all__":
                explicit_symbols = _extract_string_list_literal(node.value) if node.value else []

    public_symbols = _dedupe_keep_order(public_symbols)
    explicit_symbols = _dedupe_keep_order(explicit_symbols)
    return {
        "public_symbols": public_symbols,
        "explicit_symbols": explicit_symbols,
    }


def _extract_declared_symbols_from_generated_entry(entry: Dict[str, Any]) -> Set[str]:
    candidates: Set[str] = set()
    if not isinstance(entry, dict):
        return candidates

    explicit_symbols = entry.get("component_export_symbols", [])
    if isinstance(explicit_symbols, list):
        for symbol in explicit_symbols:
            token = str(symbol or "").strip()
            if token:
                candidates.add(token)

    component_name = str(entry.get("component", "")).strip()
    if component_name:
        candidates.add(_to_pascal_case(component_name))
        candidates.add(_to_snake_case(component_name))

    planned_path = str(entry.get("planned_file_path", "")).strip().replace("\\", "/")
    if planned_path:
        stem = Path(planned_path).stem
        if stem and stem != "__init__":
            candidates.add(stem)
            candidates.add(_to_pascal_case(stem))

    responsibilities = entry.get("component_responsibilities", [])
    if isinstance(responsibilities, list):
        for resp in responsibilities:
            text = str(resp or "")
            # Symbols that are already code-like from planning/responsibility text.
            for token in re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text):
                candidates.add(token)
            for token in re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", text):
                candidates.add(token)

    return {item for item in candidates if item}


def _resolve_active_python_files(
    generated_root: Path,
    generated_entries: List[dict],
    layout_root: str,
) -> List[Path]:
    layout_root = str(layout_root or "").strip().strip("/")
    if not layout_root:
        return []
    layout_dir = (generated_root / layout_root).resolve()
    active_files: List[Path] = []

    for entry in generated_entries:
        if not isinstance(entry, dict):
            continue
        files = entry.get("files", {})
        if not isinstance(files, dict):
            continue
        code_path_raw = files.get("code")
        if not code_path_raw:
            continue
        code_path = Path(str(code_path_raw))
        if not code_path.is_absolute():
            code_path = generated_root / code_path
        code_path = code_path.resolve()
        if not code_path.exists() or code_path.suffix != ".py":
            continue
        try:
            code_path.relative_to(layout_dir)
        except Exception:
            continue
        active_files.append(code_path)

    # Fallback for legacy runs with missing generated_entries metadata.
    if not active_files and layout_dir.exists():
        active_files = sorted(path.resolve() for path in layout_dir.rglob("*.py") if path.is_file())

    unique: Dict[str, Path] = {str(path): path for path in active_files}
    return sorted(unique.values(), key=lambda p: str(p))


def _validate_init_export_result(
    generated_root: Path,
    package_dirs: Set[Path],
    active_py_files: List[Path],
    written_init_files: List[Path],
) -> Dict[str, Any]:
    compile_failures: List[Dict[str, str]] = []
    files_for_compile = list(active_py_files) + list(written_init_files)
    unique_compile_files = {str(path.resolve()): path for path in files_for_compile}
    for file_path in sorted(unique_compile_files.values(), key=lambda p: str(p)):
        try:
            py_compile.compile(str(file_path), doraise=True)
        except Exception as exc:
            compile_failures.append({"file": str(file_path), "error": str(exc)})

    import_failures: List[Dict[str, str]] = []
    import_success = 0
    package_module_names = sorted(
        ".".join((pkg.relative_to(generated_root).parts))
        for pkg in package_dirs
    )
    original_sys_path = list(sys.path)
    try:
        sys.path.insert(0, str(generated_root.resolve()))
        importlib.invalidate_caches()
        for module_name in package_module_names:
            try:
                importlib.import_module(module_name)
                import_success += 1
            except Exception as exc:
                import_failures.append(
                    {
                        "module": module_name,
                        "error": str(exc),
                    }
                )
    finally:
        sys.path[:] = original_sys_path

    return {
        "compile_failure_count": len(compile_failures),
        "compile_failures": compile_failures[:200],
        "import_success_count": import_success,
        "import_failure_count": len(import_failures),
        "import_failures": import_failures[:200],
    }


def _collect_failed_init_files_for_llm_fix(
    generated_root: Path,
    validation: Dict[str, Any],
) -> List[Tuple[Path, List[Dict[str, Any]]]]:
    candidates: Dict[str, List[Dict[str, Any]]] = {}
    import_failures = validation.get("import_failures", []) if isinstance(validation, dict) else []
    for failure in import_failures:
        if not isinstance(failure, dict):
            continue
        module_name = str(failure.get("module", "")).strip()
        if not module_name:
            continue
        package_dir = generated_root / module_name.replace(".", "/")
        init_file = package_dir / "__init__.py"
        if init_file.exists():
            candidates.setdefault(str(init_file.resolve()), []).append(failure)

    compile_failures = validation.get("compile_failures", []) if isinstance(validation, dict) else []
    for failure in compile_failures:
        if not isinstance(failure, dict):
            continue
        file_path = str(failure.get("file", "")).strip()
        if not file_path:
            continue
        path = Path(file_path)
        if path.name != "__init__.py":
            continue
        if path.exists():
            candidates.setdefault(str(path.resolve()), []).append(failure)

    rows: List[Tuple[Path, List[Dict[str, Any]]]] = []
    for init_file, failures in sorted(candidates.items(), key=lambda item: item[0]):
        rows.append((Path(init_file), failures))
    return rows


def _llm_fix_failed_init_exports(
    generated_root: Path,
    api_config: Dict[str, Any],
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    if not api_config.get("api_key"):
        return {"enabled": False, "reason": "missing_api_key", "attempted": 0, "fixed": 0}

    candidates = _collect_failed_init_files_for_llm_fix(generated_root, validation)
    if not candidates:
        return {"enabled": True, "reason": "no_failed_init_files", "attempted": 0, "fixed": 0}

    try:
        from agents.llm_client import LLMClient
    except Exception as exc:
        return {"enabled": False, "reason": f"llm_client_import_failed: {exc}", "attempted": 0, "fixed": 0}

    llm = LLMClient(api_config, str(generated_root.parent), agent_name="init_export_fix")
    attempted = 0
    fixed = 0
    failures: List[Dict[str, Any]] = []
    fixed_files: List[str] = []

    for init_file, related_failures in candidates:
        attempted += 1
        package_dir = init_file.parent
        modules = sorted(path.stem for path in package_dir.glob("*.py") if path.name != "__init__.py")
        children = sorted(path.name for path in package_dir.iterdir() if path.is_dir() and (path / "__init__.py").exists())
        current_text = init_file.read_text(encoding="utf-8")
        prompt = f"""You are fixing a Python package __init__.py to maximize import reliability.

Package directory: {package_dir}
Sibling modules: {modules}
Child packages: {children}
Related failures:
{json.dumps(related_failures, ensure_ascii=False, indent=2)}

Current __init__.py:
```python
{current_text}
```

Constraints:
1) Prioritize import stability over aggressive symbol re-export.
2) You may use:
   - from . import module_name
   - optional guarded symbol imports if clearly safe.
3) Must define __all__.
4) Return strict JSON only:
{{"init_code": "<full new __init__.py content>"}}
"""
        try:
            response = llm.call_json(
                [
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=32768,
            )
            init_code = str(response.get("init_code", "")).strip() if isinstance(response, dict) else ""
            if not init_code:
                failures.append({"file": str(init_file), "error": "empty_init_code"})
                continue
            init_file.write_text(init_code + ("\n" if not init_code.endswith("\n") else ""), encoding="utf-8")
            fixed += 1
            fixed_files.append(str(init_file))
        except Exception as exc:
            failures.append({"file": str(init_file), "error": str(exc)})

    return {
        "enabled": True,
        "attempted": attempted,
        "fixed": fixed,
        "fixed_files": fixed_files,
        "failures": failures[:200],
    }


def _build_package_init_exports(
    generated_root: Path,
    generated_entries: List[dict],
    layout_root: str,
    api_config: Optional[Dict[str, Any]] = None,
    package_api_plan: Optional[Dict[str, Any]] = None,
    lazy_imports: bool = False,
) -> Dict[str, Any]:
    layout_root = str(layout_root or "").strip().strip("/")
    if not layout_root:
        return {"enabled": False, "reason": "empty_layout_root"}

    active_py_files = _resolve_active_python_files(generated_root, generated_entries, layout_root)
    if not active_py_files:
        return {
            "enabled": True,
            "layout_root": layout_root,
            "active_python_files": 0,
            "packages_total": 0,
            "packages_updated": 0,
            "reason": "no_active_python_files",
        }

    layout_dir = (generated_root / layout_root).resolve()
    package_dirs: Set[Path] = set()
    package_to_modules: Dict[Path, Set[str]] = {}
    module_declared_symbols: Dict[Path, Set[str]] = {}
    created_init_files: List[str] = []

    for entry in generated_entries:
        if not isinstance(entry, dict):
            continue
        files = entry.get("files", {})
        if not isinstance(files, dict):
            continue
        code_path_raw = files.get("code")
        if not code_path_raw:
            continue
        code_path = Path(str(code_path_raw))
        if not code_path.is_absolute():
            code_path = generated_root / code_path
        code_path = code_path.resolve()
        if not code_path.exists() or code_path.suffix != ".py":
            continue
        declared = _extract_declared_symbols_from_generated_entry(entry)
        if not declared:
            continue
        module_declared_symbols.setdefault(code_path, set()).update(declared)

    if isinstance(package_api_plan, dict):
        for row in package_api_plan.get("components", []):
            if not isinstance(row, dict):
                continue
            planned_file_path = str(row.get("planned_file_path", "")).strip().replace("\\", "/")
            if not planned_file_path:
                continue
            module_path = (generated_root / planned_file_path).resolve()
            if not module_path.exists() or module_path.suffix != ".py":
                continue
            symbols = row.get("export_symbols", [])
            if not isinstance(symbols, list):
                continue
            declared = {str(symbol).strip() for symbol in symbols if str(symbol).strip()}
            if declared:
                module_declared_symbols.setdefault(module_path, set()).update(declared)

    for py_file in active_py_files:
        current = py_file.parent.resolve()
        while True:
            package_dirs.add(current)
            if current == layout_dir:
                break
            if layout_dir not in current.parents:
                break
            current = current.parent
        module_name = py_file.stem
        if module_name != "__init__":
            package_to_modules.setdefault(py_file.parent.resolve(), set()).add(module_name)

    # Ensure __init__.py exists for each package directory.
    for package_dir in sorted(package_dirs, key=lambda p: str(p)):
        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
            created_init_files.append(str(init_file))

    child_packages: Dict[Path, List[str]] = {}
    for pkg in package_dirs:
        children = [child.name for child in package_dirs if child.parent == pkg]
        child_packages[pkg] = sorted(children)

    packages_updated = 0
    packages_skipped_custom = 0
    total_exported_symbols = 0
    duplicate_symbol_skips: List[Dict[str, Any]] = []
    package_reports: List[Dict[str, Any]] = []
    written_init_files: List[Path] = []

    # Deepest first, then root package.
    package_order = sorted(package_dirs, key=lambda p: (-len(p.relative_to(layout_dir).parts), str(p)))
    for package_dir in package_order:
        init_file = package_dir / "__init__.py"
        existing_text = init_file.read_text(encoding="utf-8") if init_file.exists() else ""
        if existing_text.strip() and _INIT_EXPORT_MARKER not in existing_text:
            packages_skipped_custom += 1
            package_reports.append(
                {
                    "package_dir": str(package_dir),
                    "status": "skipped_custom_init",
                }
            )
            continue

        import_lines: List[str] = []
        lazy_modules: Dict[str, str] = {}
        lazy_symbols: Dict[str, str] = {}
        exported_names: List[str] = []
        used_export_names: Set[str] = set()
        module_rows: List[Dict[str, Any]] = []

        for module_name in sorted(package_to_modules.get(package_dir, set())):
            module_file = package_dir / f"{module_name}.py"
            if not module_file.exists():
                continue
            try:
                symbols_info = _extract_module_export_symbols(module_file)
                explicit = symbols_info.get("explicit_symbols", []) or []
                public = symbols_info.get("public_symbols", []) or []
            except Exception as exc:
                logging.warning("Init-export symbol extraction failed for %s: %s", module_file, exc)
                explicit = []
                public = []

            # Default explicit mode: only export symbols declared by component plan/responsibilities.
            declared_candidates = module_declared_symbols.get(module_file.resolve(), set())
            explicit_declared = [symbol for symbol in explicit if symbol in declared_candidates]
            public_declared = [symbol for symbol in public if symbol in declared_candidates]
            chosen = _dedupe_keep_order(explicit_declared or public_declared)
            mode = "declared_explicit" if explicit_declared else ("declared_public" if public_declared else "module_fallback")

            selected: List[str] = []
            for symbol in chosen:
                if symbol in used_export_names:
                    duplicate_symbol_skips.append(
                        {
                            "package_dir": str(package_dir),
                            "module": module_name,
                            "symbol": symbol,
                        }
                    )
                    continue
                used_export_names.add(symbol)
                selected.append(symbol)

            if selected:
                if lazy_imports:
                    for symbol in selected:
                        lazy_symbols[symbol] = f".{module_name}"
                else:
                    import_lines.append(f"from .{module_name} import {', '.join(selected)}")
                exported_names.extend(selected)
            else:
                if lazy_imports:
                    lazy_modules[module_name] = f".{module_name}"
                else:
                    import_lines.append(f"from . import {module_name}")
                if module_name not in used_export_names:
                    used_export_names.add(module_name)
                    exported_names.append(module_name)

            module_rows.append(
                {
                    "module": module_name,
                    "symbol_mode": mode,
                    "exported_symbols": selected,
                    "fallback_module_import": not bool(selected),
                }
            )

        for child_name in child_packages.get(package_dir, []):
            if lazy_imports:
                lazy_modules[child_name] = f".{child_name}"
            else:
                import_lines.append(f"from . import {child_name}")
            if child_name not in used_export_names:
                used_export_names.add(child_name)
                exported_names.append(child_name)

        exported_names = _dedupe_keep_order(exported_names)
        total_exported_symbols += len(exported_names)
        body_lines: List[str] = [
            _INIT_EXPORT_MARKER,
            "# This file is generated to provide stable package/subpackage imports.",
            "",
        ]
        if lazy_imports and (lazy_modules or lazy_symbols):
            body_lines.extend(
                [
                    "import importlib as _importlib",
                    "",
                    "_LAZY_MODULES = {",
                ]
            )
            for name, module in sorted(lazy_modules.items()):
                body_lines.append(f'    "{name}": "{module}",')
            body_lines.extend(["}", "_LAZY_SYMBOLS = {"])
            for name, module in sorted(lazy_symbols.items()):
                body_lines.append(f'    "{name}": "{module}",')
            body_lines.extend(
                [
                    "}",
                    "",
                    "def __getattr__(name):",
                    "    if name in _LAZY_MODULES:",
                    "        value = _importlib.import_module(_LAZY_MODULES[name], __name__)",
                    "        globals()[name] = value",
                    "        return value",
                    "    if name in _LAZY_SYMBOLS:",
                    "        module = _importlib.import_module(_LAZY_SYMBOLS[name], __name__)",
                    "        value = getattr(module, name)",
                    "        globals()[name] = value",
                    "        return value",
                    "    raise AttributeError(f\"module {__name__!r} has no attribute {name!r}\")",
                    "",
                ]
            )
        elif import_lines:
            body_lines.extend(import_lines)
            body_lines.append("")
        body_lines.append("__all__ = [")
        for name in exported_names:
            body_lines.append(f'    "{name}",')
        body_lines.append("]")
        body_lines.append("")
        init_content = "\n".join(body_lines)

        init_file.write_text(init_content, encoding="utf-8")
        written_init_files.append(init_file)
        packages_updated += 1
        package_reports.append(
            {
                "package_dir": str(package_dir),
                "status": "updated",
                "module_count": len(module_rows),
                "child_package_count": len(child_packages.get(package_dir, [])),
                "module_rows": module_rows,
                "all_exports": exported_names,
            }
        )

    validation = _validate_init_export_result(
        generated_root=generated_root,
        package_dirs=package_dirs,
        active_py_files=active_py_files,
        written_init_files=written_init_files,
    )

    llm_correction: Dict[str, Any] = {"enabled": False, "reason": "not_triggered"}
    if validation.get("compile_failure_count", 0) > 0 or validation.get("import_failure_count", 0) > 0:
        llm_correction = _llm_fix_failed_init_exports(
            generated_root=generated_root,
            api_config=api_config or {},
            validation=validation,
        )
        if llm_correction.get("fixed", 0) > 0:
            validation = _validate_init_export_result(
                generated_root=generated_root,
                package_dirs=package_dirs,
                active_py_files=active_py_files,
                written_init_files=written_init_files,
            )

    return {
        "enabled": True,
        "layout_root": layout_root,
        "active_python_files": len(active_py_files),
        "packages_total": len(package_dirs),
        "packages_updated": packages_updated,
        "packages_skipped_custom_init": packages_skipped_custom,
        "created_init_files": created_init_files,
        "total_exported_symbols": total_exported_symbols,
        "duplicate_symbol_skips": duplicate_symbol_skips[:200],
        "validation": validation,
        "llm_correction": llm_correction,
        "packages": package_reports,
    }


def load_requirements(args: argparse.Namespace) -> str:
    if args.requirements_file and args.requirements_file.exists():
        return args.requirements_file.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Requirements file not found: {args.requirements_file}")


def _nodes_from_requirement_items(items: List[dict], source_file: Path) -> List[RequirementNode]:
    nodes: List[RequirementNode] = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("title") or item.get("requirement") or "").strip()
        description = str(
            item.get("description")
            or item.get("details")
            or item.get("summary")
            or item.get("text")
            or ""
        ).strip()
        if not name:
            name = f"requirement_{i}"
        metadata = {k: v for k, v in item.items() if k not in {"name", "description"}}
        metadata["source_file"] = str(source_file)
        nodes.append(RequirementNode(name=name, description=description, metadata=metadata))
    return nodes


def _load_jsonl_requirements(file_path: Path) -> List[RequirementNode]:
    nodes: List[RequirementNode] = []
    raw_lines = file_path.read_text(encoding="utf-8").splitlines()
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            items: List[dict] = []
            for key, value in payload.items():
                if isinstance(value, dict):
                    candidate = dict(value)
                    candidate.setdefault("name", value.get("requirement") or value.get("title") or key)
                    candidate.setdefault("description", value.get("details") or value.get("summary") or "")
                    candidate.setdefault("repo_key", key)
                    items.append(candidate)
                else:
                    items.append({"name": key, "description": str(value)})
            nodes.extend(_nodes_from_requirement_items(items, file_path))
        elif isinstance(payload, list):
            nodes.extend(_nodes_from_requirement_items(payload, file_path))
    return nodes


def load_new_requirements(file_path: Path) -> List[RequirementNode]:
    """Load new requirements for DAG evolution from JSON/JSONL or text."""
    if not file_path.exists():
        raise FileNotFoundError(f"Evolution requirements file not found: {file_path}")
    
    if file_path.suffix.lower() == ".jsonl":
        return _load_jsonl_requirements(file_path)

    raw_text = file_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []
    
    nodes: List[RequirementNode] = []
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        data = None
    
    if isinstance(data, str):
        return [
            RequirementNode(
                name=data.strip() or "requirement_1",
                description=data.strip(),
                metadata={"source_file": str(file_path), "input_format": "json_string"},
            )
        ]
    if isinstance(data, dict):
        if "requirements" in data and isinstance(data["requirements"], list):
            items = data["requirements"]
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    if items:
        return _nodes_from_requirement_items(items, file_path)
    if data is not None:
        return nodes
    
    for i, line in enumerate(raw_text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            name, description = stripped.split(":", 1)
            name = name.strip()
            description = description.strip()
        else:
            name = stripped
            description = stripped
        nodes.append(
            RequirementNode(
                name=name or f"requirement_{i}",
                description=description,
                metadata={"source_file": str(file_path), "input_format": "text"},
            )
        )
    return nodes


def load_json_if_exists(file_path: Path, force_regenerate: bool = False) -> Optional[dict]:
    """Load JSON from file if it exists, otherwise return None."""
    if force_regenerate:
        return None
    
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logging.warning(f"Failed to load existing file: {file_path}")
            return None
    return None


def outputs_stale(output_path: Path, input_paths: List[Path]) -> bool:
    """Return True when output is missing or older than any existing input artifact."""
    if not output_path.exists():
        return True
    try:
        output_mtime = output_path.stat().st_mtime
    except FileNotFoundError:
        return True
    for input_path in input_paths:
        if not input_path.exists():
            continue
        try:
            if input_path.stat().st_mtime > output_mtime:
                return True
        except FileNotFoundError:
            continue
    return False


def save_json(data: dict, file_path: Path) -> None:
    """Save JSON data to file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _stable_fingerprint(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_component_merge_input_snapshot(
    *,
    architectures: List[dict],
    output_dir: Path,
    repo: str,
    source_path: Optional[Path],
    requirements_path: Path,
    active_parent_names: Set[str],
) -> None:
    """Persist the fixed input used immediately before the first component merge."""
    snapshot_path = output_dir / "architectures_pre_component_merge.json"
    manifest_path = output_dir / "component_merge_input_manifest.json"
    save_json(architectures, snapshot_path)
    manifest = {
        "repo": repo,
        "stage": "pre_component_merge",
        "source_path": str(source_path) if source_path else "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parent_count": len(active_parent_names),
        "architecture_count": len(architectures),
        "component_count": count_architecture_components(architectures),
        "hashes": {
            "requirements_file": _file_sha256(requirements_path),
            "requirements_merge_result": _file_sha256(output_dir / "requirements_merge_result.json"),
            "requirements_for_dag": _file_sha256(output_dir / "requirements_for_dag.json"),
            "requirement_dag": _file_sha256(output_dir / "requirement_dag.json"),
            "decomposed_dag": _file_sha256(output_dir / "decomposed_dag.json"),
            "edges_for_dag": _file_sha256(output_dir / "edges_for_dag.json"),
            "dependency_graph": _file_sha256(output_dir / "dependency_graph.json"),
            "plan": _file_sha256(output_dir / "plan.json"),
            "architectures_pre_component_merge": _stable_fingerprint(architectures),
        },
    }
    save_json(manifest, manifest_path)


def _artifact_matches_fingerprint(artifact: Any, expected_fingerprint: str) -> bool:
    if not isinstance(artifact, dict):
        return False
    meta = artifact.get("_meta")
    if not isinstance(meta, dict):
        return False
    return str(meta.get("input_fingerprint") or "").strip() == expected_fingerprint


def _is_legacy_module_plan_reusable(artifact: Any) -> bool:
    return isinstance(artifact, dict) and (
        isinstance(artifact.get("module_families"), list)
        or isinstance(artifact.get("plans"), list)
    )


def _is_legacy_module_assignment_reusable(artifact: Any) -> bool:
    return isinstance(artifact, dict) and isinstance(
        artifact.get("component_package_path_index"), dict
    )


def _is_legacy_layout_plan_reusable(
    layout_grouping_report: Any,
    package_api_plan: Any,
) -> bool:
    return (
        isinstance(layout_grouping_report, dict)
        and isinstance(package_api_plan, dict)
        and isinstance(layout_grouping_report.get("candidate_packages"), list)
        and isinstance(package_api_plan.get("component_index"), dict)
    )


def _is_action_refinement_report_reusable(report: Any) -> bool:
    return isinstance(report, dict) and isinstance(report.get("stats"), dict)


def _is_multi_action_refinement_report_reusable(report: Any, rounds: int) -> bool:
    if not _is_action_refinement_report_reusable(report):
        return False
    stats = report.get("stats", {})
    return (
        int(stats.get("rounds_requested", 0) or 0) == int(rounds)
        and int(stats.get("rounds_completed", 0) or 0) > 0
        and isinstance(report.get("rounds"), list)
    )


def _attach_input_fingerprint(artifact: Any, input_fingerprint: str) -> Any:
    if not isinstance(artifact, dict):
        return artifact
    updated = dict(artifact)
    meta = dict(updated.get("_meta", {}) or {})
    meta["input_fingerprint"] = input_fingerprint
    updated["_meta"] = meta
    return updated


class StageTimer:
    """Lightweight stage timing tracker for the main pipeline."""

    def __init__(self, existing_report: Optional[Dict[str, Any]] = None) -> None:
        self._active: Dict[str, Dict[str, Any]] = {}
        self._records: List[Dict[str, Any]] = []
        self._stage_history: List[Dict[str, Any]] = []
        self._status: str = "running"
        self._error: str = ""
        self._persist_callback: Optional[Any] = None
        self._run_started_at: str = self._timestamp_now()
        self._updated_at: str = self._run_started_at
        self._base_total_recorded_duration_sec: float = 0.0
        self._previous_runs: List[Dict[str, Any]] = []

        if isinstance(existing_report, dict):
            self._base_total_recorded_duration_sec = self._safe_float(
                existing_report.get("total_recorded_duration_sec", 0.0)
            )
            existing_runs = existing_report.get("runs")
            if isinstance(existing_runs, list):
                self._previous_runs = [dict(item) for item in existing_runs if isinstance(item, dict)]
            else:
                legacy_run = self._legacy_run_summary(existing_report)
                if legacy_run is not None:
                    self._previous_runs = [legacy_run]

            existing_history = existing_report.get("stage_history")
            if isinstance(existing_history, list):
                self._stage_history = [dict(item) for item in existing_history if isinstance(item, dict)]
            else:
                legacy_stages = existing_report.get("stages")
                if isinstance(legacy_stages, list):
                    legacy_run_started_at = str(existing_report.get("run_started_at") or "").strip()
                    for item in legacy_stages:
                        if not isinstance(item, dict):
                            continue
                        stage_row = dict(item)
                        if legacy_run_started_at and "run_started_at" not in stage_row:
                            stage_row["run_started_at"] = legacy_run_started_at
                        self._stage_history.append(stage_row)

    @contextmanager
    def stage(self, name: str, **meta: Any):
        self.begin(name)
        try:
            yield
            status = "completed"
        except Exception:
            status = "failed"
            raise
        finally:
            self.end(name, status=status, **meta)

    def begin(self, name: str) -> None:
        self._active[name] = {
            "perf_counter": time.perf_counter(),
            "started_at": self._timestamp_now(),
        }
        self._updated_at = self._timestamp_now()

    def end(self, name: str, status: str = "completed", **meta: Any) -> None:
        state = self._active.pop(name, None)
        if not isinstance(state, dict):
            return
        ended_at = self._timestamp_now()
        record = {
            "stage": name,
            "duration_sec": round(time.perf_counter() - float(state.get("perf_counter", time.perf_counter())), 3),
            "status": status,
            "meta": meta,
            "started_at": state.get("started_at"),
            "ended_at": ended_at,
            "run_started_at": self._run_started_at,
        }
        self._records.append(record)
        self._stage_history.append(dict(record))
        self._updated_at = ended_at
        self._auto_persist()

    def fail(self, exc: BaseException) -> None:
        self._status = "failed"
        self._error = f"{type(exc).__name__}: {exc}"
        self._updated_at = self._timestamp_now()
        self._auto_persist()

    def finish(self) -> None:
        if self._status == "running":
            self._status = "completed"
        self._updated_at = self._timestamp_now()
        self._auto_persist()

    def set_persist_callback(self, callback: Any) -> None:
        self._persist_callback = callback

    @staticmethod
    def _timestamp_now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    def _current_run_duration_sec(self) -> float:
        return round(sum(self._safe_float(item.get("duration_sec", 0.0)) for item in self._records), 3)

    def _current_run_summary(self) -> Dict[str, Any]:
        return {
            "run_started_at": self._run_started_at,
            "run_finished_at": self._updated_at,
            "status": self._status,
            "error": self._error,
            "recorded_duration_sec": self._current_run_duration_sec(),
            "stage_count": len(self._records),
        }

    def _legacy_run_summary(self, existing_report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(existing_report, dict):
            return None
        has_stage_rows = isinstance(existing_report.get("stages"), list) and bool(existing_report.get("stages"))
        legacy_total = self._safe_float(existing_report.get("total_recorded_duration_sec", 0.0))
        if not has_stage_rows and legacy_total <= 0:
            return None
        return {
            "run_started_at": str(existing_report.get("run_started_at") or "").strip(),
            "run_finished_at": str(existing_report.get("completed_at") or existing_report.get("updated_at") or "").strip(),
            "status": str(existing_report.get("status") or "completed"),
            "error": str(existing_report.get("error") or ""),
            "recorded_duration_sec": legacy_total,
            "stage_count": len(existing_report.get("stages", [])) if isinstance(existing_report.get("stages"), list) else 0,
        }

    def _auto_persist(self) -> None:
        if self._persist_callback is None:
            return
        try:
            self._persist_callback()
        except Exception as exc:
            logging.debug("Failed to auto-persist stage timing report: %s", exc)

    def to_dict(self) -> Dict[str, Any]:
        records = list(self._records)
        records.sort(key=lambda item: item.get("duration_sec", 0.0), reverse=True)
        active = [
            {
                "stage": name,
                "duration_sec": round(time.perf_counter() - float(state.get("perf_counter", time.perf_counter())), 3),
                "status": "active_on_exit",
                "started_at": state.get("started_at"),
                "meta": {},
            }
            for name, state in self._active.items()
        ]
        current_run_duration_sec = self._current_run_duration_sec()
        total_recorded_duration_sec = round(self._base_total_recorded_duration_sec + current_run_duration_sec, 3)
        return {
            "status": self._status,
            "error": self._error,
            "run_started_at": self._run_started_at,
            "updated_at": self._updated_at,
            "completed_at": self._updated_at if self._status in {"completed", "failed"} and not self._active else "",
            "previous_total_recorded_duration_sec": round(self._base_total_recorded_duration_sec, 3),
            "current_run_recorded_duration_sec": current_run_duration_sec,
            "total_recorded_duration_sec": total_recorded_duration_sec,
            "stages": records,
            "stage_history": list(self._stage_history),
            "runs": [*self._previous_runs, self._current_run_summary()],
            "active_on_exit": active,
        }

    def log_summary(self) -> None:
        if not self._records:
            return
        top = sorted(self._records, key=lambda item: item.get("duration_sec", 0.0), reverse=True)[:8]
        summary = ", ".join(
            f"{item['stage']}={float(item['duration_sec']):.2f}s"
            for item in top
        )
        logging.info("Stage timing summary: %s", summary)


def extract_requirements_items(payload: Any) -> List[dict]:
    """Extract a normalized requirement list from various payload shapes."""
    items: List[Any]
    if isinstance(payload, dict):
        if isinstance(payload.get("requirements_after_merge"), list):
            items = payload.get("requirements_after_merge", [])
        elif isinstance(payload.get("requirements"), list):
            items = payload.get("requirements", [])
        else:
            items = []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    normalized: List[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        description = str(item.get("description", "")).strip()
        normalized.append({"name": name, "description": description})
    return normalized


def _normalize_name_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_operation_name(operation: Any) -> str:
    value = str(operation or "").strip().lower()
    return {
        "create": "add",
        "new": "add",
    }.get(value, value or "add")


def build_evolution_action_override(decision: dict[str, Any]) -> dict[str, Any] | None:
    """Convert strategist decision to a validated evolution action."""
    action = dict(decision or {})
    tag = str(action.get("tag") or "").upper()
    relation_type = str(action.get("relation_type") or "").upper()
    affected = _normalize_name_list(action.get("affected_requirements"))
    target = str(action.get("target") or "").strip()
    targets = _normalize_name_list(action.get("targets"))
    operation = _normalize_operation_name(action.get("operation"))

    if not target and affected:
        target = affected[0]
    if not targets and affected:
        targets = affected

    if tag == "EXISTING":
        return None

    if tag == "ADD":
        operation = "add"
    elif tag == "RELATION":
        if relation_type == "CHILD":
            operation = "add"
            if not _normalize_name_list(action.get("suggested_parents")):
                if affected:
                    action["suggested_parents"] = affected
                elif target:
                    action["suggested_parents"] = [target]
        elif relation_type == "REVISE":
            operation = "revise"
            if target:
                action.setdefault("target", target)
        elif relation_type == "MERGE":
            operation = "merge"
            if targets:
                action.setdefault("targets", targets)
        elif relation_type == "SPLIT":
            operation = "split"
            if target:
                action.setdefault("target", target)
        elif relation_type == "DELETE":
            operation = "delete"
            if targets:
                action.setdefault("targets", targets)

    if operation == "revise" and not str(action.get("target") or "").strip() and target:
        action["target"] = target
    if operation == "split" and not str(action.get("target") or "").strip() and target:
        action["target"] = target
    if operation in {"merge", "delete"} and not _normalize_name_list(action.get("targets")) and targets:
        action["targets"] = targets

    missing_target = operation in {"revise", "split"} and not str(action.get("target") or "").strip()
    missing_targets = operation in {"merge", "delete"} and not _normalize_name_list(action.get("targets"))
    if missing_target or missing_targets:
        logging.warning(
            "Evolution decision is missing target info for operation '%s'; fallback to add",
            operation,
        )
        operation = "add"

    action["operation"] = operation
    return action


def _parent_from_architecture_entry(arch: dict[str, Any]) -> str:
    return str(arch.get("parent_task") or arch.get("task") or "").strip()


def filter_architectures_for_active_parents(
    architectures: Any,
    active_parents: set[str],
) -> List[dict]:
    """Keep architecture entries that still map to active requirement parents."""
    if not isinstance(architectures, list):
        return []
    filtered: List[dict] = []
    for arch in architectures:
        if not isinstance(arch, dict):
            continue
        parent = _parent_from_architecture_entry(arch)
        if not parent or parent not in active_parents:
            continue
        normalized = dict(arch)
        normalized["parent_task"] = parent
        normalized.setdefault("task", parent)
        filtered.append(normalized)
    return filtered


def _architecture_entry_is_complete(entry: dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    parent = _parent_from_architecture_entry(entry)
    if not parent:
        return False
    architecture = entry.get("architecture", {})
    if not isinstance(architecture, dict):
        return False
    components = architecture.get("components", [])
    return isinstance(components, list) and len(components) > 0


def detect_completed_architecture_parents(
    architectures: Any,
    active_parents: set[str],
) -> set[str]:
    if not isinstance(architectures, list):
        return set()
    completed: set[str] = set()
    for arch in architectures:
        if not _architecture_entry_is_complete(arch):
            continue
        parent = _parent_from_architecture_entry(arch)
        if parent in active_parents:
            completed.add(parent)
    return completed


def _parent_from_action_entry(action: dict[str, Any]) -> str:
    return str(action.get("parent_task") or action.get("task") or "").strip()


def merge_actions_for_architectures(
    existing_actions: Any,
    new_actions: List[dict],
    architectures: List[dict],
) -> List[dict]:
    """Align actions with architecture parent order and keep entries by parent key."""
    existing_by_parent: dict[str, dict] = {}
    if isinstance(existing_actions, list):
        for entry in existing_actions:
            if not isinstance(entry, dict):
                continue
            parent = _parent_from_action_entry(entry)
            if parent:
                existing_by_parent[parent] = dict(entry)

    new_by_parent: dict[str, dict] = {}
    for entry in new_actions:
        if not isinstance(entry, dict):
            continue
        parent = _parent_from_action_entry(entry)
        if parent:
            new_by_parent[parent] = dict(entry)

    merged: List[dict] = []
    for arch in architectures:
        parent = _parent_from_architecture_entry(arch)
        if not parent:
            continue
        chosen = dict(new_by_parent.get(parent) or existing_by_parent.get(parent) or {})
        chosen["task"] = parent
        chosen.setdefault("actions", [])
        merged.append(chosen)
    return merged


def apply_action_hints_to_architectures(
    architectures: List[dict],
    actions: List[dict],
) -> List[dict]:
    """Propagate strategist component actions back into architectures before codegen."""
    if not isinstance(architectures, list):
        return []

    actions_by_parent: dict[str, dict[str, dict[str, Any]]] = {}
    if isinstance(actions, list):
        for entry in actions:
            if not isinstance(entry, dict):
                continue
            parent = _parent_from_action_entry(entry)
            if not parent:
                continue
            per_component = actions_by_parent.setdefault(parent, {})
            rows = entry.get("actions", [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                component = str(row.get("component") or "").strip()
                if component:
                    per_component[component] = row

    updated: List[dict] = []
    for arch in architectures:
        if not isinstance(arch, dict):
            continue
        arch_copy = dict(arch)
        parent = _parent_from_architecture_entry(arch_copy)
        architecture = dict(arch_copy.get("architecture", {}) or {})
        components = architecture.get("components", [])
        hinted_components: List[dict] = []
        for component in components if isinstance(components, list) else []:
            if not isinstance(component, dict):
                hinted_components.append(component)
                continue
            comp_copy = dict(component)
            comp_copy.pop("recommended_action", None)
            comp_copy.pop("recommended_action_rationale", None)
            comp_copy.pop("recommended_target_component", None)
            comp_copy.pop("recommended_action_origin", None)
            comp_copy.pop("split_partition_evidence", None)
            action_row = actions_by_parent.get(parent, {}).get(str(comp_copy.get("name") or "").strip())
            if isinstance(action_row, dict):
                comp_copy["recommended_action"] = str(action_row.get("action") or "").strip()
                rationale = str(action_row.get("rationale") or "").strip()
                if rationale:
                    comp_copy["recommended_action_rationale"] = rationale
                target_component = str(action_row.get("target_component") or "").strip()
                if target_component:
                    comp_copy["recommended_target_component"] = target_component
                action_origin = str(action_row.get("action_origin") or "").strip()
                if action_origin:
                    comp_copy["recommended_action_origin"] = action_origin
                partition_evidence = action_row.get("split_partition_evidence")
                if isinstance(partition_evidence, dict):
                    comp_copy["split_partition_evidence"] = partition_evidence
            hinted_components.append(comp_copy)
        architecture["components"] = hinted_components
        arch_copy["architecture"] = architecture
        updated.append(arch_copy)
    return updated


def count_architecture_components(architectures: List[dict]) -> int:
    """Count architecture components across parent entries."""
    return sum(
        len((arch.get("architecture", {}) or {}).get("components", []) or [])
        for arch in architectures
        if isinstance(arch, dict)
    )


def count_action_types(actions: List[dict]) -> Dict[str, int]:
    """Count strategist action labels across all parent action entries."""
    counts: Dict[str, int] = {}
    if not isinstance(actions, list):
        return counts
    for entry in actions:
        if not isinstance(entry, dict):
            continue
        rows = entry.get("actions", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").strip().lower() or "unknown"
            counts[action] = counts.get(action, 0) + 1
    return counts


def collect_saved_component_actions(actions: List[dict]) -> Dict[str, Set[str]]:
    """Return components whose strategist action is save, grouped by parent."""
    stopped: Dict[str, Set[str]] = {}
    if not isinstance(actions, list):
        return stopped
    for entry in actions:
        if not isinstance(entry, dict):
            continue
        parent = _parent_from_action_entry(entry)
        rows = entry.get("actions", [])
        if not parent or not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("action") or "").strip().lower() != "save":
                continue
            component = str(row.get("component") or "").strip()
            if component:
                stopped.setdefault(parent, set()).add(component)
    return stopped


def count_stopped_components(stopped_components: Dict[str, Set[str]]) -> int:
    return sum(len(names) for names in stopped_components.values())


def count_unstopped_architecture_components(
    architectures: List[dict],
    stopped_components: Dict[str, Set[str]],
) -> int:
    count = 0
    for arch in architectures if isinstance(architectures, list) else []:
        if not isinstance(arch, dict):
            continue
        parent = _parent_from_architecture_entry(arch)
        architecture = arch.get("architecture", {}) if isinstance(arch.get("architecture", {}), dict) else {}
        components = architecture.get("components", [])
        if not isinstance(components, list):
            continue
        stopped = stopped_components.get(parent, set())
        for component in components:
            if not isinstance(component, dict):
                count += 1
                continue
            if str(component.get("name") or "").strip() not in stopped:
                count += 1
    return count


def filter_architectures_for_unstopped_components(
    architectures: List[dict],
    stopped_components: Dict[str, Set[str]],
) -> List[dict]:
    """Build action-selection inputs excluding components already saved by the strategist."""
    if not isinstance(architectures, list):
        return []
    filtered: List[dict] = []
    for arch in architectures:
        if not isinstance(arch, dict):
            continue
        parent = _parent_from_architecture_entry(arch)
        architecture = dict(arch.get("architecture", {}) or {})
        components = architecture.get("components", [])
        if not isinstance(components, list):
            components = []
        stopped = stopped_components.get(parent, set())
        active_components = [
            component
            for component in components
            if not isinstance(component, dict)
            or str(component.get("name") or "").strip() not in stopped
        ]
        if not active_components:
            continue
        arch_copy = dict(arch)
        architecture["components"] = active_components
        arch_copy["architecture"] = architecture
        filtered.append(arch_copy)
    return filtered


def add_saved_component_actions(
    actions: List[dict],
    architectures: List[dict],
    stopped_components: Dict[str, Set[str]],
) -> List[dict]:
    """Carry stopped components forward as save actions while active components keep new actions."""
    if not stopped_components:
        return actions
    actions_by_parent: Dict[str, dict] = {}
    for entry in actions if isinstance(actions, list) else []:
        if not isinstance(entry, dict):
            continue
        parent = _parent_from_action_entry(entry)
        if parent:
            actions_by_parent[parent] = dict(entry)

    merged: List[dict] = []
    for arch in architectures if isinstance(architectures, list) else []:
        if not isinstance(arch, dict):
            continue
        parent = _parent_from_architecture_entry(arch)
        if not parent:
            continue
        chosen = dict(actions_by_parent.get(parent) or {"task": parent, "actions": []})
        rows = list(chosen.get("actions", []) if isinstance(chosen.get("actions", []), list) else [])
        seen = {
            str(row.get("component") or "").strip()
            for row in rows
            if isinstance(row, dict)
        }
        architecture = arch.get("architecture", {}) if isinstance(arch.get("architecture", {}), dict) else {}
        components = architecture.get("components", [])
        for component in components if isinstance(components, list) else []:
            if not isinstance(component, dict):
                continue
            name = str(component.get("name") or "").strip()
            if name and name in stopped_components.get(parent, set()) and name not in seen:
                rows.append(
                    {
                        "component": name,
                        "action": "save",
                        "rationale": "Component was accepted as stable in a previous feedback round.",
                    }
                )
        chosen["task"] = parent
        chosen["actions"] = rows
        merged.append(chosen)
    return merged


def _build_decomposed_edge_set(
    decomposed_dag: RequirementDAG | Dict[str, Any] | None,
) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    if decomposed_dag is None:
        return set(), set()
    if isinstance(decomposed_dag, RequirementDAG):
        nodes = set(decomposed_dag.nodes.keys())
        edges: Set[Tuple[str, str]] = set()
        for source, children in decomposed_dag.adjacency.items():
            for child in children:
                if source != child:
                    edges.add(tuple(sorted((str(source), str(child)))))
        return nodes, edges
    if isinstance(decomposed_dag, dict):
        nodes_payload = decomposed_dag.get("nodes", [])
        edges_payload = decomposed_dag.get("edges", [])
        nodes = {
            str(node.get("name", "")).strip()
            for node in nodes_payload
            if isinstance(node, dict) and str(node.get("name", "")).strip()
        }
        edges: Set[Tuple[str, str]] = set()
        if isinstance(edges_payload, list):
            for edge in edges_payload:
                if not isinstance(edge, dict):
                    continue
                source = str(edge.get("source", "")).strip()
                target = str(edge.get("target", "")).strip()
                if source and target and source != target:
                    edges.add(tuple(sorted((source, target))))
        return nodes, edges
    return set(), set()


def _component_metric_payload(
    component: Dict[str, Any],
    *,
    dag_nodes: Set[str],
    dag_edges: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    served = component.get("serves_subrequirements", [])
    valid = {
        str(item).strip()
        for item in (served if isinstance(served, list) else [])
        if str(item).strip() in dag_nodes
    }
    size = len(valid)
    possible_internal = size * (size - 1) / 2
    internal_edges = sum(1 for left, right in dag_edges if left in valid and right in valid)
    cohesion = (internal_edges / possible_internal) if possible_internal else 1.0
    return {
        "name": str(component.get("name") or "").strip(),
        "subrequirements": valid,
        "size": size,
        "internal_edges": internal_edges,
        "possible_internal_edges": possible_internal,
        "cohesion": cohesion,
    }


def _safe_jaccard(left: Set[str], right: Set[str]) -> float:
    union = set(left) | set(right)
    if not union:
        return 0.0
    return len(set(left) & set(right)) / len(union)


def _metric_merge_layer_groups(name: str) -> Set[str]:
    tokens = {
        tok
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", str(name or "").lower())
        if tok
    }
    vocab = {
        "core": {"core"},
        "api": {"api", "surface", "sdk", "interface", "interfaces", "contract", "contracts"},
        "engine": {"engine", "orchestrator", "manager", "runtime", "controller", "executor", "pipeline"},
        "integration": {
            "integration", "adapter", "adapters", "bridge", "resolver", "registry", "registries",
            "loader", "backend", "backends", "tooling", "tool", "renderer", "formatting", "export", "exports",
        },
    }
    groups: Set[str] = set()
    for label, terms in vocab.items():
        if tokens & terms:
            groups.add(label)
    return groups


def _metric_merge_layer_penalty(source_name: str, target_name: str) -> float:
    left = _metric_merge_layer_groups(source_name)
    right = _metric_merge_layer_groups(target_name)
    if not left or not right:
        return 1.0
    conflicts = {
        ("core", "api"),
        ("api", "core"),
        ("engine", "api"),
        ("api", "engine"),
        ("engine", "integration"),
        ("integration", "engine"),
        ("core", "integration"),
        ("integration", "core"),
    }
    for pair in conflicts:
        if pair[0] in left and pair[1] in right:
            return 0.5
    return 1.0


def _find_component_action_row(
    entry: Dict[str, Any],
    component_name: str,
) -> Dict[str, Any] | None:
    rows = entry.get("actions", [])
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("component") or "").strip() == component_name:
            return row
    return None


def _build_metric_merge_judge(
    component_merge_agent: Any,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def _judge(candidate: Dict[str, Any]) -> Dict[str, Any]:
        if component_merge_agent is None or getattr(component_merge_agent, "llm_client", None) is None:
            return {
                "approved": False,
                "reason": "missing_llm_client",
                "risk": "high",
                "same_responsibility": False,
                "interface_conflict": True,
                "behavior_conflict": True,
            }
        payload = {
            "candidate_group": {
                "source_ids": ["C1", "C2"],
                "merged_name": candidate["target_component"],
                "merged_responsibilities": candidate.get("target_responsibilities", []),
                "merged_serves_subrequirements": sorted(
                    candidate.get("source_subrequirements", set())
                    | candidate.get("target_subrequirements", set())
                ),
                "reasoning": candidate.get("reason", ""),
                "confidence": 0.0,
            },
            "score_summary": {
                "pair_count": 1,
                "best_pair_score": float(candidate.get("coupling", 0.0)),
                "avg_pair_score": float(candidate.get("coupling", 0.0)),
                "worst_pair_score": float(candidate.get("coupling", 0.0)),
                "strongest_pair": ["C1", "C2"],
            },
            "relaxed_thresholds": {
                "best_pair_score": float(candidate.get("coupling", 0.0)),
                "avg_pair_score": float(candidate.get("coupling", 0.0)),
                "worst_pair_score": float(candidate.get("coupling", 0.0)),
            },
            "components": [
                {
                    "id": "C1",
                    "parent_task": candidate["parent_task"],
                    "name": candidate["source_component"],
                    "responsibilities": candidate.get("source_responsibilities", []),
                    "serves_subrequirements": sorted(candidate.get("source_subrequirements", set())),
                    "recommended_action": "merge",
                },
                {
                    "id": "C2",
                    "parent_task": candidate["parent_task"],
                    "name": candidate["target_component"],
                    "responsibilities": candidate.get("target_responsibilities", []),
                    "serves_subrequirements": sorted(candidate.get("target_subrequirements", set())),
                    "recommended_action": "merge",
                },
            ],
        }
        prompt = f"""
Review whether this metric-triggered component merge is safe.

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Definition of coupling for this review:
- `best_pair_score`, `avg_pair_score`, and `worst_pair_score` are all the same scalar coupling score for this two-component candidate.
- That coupling score is computed as:
  `Jaccard(served_subrequirements) * layering_penalty`
- `Jaccard(served_subrequirements)` means overlap of served subrequirements divided by their union.
- `layering_penalty < 1` means the names suggest likely complementary layering such as core/API, engine/integration, or core/adapter, so a high raw overlap should be discounted.
- Treat high coupling as evidence that the components occupy a similar requirement scope, but do not approve merges that would collapse a clean layered separation.

Approve only if the smaller component is mostly redundant, the two components have the same core responsibility, and merging them would reduce duplication rather than collapse a clean API/engine, parser/builder, or core/adapter layering.

Return JSON only:
{{
  "approved": true,
  "same_responsibility": true,
  "interface_conflict": false,
  "behavior_conflict": false,
  "risk": "low",
  "reason": "short reason"
}}
""".strip()
        try:
            response = component_merge_agent.llm_client.call_json(
                [
                    {"role": "system", "content": "You are a strict software architecture merge reviewer. Return strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
                operation_name="component_metric_merge_judge",
            )
        except Exception as exc:
            return {
                "approved": False,
                "reason": "review_call_failed",
                "risk": "high",
                "same_responsibility": False,
                "interface_conflict": True,
                "behavior_conflict": True,
                "error": str(exc),
            }
        if not isinstance(response, dict):
            return {
                "approved": False,
                "reason": "review_response_not_object",
                "risk": "high",
                "same_responsibility": False,
                "interface_conflict": True,
                "behavior_conflict": True,
            }
        risk = str(response.get("risk") or "").strip().lower()
        response["approved"] = bool(
            response.get("approved") is True
            and response.get("same_responsibility") is True
            and response.get("interface_conflict") is False
            and response.get("behavior_conflict") is False
            and risk in {"low", "medium"}
        )
        return response

    return _judge


def augment_actions_with_component_metrics(
    *,
    architectures: List[dict],
    actions: List[dict],
    decomposed_dag: RequirementDAG | Dict[str, Any] | None,
    split_cohesion_threshold: float = 2.0 / 3.0,
    split_min_subrequirements: int = 3,
    merge_judge: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    merge_max_small_subrequirements: int = 1,
) -> Tuple[List[dict], Dict[str, Any]]:
    updated = merge_actions_for_architectures([], actions, architectures)
    dag_nodes, dag_edges = _build_decomposed_edge_set(decomposed_dag)
    report: Dict[str, Any] = {
        "stats": {
            "split_upgrades": 0,
            "merge_candidates": 0,
            "merge_upgrades": 0,
        },
        "parents": [],
    }

    by_parent = {
        _parent_from_action_entry(entry): entry
        for entry in updated
        if isinstance(entry, dict) and _parent_from_action_entry(entry)
    }

    for arch in architectures if isinstance(architectures, list) else []:
        if not isinstance(arch, dict):
            continue
        parent = _parent_from_architecture_entry(arch)
        architecture = arch.get("architecture", {}) if isinstance(arch.get("architecture", {}), dict) else {}
        components = architecture.get("components", [])
        if not parent or not isinstance(components, list):
            continue
        action_entry = by_parent.setdefault(parent, {"task": parent, "actions": []})
        component_metrics = [
            {
                "component": dict(component),
                "metrics": _component_metric_payload(component, dag_nodes=dag_nodes, dag_edges=dag_edges),
            }
            for component in components
            if isinstance(component, dict)
        ]
        parent_report: Dict[str, Any] = {
            "parent_task": parent,
            "components": [
                {
                    "name": item["metrics"]["name"],
                    "cohesion": round(float(item["metrics"]["cohesion"]), 6),
                    "served_subrequirements": sorted(item["metrics"]["subrequirements"]),
                    "subrequirement_count": int(item["metrics"]["size"]),
                }
                for item in component_metrics
            ],
            "split_upgrades": [],
            "merge_candidates": [],
            "merge_upgrades": [],
        }

        split_candidates = []
        for item in component_metrics:
            name = item["metrics"]["name"]
            metrics = item["metrics"]
            row = _find_component_action_row(action_entry, name)
            current_action = str(row.get("action") or "save").strip().lower() if isinstance(row, dict) else "save"
            if current_action not in {"save", ""}:
                continue
            if metrics["size"] < max(1, int(split_min_subrequirements)):
                continue
            if float(metrics["cohesion"]) > float(split_cohesion_threshold):
                continue
            split_candidates.append((float(metrics["cohesion"]), -int(metrics["size"]), name, metrics))

        if split_candidates:
            _, _, chosen_name, chosen_metrics = sorted(split_candidates)[0]
            row = _find_component_action_row(action_entry, chosen_name)
            if row is None:
                row = {"component": chosen_name}
                action_entry.setdefault("actions", []).append(row)
            row["action"] = "split"
            row["rationale"] = (
                f"Metric split trigger: cohesion={chosen_metrics['cohesion']:.3f} "
                f"with {chosen_metrics['size']} served subrequirements."
            )
            row["action_origin"] = "metric_split"
            served_subrequirements = sorted(chosen_metrics["subrequirements"])
            induced_edges = [
                {"source": left, "target": right}
                for left, right in sorted(dag_edges)
                if left in chosen_metrics["subrequirements"] and right in chosen_metrics["subrequirements"]
            ]
            row["split_partition_evidence"] = {
                "served_subrequirements": served_subrequirements,
                "induced_edges": induced_edges,
                "cohesion": round(float(chosen_metrics["cohesion"]), 6),
                "internal_edges": int(chosen_metrics["internal_edges"]),
                "possible_internal_edges": float(chosen_metrics["possible_internal_edges"]),
            }
            report["stats"]["split_upgrades"] += 1
            parent_report["split_upgrades"].append(
                {
                    "component": chosen_name,
                    "cohesion": round(float(chosen_metrics["cohesion"]), 6),
                    "subrequirement_count": int(chosen_metrics["size"]),
                    "induced_edges": induced_edges,
                }
            )

        merge_candidates: List[Dict[str, Any]] = []
        for source in component_metrics:
            source_name = source["metrics"]["name"]
            source_row = _find_component_action_row(action_entry, source_name)
            source_action = str(source_row.get("action") or "save").strip().lower() if isinstance(source_row, dict) else "save"
            if source_action not in {"save", ""}:
                continue
            source_set = source["metrics"]["subrequirements"]
            if not source_set:
                continue
            for target in component_metrics:
                if source is target:
                    continue
                target_name = target["metrics"]["name"]
                target_row = _find_component_action_row(action_entry, target_name)
                target_action = str(target_row.get("action") or "save").strip().lower() if isinstance(target_row, dict) else "save"
                if target_action not in {"save", ""}:
                    continue
                target_set = target["metrics"]["subrequirements"]
                cross = sum(
                    1
                    for left, right in dag_edges
                    if (left in source_set and right in target_set) or (left in target_set and right in source_set)
                )
                if cross < 1:
                    continue
                jaccard = _safe_jaccard(source_set, target_set)
                if jaccard < 0.7:
                    continue
                layer_penalty = _metric_merge_layer_penalty(source_name, target_name)
                coupling = float(jaccard * layer_penalty)
                merge_candidates.append(
                    {
                        "parent_task": parent,
                        "source_component": source_name,
                        "target_component": target_name,
                        "source_subrequirements": set(source_set),
                        "target_subrequirements": set(target_set),
                        "source_responsibilities": source["component"].get("responsibilities", []),
                        "target_responsibilities": target["component"].get("responsibilities", []),
                        "source_cohesion": float(source["metrics"]["cohesion"]),
                        "target_cohesion": float(target["metrics"]["cohesion"]),
                        "cross_edges": cross,
                        "jaccard": float(jaccard),
                        "layer_penalty": float(layer_penalty),
                        "coupling": float(coupling),
                        "reason": (
                            f"Metric merge candidate: Jaccard={jaccard:.3f}, "
                            f"layer_penalty={layer_penalty:.3f}, coupling={coupling:.3f}, "
                            f"cross_edges={cross}."
                        ),
                    }
                )
        merge_candidates.sort(
            key=lambda row: (-float(row["coupling"]), -float(row["jaccard"]), -int(row["cross_edges"]), row["source_component"], row["target_component"])
        )
        report["stats"]["merge_candidates"] += len(merge_candidates)
        parent_report["merge_candidates"] = [
            {
                "source_component": row["source_component"],
                "target_component": row["target_component"],
                "jaccard": row["jaccard"],
                "layer_penalty": row["layer_penalty"],
                "coupling": row["coupling"],
                "cross_edges": row["cross_edges"],
            }
            for row in merge_candidates
        ]
        if merge_candidates and merge_judge is not None:
            candidate = merge_candidates[0]
            review = merge_judge(candidate)
            parent_report["merge_judge_review"] = review
            if isinstance(review, dict) and review.get("approved") is True:
                source_row = _find_component_action_row(action_entry, candidate["source_component"])
                if source_row is None:
                    source_row = {"component": candidate["source_component"]}
                    action_entry.setdefault("actions", []).append(source_row)
                source_row["action"] = "merge"
                source_row["target_component"] = candidate["target_component"]
                source_row["rationale"] = str(review.get("reason") or candidate["reason"])
                source_row["action_origin"] = "metric_merge_judged"
                report["stats"]["merge_upgrades"] += 1
                parent_report["merge_upgrades"].append(
                    {
                        "source_component": candidate["source_component"],
                        "target_component": candidate["target_component"],
                        "coupling": candidate["coupling"],
                    }
                )
        report["parents"].append(parent_report)
        if (
            parent_report["split_upgrades"]
            or parent_report["merge_candidates"]
            or parent_report["merge_upgrades"]
        ):
            logging.info(
                "Metric action parent=%s split_upgrades=%d merge_candidates=%d merge_upgrades=%d",
                parent,
                len(parent_report["split_upgrades"]),
                len(parent_report["merge_candidates"]),
                len(parent_report["merge_upgrades"]),
            )

    merged = []
    order = [_parent_from_architecture_entry(arch) for arch in architectures if isinstance(arch, dict)]
    for parent in order:
        if not parent:
            continue
        merged.append(by_parent.get(parent, {"task": parent, "actions": []}))
    logging.info(
        "Metric action summary: split_upgrades=%d merge_candidates=%d merge_upgrades=%d",
        int(report["stats"]["split_upgrades"]),
        int(report["stats"]["merge_candidates"]),
        int(report["stats"]["merge_upgrades"]),
    )
    return merged, report


def build_tdd_revise_action_report(
    generated_entries: Any,
    *,
    failure_threshold: int = 2,
) -> Dict[str, Any]:
    entries = generated_entries if isinstance(generated_entries, list) else []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        parent = str(entry.get("parent_task") or entry.get("task") or "").strip()
        component = str(entry.get("component") or "").strip()
        if not parent or not component:
            continue
        grouped.setdefault((parent, component), []).append(entry)

    actions_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    candidates: List[Dict[str, Any]] = []
    for (parent, component), history in grouped.items():
        consecutive_semantic_failures = 0
        last_semantic_failure: Dict[str, Any] | None = None
        for entry in history:
            generation_status = str(entry.get("generation_status") or "").strip()
            tdd_passed = entry.get("tdd_passed")
            syntax_passed = bool((entry.get("syntax_postcheck") or {}).get("passed", True))
            compile_passed = bool((entry.get("compile_postcheck") or {}).get("passed", True))
            import_passed = bool((entry.get("import_postcheck") or {}).get("passed", True))
            if generation_status != "retained_after_tdd_failure" and tdd_passed is not False:
                consecutive_semantic_failures = 0
                continue
            if not (syntax_passed and compile_passed and import_passed):
                consecutive_semantic_failures = 0
                continue
            consecutive_semantic_failures += 1
            last_semantic_failure = entry
        if consecutive_semantic_failures < max(1, int(failure_threshold)) or last_semantic_failure is None:
            continue
        last = last_semantic_failure
        rationale_seed = str(last.get("compressed_feedback") or "").strip()
        rationale_tail = rationale_seed.splitlines()[0] if rationale_seed else "Repeated semantic TDD failure."
        action_row = {
            "component": component,
            "action": "revise",
            "rationale": f"Triggered after {int(failure_threshold)} consecutive TDD failures. {rationale_tail}",
            "action_origin": "tdd_revise_threshold",
        }
        actions_by_parent.setdefault(parent, []).append(action_row)
        candidates.append(
            {
                "task": parent,
                "component": component,
                "failure_count": consecutive_semantic_failures,
                "last_feedback": rationale_seed,
            }
        )

    actions = [
        {"task": parent, "actions": rows}
        for parent, rows in sorted(actions_by_parent.items())
    ]
    return {
        "stats": {
            "failure_threshold": int(failure_threshold),
            "revise_candidates": len(candidates),
        },
        "candidates": candidates,
        "actions": actions,
    }


def _filter_revise_only_actions(actions: List[dict]) -> List[dict]:
    filtered: List[dict] = []
    for entry in actions if isinstance(actions, list) else []:
        if not isinstance(entry, dict):
            continue
        rows = entry.get("actions", [])
        if not isinstance(rows, list):
            rows = []
        revise_rows = [
            dict(row)
            for row in rows
            if isinstance(row, dict) and str(row.get("action") or "").strip().lower() == "revise"
        ]
        if revise_rows:
            filtered.append(
                {
                    "task": _parent_from_action_entry(entry),
                    "actions": revise_rows,
                }
            )
    return filtered


def _build_default_empty_actions(architectures: List[dict]) -> List[dict]:
    return merge_actions_for_architectures([], [], architectures)


def _combine_actions_for_architectures(
    architectures: List[dict],
    *action_lists: List[dict],
) -> List[dict]:
    combined_by_parent: Dict[str, Dict[str, dict]] = {}
    for action_list in action_lists:
        for entry in action_list if isinstance(action_list, list) else []:
            if not isinstance(entry, dict):
                continue
            parent = _parent_from_action_entry(entry)
            if not parent:
                continue
            component_rows = combined_by_parent.setdefault(parent, {})
            for row in entry.get("actions", []) if isinstance(entry.get("actions", []), list) else []:
                if not isinstance(row, dict):
                    continue
                component = str(row.get("component") or "").strip()
                if not component:
                    continue
                component_rows[component] = dict(row)

    merged: List[dict] = []
    for arch in architectures:
        parent = _parent_from_architecture_entry(arch)
        if not parent or parent not in combined_by_parent:
            continue
        rows = list(combined_by_parent[parent].values())
        if rows:
            merged.append({"task": parent, "actions": rows})
    return merged


def _rebuild_flattened_architectures(output_dir: Path, architectures: List[dict]) -> None:
    flattened_architectures = []
    for arch_result in architectures:
        if not isinstance(arch_result, dict):
            continue
        architecture = arch_result.get("architecture", {})
        if not isinstance(architecture, dict):
            architecture = {}
        flattened_architectures.append(
            {
                "parent_task": arch_result.get("parent_task"),
                "description": architecture.get("requirement", {}).get("description", ""),
                "components": architecture.get("components", []),
                "component_count": architecture.get("component_count", len(architecture.get("components", []) if isinstance(architecture.get("components", []), list) else [])),
            }
        )
    save_json(flattened_architectures, output_dir / "architectures_flattened.json")


def _refresh_layout_plan_artifacts(
    output_dir: Path,
    architectures: List[dict],
    layout_policy: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    layout_grouping_report_path = output_dir / "layout_grouping_report.json"
    layout_plan_fingerprint = _stable_fingerprint({
        "architectures": architectures,
        "layout_policy": layout_policy,
    })
    existing_layout_grouping_report = load_json_if_exists(layout_grouping_report_path, False)
    package_api_plan_path = output_dir / "package_api_plan.json"
    existing_package_api_plan = load_json_if_exists(package_api_plan_path, False)
    if _artifact_matches_fingerprint(existing_layout_grouping_report, layout_plan_fingerprint) and _artifact_matches_fingerprint(existing_package_api_plan, layout_plan_fingerprint):
        layout_grouping_report = existing_layout_grouping_report
        package_api_plan = existing_package_api_plan
        logging.info(
            "Reusing existing layout/package plan artifacts (fingerprint=%s)",
            layout_plan_fingerprint[:12],
        )
    elif _is_legacy_layout_plan_reusable(existing_layout_grouping_report, existing_package_api_plan):
        layout_grouping_report = _attach_input_fingerprint(existing_layout_grouping_report, layout_plan_fingerprint)
        package_api_plan = _attach_input_fingerprint(existing_package_api_plan, layout_plan_fingerprint)
        logging.info(
            "Reusing legacy layout/package plan artifacts and backfilling fingerprint (%s)",
            layout_plan_fingerprint[:12],
        )
        save_json(layout_grouping_report, layout_grouping_report_path)
        save_json(package_api_plan, package_api_plan_path)
    else:
        layout_grouping_report = _build_canonical_package_grouping(architectures, layout_policy)
        layout_grouping_report = _attach_input_fingerprint(layout_grouping_report, layout_plan_fingerprint)
        save_json(layout_grouping_report, layout_grouping_report_path)

        package_api_plan = _build_package_api_plan(architectures, layout_policy)
        package_api_plan = _attach_input_fingerprint(package_api_plan, layout_plan_fingerprint)
        save_json(package_api_plan, package_api_plan_path)

    if isinstance(layout_grouping_report, dict):
        layout_policy["canonical_packages"] = layout_grouping_report.get("candidate_packages", []) or []
        layout_policy["default_subpackage"] = (
            str(layout_grouping_report.get("default_package", "")).strip() or "core"
        )
        layout_policy["component_package_index"] = (
            layout_grouping_report.get("component_assignment_index", {}) or {}
        )
    package_path_index: Dict[str, str] = {}
    if isinstance(package_api_plan, dict):
        for key, row in (package_api_plan.get("component_index", {}) or {}).items():
            if not isinstance(row, dict):
                continue
            subpath = str(row.get("package_subpath") or row.get("canonical_package") or "").strip().strip("/")
            if subpath:
                package_path_index[key] = subpath
    layout_policy["component_package_path_index"] = package_path_index
    return layout_grouping_report, package_api_plan


def _apply_module_plan_to_layout_policy(
    output_dir: Path,
    architectures: List[dict],
    actions: List[dict],
    layout_policy: Dict[str, Any],
    module_planning_agent: Any,
) -> Dict[str, Any]:
    module_plan_path = output_dir / "module_plan.json"
    if module_planning_agent is None:
        save_json(
            {
                "plans": [],
                "component_package_path_index": {},
                "stats": {"planned_components": 0, "non_generic_subpaths": 0},
            },
            module_plan_path,
        )
        return {}
    module_plan_fingerprint = _stable_fingerprint({
        "architectures": architectures,
        "actions": actions,
        "layout_policy": layout_policy,
    })
    existing_module_plan = load_json_if_exists(module_plan_path, False)
    if _artifact_matches_fingerprint(existing_module_plan, module_plan_fingerprint):
        logging.info(
            "Reusing existing module plan (fingerprint=%s)",
            module_plan_fingerprint[:12],
        )
        module_plan = existing_module_plan
        if isinstance(module_plan, dict):
            layout_policy["component_package_path_index"] = (
                module_plan.get("component_package_path_index", {}) or {}
            )
        return module_plan if isinstance(module_plan, dict) else {}
    if _is_legacy_module_plan_reusable(existing_module_plan):
        logging.info(
            "Reusing legacy module plan and backfilling fingerprint (%s)",
            module_plan_fingerprint[:12],
        )
        module_plan = _attach_input_fingerprint(existing_module_plan, module_plan_fingerprint)
        if isinstance(module_plan, dict):
            layout_policy["component_package_path_index"] = (
                module_plan.get("component_package_path_index", {}) or {}
            )
        save_json(module_plan, module_plan_path)
        return module_plan if isinstance(module_plan, dict) else {}
    module_plan = module_planning_agent.plan_modules(
        architectures=architectures,
        actions=actions,
        layout_policy=layout_policy,
    )
    module_plan = _attach_input_fingerprint(module_plan, module_plan_fingerprint)
    if isinstance(module_plan, dict):
        layout_policy["component_package_path_index"] = (
            module_plan.get("component_package_path_index", {}) or {}
        )
    save_json(module_plan, module_plan_path)
    return module_plan if isinstance(module_plan, dict) else {}


def _apply_module_assignment_to_layout_policy(
    output_dir: Path,
    architectures: List[dict],
    actions: List[dict],
    layout_policy: Dict[str, Any],
    module_plan: Dict[str, Any],
    module_assignment_agent: Any,
) -> Dict[str, Any]:
    module_assignment_path = output_dir / "module_assignment.json"
    if module_assignment_agent is None:
        result = {
            "assignments": [],
            "component_package_path_index": {},
            "stats": {"assigned_components": 0, "non_generic_subpaths": 0},
        }
        save_json(result, module_assignment_path)
        return result
    module_assignment_fingerprint = _stable_fingerprint({
        "architectures": architectures,
        "actions": actions,
        "layout_policy": layout_policy,
        "module_plan": module_plan,
    })
    existing_module_assignment = load_json_if_exists(module_assignment_path, False)
    if _artifact_matches_fingerprint(existing_module_assignment, module_assignment_fingerprint):
        logging.info(
            "Reusing existing module assignment (fingerprint=%s)",
            module_assignment_fingerprint[:12],
        )
        module_assignment = existing_module_assignment
        if isinstance(module_assignment, dict):
            layout_policy["component_package_path_index"] = (
                module_assignment.get("component_package_path_index", {}) or {}
            )
        return module_assignment if isinstance(module_assignment, dict) else {}
    if _is_legacy_module_assignment_reusable(existing_module_assignment):
        logging.info(
            "Reusing legacy module assignment and backfilling fingerprint (%s)",
            module_assignment_fingerprint[:12],
        )
        module_assignment = _attach_input_fingerprint(existing_module_assignment, module_assignment_fingerprint)
        if isinstance(module_assignment, dict):
            layout_policy["component_package_path_index"] = (
                module_assignment.get("component_package_path_index", {}) or {}
            )
        save_json(module_assignment, module_assignment_path)
        return module_assignment if isinstance(module_assignment, dict) else {}
    module_assignment = module_assignment_agent.assign_modules(
        architectures=architectures,
        actions=actions,
        layout_policy=layout_policy,
        module_plan=module_plan,
    )
    module_assignment = _attach_input_fingerprint(module_assignment, module_assignment_fingerprint)
    if isinstance(module_assignment, dict):
        layout_policy["component_package_path_index"] = (
            module_assignment.get("component_package_path_index", {}) or {}
        )
    save_json(module_assignment, module_assignment_path)
    return module_assignment if isinstance(module_assignment, dict) else {}


def _action_entry_is_complete(entry: dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    parent = _parent_from_action_entry(entry)
    if not parent:
        return False
    actions = entry.get("actions")
    return isinstance(actions, list)


def detect_completed_action_parents(
    actions: Any,
    active_parents: set[str],
) -> set[str]:
    if not isinstance(actions, list):
        return set()
    completed: set[str] = set()
    for action in actions:
        if not _action_entry_is_complete(action):
            continue
        parent = _parent_from_action_entry(action)
        if parent in active_parents:
            completed.add(parent)
    return completed


def detect_completed_component_merge_parents(
    merge_report: Any,
    active_parents: set[str],
) -> set[str]:
    if not isinstance(merge_report, dict):
        return set()
    parents = merge_report.get("parents", [])
    if not isinstance(parents, list):
        return set()
    completed: set[str] = set()
    for entry in parents:
        if not isinstance(entry, dict):
            continue
        parent = str(entry.get("parent_task") or "").strip()
        if parent and parent in active_parents:
            completed.add(parent)
    return completed


def _parent_from_generated_entry(entry: dict[str, Any]) -> str:
    return str(entry.get("parent_task") or entry.get("task") or "").strip()


def filter_generated_files_for_active_parents(
    generated_entries: Any,
    active_parents: set[str],
) -> List[dict]:
    """Keep generated file entries whose parent still exists in current DAG."""
    if not isinstance(generated_entries, list):
        return []
    filtered: List[dict] = []
    for entry in generated_entries:
        if not isinstance(entry, dict):
            continue
        parent = _parent_from_generated_entry(entry)
        if not parent or parent not in active_parents:
            continue
        normalized = dict(entry)
        normalized["parent_task"] = parent
        filtered.append(normalized)
    return filtered


def _generated_entry_component_key(entry: dict[str, Any]) -> str:
    parent = _parent_from_generated_entry(entry)
    component = str(entry.get("component") or "").strip()
    if not parent or not component:
        return ""
    return f"{parent}::{component}"


def _generated_entry_has_empty_files(entry: dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    files = entry.get("files")
    return not isinstance(files, dict) or not bool(files)


def filter_architectures_for_component_keys(
    architectures: Any,
    component_keys: set[str],
) -> List[dict]:
    """Keep only architecture components matching parent::component keys."""
    if not isinstance(architectures, list) or not component_keys:
        return []
    filtered_architectures: List[dict] = []
    for arch in architectures:
        if not isinstance(arch, dict):
            continue
        parent = _parent_from_architecture_entry(arch)
        if not parent:
            continue
        architecture = arch.get("architecture", {})
        if not isinstance(architecture, dict):
            continue
        components = architecture.get("components", [])
        if not isinstance(components, list):
            continue
        filtered_components = [
            dict(component)
            for component in components
            if isinstance(component, dict)
            and f"{parent}::{str(component.get('name') or '').strip()}" in component_keys
        ]
        if not filtered_components:
            continue
        arch_copy = dict(arch)
        architecture_copy = dict(architecture)
        architecture_copy["components"] = filtered_components
        architecture_copy["component_count"] = len(filtered_components)
        arch_copy["architecture"] = architecture_copy
        filtered_architectures.append(arch_copy)
    return filtered_architectures


def prune_memory_components_for_active_parents(memory_agent: Any, active_parents: set[str]) -> int:
    """Remove component-memory records whose requirement parent no longer exists."""
    snapshot = getattr(memory_agent, "snapshot", None)
    if snapshot is None:
        return 0

    implemented_components = getattr(snapshot, "implemented_components", None)
    if not isinstance(implemented_components, dict):
        return 0

    stale_keys: List[str] = []
    for key, impl in implemented_components.items():
        requirement_node = str(getattr(impl, "requirement_node", "") or "").strip()
        parent = requirement_node or (key.split("::", 1)[0] if isinstance(key, str) else "")
        if parent in active_parents:
            continue
        stale_keys.append(key)

    for key in stale_keys:
        implemented_components.pop(key, None)

    return len(stale_keys)


def summarize_evolution_operations(
    operation_records: List[dict[str, Any]],
    active_parents: set[str] | None = None,
) -> dict[str, Any]:
    """
    Summarize operation records into regeneration scopes and source-target mapping.

    Returns:
    - regen_parents: active parents that should be regenerated
    - removed_parents: parents removed by evolution
    - source_parents_by_target: mapping target_parent -> source parent names
    """
    regen_parents: set[str] = set()
    removed_parents: set[str] = set()
    source_parents_by_target: dict[str, set[str]] = {}

    def _add_source(target: str, source: str) -> None:
        if not target or not source:
            return
        source_parents_by_target.setdefault(target, set()).add(source)

    for record in operation_records:
        if not isinstance(record, dict):
            continue
        op_type = str(record.get("operation_type") or "").lower()
        details = record.get("details") or {}
        if not isinstance(details, dict):
            details = {}

        if op_type in {"add", "create"}:
            affected = record.get("affected_nodes") or []
            for node in affected:
                node_name = str(node).strip()
                if node_name:
                    regen_parents.add(node_name)
            continue

        if op_type == "split":
            original = str(details.get("original") or "").strip()
            created = [str(item).strip() for item in details.get("created", []) if str(item).strip()]
            removed_parents.update([original] if original else [])
            regen_parents.update(created)
            for target in created:
                _add_source(target, original)
            continue

        if op_type == "merge":
            merged_to = str(details.get("merged_to") or "").strip()
            merged_from = [str(item).strip() for item in details.get("merged_from", []) if str(item).strip()]
            if merged_to:
                regen_parents.add(merged_to)
            removed_parents.update(merged_from)
            for source in merged_from:
                _add_source(merged_to, source)
            continue

        if op_type == "revise":
            original = str(details.get("original") or "").strip()
            revised_to = str(details.get("revised_to") or "").strip()
            if revised_to:
                regen_parents.add(revised_to)
            if original:
                removed_parents.add(original)
            _add_source(revised_to, original)
            continue

        if op_type == "delete":
            deleted = [str(item).strip() for item in details.get("deleted", []) if str(item).strip()]
            added = str(details.get("added") or "").strip()
            removed_parents.update(deleted)
            if added:
                regen_parents.add(added)
                for source in deleted:
                    _add_source(added, source)
            continue

    if active_parents is not None:
        regen_parents = {name for name in regen_parents if name in active_parents}
        source_parents_by_target = {
            target: {source for source in sources if source}
            for target, sources in source_parents_by_target.items()
            if target in regen_parents
        }

    return {
        "regen_parents": regen_parents,
        "removed_parents": removed_parents,
        "source_parents_by_target": source_parents_by_target,
    }


def build_parent_component_index(architectures: Any) -> dict[str, List[dict]]:
    """Build parent->components index from architecture artifacts."""
    index: dict[str, List[dict]] = {}
    if not isinstance(architectures, list):
        return index
    for arch in architectures:
        if not isinstance(arch, dict):
            continue
        parent = _parent_from_architecture_entry(arch)
        if not parent:
            continue
        components = arch.get("architecture", {}).get("components", [])
        if not isinstance(components, list):
            continue
        index[parent] = [dict(comp) for comp in components if isinstance(comp, dict)]
    return index


def dedupe_components_by_name(components: List[dict]) -> List[dict]:
    """Dedupe component list by component name while preserving first occurrence."""
    seen: set[str] = set()
    deduped: List[dict] = []
    for comp in components:
        if not isinstance(comp, dict):
            continue
        name = str(comp.get("name") or "").strip().lower()
        key = name or json.dumps(comp, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(comp)
    return deduped


def select_generated_entries_for_parents(
    generated_entries: Any,
    parent_scope: set[str],
) -> List[dict]:
    """Select generated-file entries whose parent is inside provided scope."""
    if not isinstance(generated_entries, list):
        return []
    selected: List[dict] = []
    for entry in generated_entries:
        if not isinstance(entry, dict):
            continue
        parent = _parent_from_generated_entry(entry)
        if not parent or parent not in parent_scope:
            continue
        normalized = dict(entry)
        normalized["parent_task"] = parent
        selected.append(normalized)
    return selected


def _code_file_has_unimplemented_tdd_placeholder(code_path: Path) -> bool:
    """Return True when a generated Python file still contains explicit TDD placeholders."""
    if code_path.suffix != ".py" or not code_path.exists():
        return False
    try:
        content = code_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return (
        'NotImplementedError("TDD")' in content
        or "NotImplementedError('TDD')" in content
    )


def _tokenize_responsibility_text(text: str) -> List[str]:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(text or ""))
    ]
    stop_words = {
        "the", "and", "for", "with", "from", "into", "onto", "that", "this",
        "using", "used", "via", "when", "where", "while", "across", "under",
        "over", "each", "main", "public", "api", "support", "provide",
        "returns", "return", "result", "results", "data", "model", "models",
        "component", "components", "service", "services", "library", "manager",
        "engine", "adapter", "adapters", "system", "systems", "utils",
        "utilities", "core", "layer", "layers",
    }
    return [tok for tok in tokens if tok not in stop_words]


def _code_file_has_weak_responsibility_realization(
    code_path: Path,
    responsibilities: Any,
    component_name: str = "",
) -> bool:
    return bool(
        _find_weak_responsibility_realization_gaps(
            code_path,
            responsibilities,
            component_name=component_name,
        )
    )


def _find_weak_responsibility_realization_gaps(
    code_path: Path,
    responsibilities: Any,
    component_name: str = "",
) -> List[str]:
    if code_path.suffix != ".py" or not code_path.exists():
        return []
    if not isinstance(responsibilities, list) or not responsibilities:
        return []
    try:
        content = code_path.read_text(encoding="utf-8")
    except Exception:
        return []

    haystack = content.lower()
    component_tokens = set(_tokenize_responsibility_text(component_name))
    gaps: List[str] = []
    for responsibility in responsibilities:
        resp_text = str(responsibility or "").strip()
        if not resp_text:
            continue
        resp_tokens = [
            tok for tok in _tokenize_responsibility_text(resp_text)
            if tok not in component_tokens
        ]
        if not resp_tokens:
            continue
        strong_tokens = [tok for tok in resp_tokens if len(tok) >= 5 or tok.isupper()]
        probe_tokens = strong_tokens[:4] or resp_tokens[:3]
        matched = sum(
            1 for tok in probe_tokens
            if tok in haystack or tok.replace("_", " ") in haystack
        )
        min_required = 2 if len(probe_tokens) >= 3 else 1
        if matched < min_required:
            gaps.append(resp_text)
    return gaps


def _format_signature_from_ast(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arg_names = [arg.arg for arg in node.args.args]
        if node.args.vararg:
            arg_names.append(f"*{node.args.vararg.arg}")
        elif node.args.kwonlyargs:
            arg_names.append("*")
        arg_names.extend(arg.arg for arg in node.args.kwonlyargs)
        if node.args.kwarg:
            arg_names.append(f"**{node.args.kwarg.arg}")
        return f"def {node.name}({', '.join(arg_names)})"
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    return ""


def _extract_signature_feedback(code_path: Path, *, max_items: int = 8) -> List[str]:
    if code_path.suffix != ".py" or not code_path.exists():
        return []
    try:
        source = code_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return []

    signatures: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_sig = _format_signature_from_ast(node)
            if class_sig:
                signatures.append(class_sig)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                    method_sig = _format_signature_from_ast(child)
                    if method_sig:
                        signatures.append(f"{node.name}.{method_sig}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            fn_sig = _format_signature_from_ast(node)
            if fn_sig:
                signatures.append(fn_sig)
        if len(signatures) >= max_items:
            break
    return signatures[:max_items]


def _build_compressed_feedback(
    *,
    reasons: List[str],
    planned_file_path: str,
    export_symbols: List[str],
    signatures: List[str],
    responsibilities: List[str],
    char_budget: int = 1200,
) -> str:
    sections: List[str] = []

    def append_section(title: str, items: List[str], *, prefix: str = "- ") -> None:
        nonlocal sections
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        if not cleaned:
            return
        candidate = title + "\n" + "\n".join(f"{prefix}{item}" for item in cleaned)
        prospective = "\n\n".join(sections + [candidate]).strip()
        if len(prospective) <= char_budget:
            sections.append(candidate)
            return
        kept: List[str] = []
        for item in cleaned:
            trial = title + "\n" + "\n".join(f"{prefix}{x}" for x in kept + [item])
            prospective = "\n\n".join(sections + [trial]).strip()
            if len(prospective) > char_budget:
                break
            kept.append(item)
        if kept:
            sections.append(title + "\n" + "\n".join(f"{prefix}{item}" for item in kept))

    append_section("Previous attempt failed these checks:", reasons[:4])
    if planned_file_path.strip():
        append_section("Path constraint:", [f"Keep planned file path unchanged: {planned_file_path.strip()}"])
    append_section("Required exported symbols:", export_symbols[:6])
    append_section("Existing public signatures from the failed attempt:", signatures[:8])
    append_section("Prioritize these responsibilities:", responsibilities[:4])
    return "\n\n".join(sections).strip()


def _resolve_generated_entry_code_path(entry: dict[str, Any]) -> Path | None:
    files = entry.get("files", {})
    if not isinstance(files, dict):
        return None
    code_path_raw = str(files.get("code", "")).strip()
    if not code_path_raw:
        return None
    return Path(code_path_raw).resolve()


def _find_generated_code_root(code_path: Path) -> Path:
    for parent in [code_path.parent, *code_path.parents]:
        if parent.name == "generated_code":
            return parent
    return code_path.parent


def _run_python_module_import_postcheck(repo_root: Path, module_name: str) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "").strip(os.pathsep)
    script = """import importlib
import sys
import traceback

repo_root = sys.argv[1]
module_name = sys.argv[2]
sys.path.insert(0, repo_root)
importlib.invalidate_caches()

try:
    importlib.import_module(module_name)
    print("POSTCHECK_OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, str(repo_root), module_name],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"Import postcheck timed out after 60s: {exc}"
    output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return proc.returncode == 0, output


def _evaluate_generated_entry_import_postcheck(entry: dict[str, Any]) -> dict[str, Any]:
    code_path = _resolve_generated_entry_code_path(entry)
    if code_path is None:
        return {
            "passed": False,
            "code_file": "",
            "module": "",
            "error": "No generated code file was recorded for import postcheck.",
        }
    if not code_path.exists():
        return {
            "passed": False,
            "code_file": str(code_path),
            "module": "",
            "error": f"Recorded code file is missing on disk: {code_path}",
        }
    if code_path.suffix != ".py":
        return {
            "passed": True,
            "code_file": str(code_path),
            "module": "",
            "error": "",
        }

    repo_root = _find_generated_code_root(code_path)
    module_name = str(entry.get("module_path") or "").strip()
    if not module_name:
        try:
            rel_path = str(code_path.relative_to(repo_root)).replace("\\", "/")
        except Exception:
            rel_path = str(entry.get("planned_file_path") or "").strip()
        module_name = _module_from_relative_py_path(rel_path)

    ok, output = _run_python_module_import_postcheck(repo_root, module_name)
    return {
        "passed": ok,
        "code_file": str(code_path),
        "module": module_name,
        "error": "" if ok else output[-4000:],
    }


def _evaluate_generated_entry_syntax_postcheck(entry: dict[str, Any]) -> dict[str, Any]:
    existing = entry.get("syntax_postcheck")
    if isinstance(existing, dict) and "passed" in existing:
        return {
            "passed": bool(existing.get("passed")),
            "code_file": str(existing.get("code_file") or ""),
            "error": str(existing.get("error") or ""),
        }
    code_path = _resolve_generated_entry_code_path(entry)
    if code_path is None:
        return {"passed": False, "code_file": "", "error": "No generated code file was recorded for syntax postcheck."}
    if not code_path.exists():
        return {"passed": False, "code_file": str(code_path), "error": f"Recorded code file is missing on disk: {code_path}"}
    if code_path.suffix != ".py":
        return {"passed": True, "code_file": str(code_path), "error": ""}
    try:
        ast.parse(code_path.read_text(encoding="utf-8"), filename=str(code_path))
        return {"passed": True, "code_file": str(code_path), "error": ""}
    except SyntaxError as exc:
        return {"passed": False, "code_file": str(code_path), "error": str(exc)}


def _evaluate_generated_entry_compile_postcheck(entry: dict[str, Any]) -> dict[str, Any]:
    existing = entry.get("compile_postcheck")
    if isinstance(existing, dict) and "passed" in existing:
        return {
            "passed": bool(existing.get("passed")),
            "code_file": str(existing.get("code_file") or ""),
            "error": str(existing.get("error") or ""),
        }
    code_path = _resolve_generated_entry_code_path(entry)
    if code_path is None:
        return {"passed": False, "code_file": "", "error": "No generated code file was recorded for compile postcheck."}
    if not code_path.exists():
        return {"passed": False, "code_file": str(code_path), "error": f"Recorded code file is missing on disk: {code_path}"}
    if code_path.suffix != ".py":
        return {"passed": True, "code_file": str(code_path), "error": ""}
    try:
        py_compile.compile(str(code_path), doraise=True)
        return {"passed": True, "code_file": str(code_path), "error": ""}
    except py_compile.PyCompileError as exc:
        return {"passed": False, "code_file": str(code_path), "error": str(exc)}


def _evaluate_generated_entry_status(entry: dict[str, Any]) -> dict[str, Any]:
    component = str(entry.get("component") or "").strip()
    parent = _parent_from_generated_entry(entry) or str(entry.get("task") or "").strip()
    files = entry.get("files", {})
    code_path = Path(str(files.get("code", ""))) if isinstance(files, dict) and files.get("code") else None
    reasons: List[str] = []
    generation_status = str(entry.get("generation_status") or "").strip()
    tdd_passed = entry.get("tdd_passed")
    tdd_final_pytest_rc = entry.get("tdd_final_pytest_rc")

    if code_path is None:
        reasons.append("No generated code file was recorded for the previous attempt.")
    elif not code_path.exists():
        reasons.append(f"Recorded code file is missing on disk: {code_path}")
    if generation_status == "retained_after_tdd_failure" or tdd_passed is False:
        reasons.append(
            "Previous attempt was retained on disk after TDD failure"
            + (f" (final_pytest_rc={tdd_final_pytest_rc})" if tdd_final_pytest_rc is not None else ".")
        )

    weak_realization: List[str] = []
    structured_contract_issues: List[str] = []
    signature_feedback: List[str] = []
    syntax_postcheck = _evaluate_generated_entry_syntax_postcheck(entry)
    compile_postcheck = _evaluate_generated_entry_compile_postcheck(entry)
    import_postcheck = _evaluate_generated_entry_import_postcheck(entry)
    if code_path is not None and code_path.exists():
        if _code_file_has_unimplemented_tdd_placeholder(code_path):
            reasons.append("Previous attempt left explicit TDD placeholders in the code file.")
        weak_realization = _find_weak_responsibility_realization_gaps(
            code_path,
            entry.get("component_responsibilities", []),
            component_name=component,
        )
        try:
            structured_contract_issues = find_structured_contract_issues(
                code_path.read_text(encoding="utf-8"),
                str(code_path),
            )
        except Exception:
            structured_contract_issues = []
        signature_feedback = _extract_signature_feedback(code_path)
        if weak_realization:
            reasons.append(
                "Responsibilities weakly realized: "
                + "; ".join(weak_realization[:3])
                + (" ..." if len(weak_realization) > 3 else "")
            )
        if structured_contract_issues:
            reasons.append(
                "Structured state-contract issues detected: "
                + "; ".join(structured_contract_issues[:2])
                + (" ..." if len(structured_contract_issues) > 2 else "")
            )
    if not syntax_postcheck.get("passed"):
        error_text = str(syntax_postcheck.get("error") or "").strip()
        reasons.append(
            "Syntax postcheck failed: "
            + (error_text[:300] + ("..." if len(error_text) > 300 else ""))
        )
    if not compile_postcheck.get("passed"):
        error_text = str(compile_postcheck.get("error") or "").strip()
        reasons.append(
            "Compile postcheck failed: "
            + (error_text[:300] + ("..." if len(error_text) > 300 else ""))
        )
    if not import_postcheck.get("passed"):
        module_name = str(import_postcheck.get("module") or "").strip()
        error_text = str(import_postcheck.get("error") or "").strip()
        if module_name:
            reasons.append(
                f"Import postcheck failed for module '{module_name}': "
                + (error_text[:300] + ("..." if len(error_text) > 300 else ""))
            )
        else:
            reasons.append(
                "Import postcheck failed: "
                + (error_text[:300] + ("..." if len(error_text) > 300 else ""))
            )

    passed = not reasons
    responsibilities = [
        str(resp).strip()
        for resp in (entry.get("component_responsibilities", []) if isinstance(entry.get("component_responsibilities", []), list) else [])
        if str(resp).strip()
    ]
    export_symbols = [
        str(sym).strip()
        for sym in (entry.get("component_export_symbols", []) if isinstance(entry.get("component_export_symbols", []), list) else [])
        if str(sym).strip()
    ]
    compressed_feedback = _build_compressed_feedback(
        reasons=reasons,
        planned_file_path=str(entry.get("planned_file_path") or ""),
        export_symbols=export_symbols,
        signatures=signature_feedback,
        responsibilities=responsibilities,
        char_budget=1200,
    )

    return {
        "task": parent,
        "component": component,
        "code_file": str(code_path) if code_path else "",
        "planned_file_path": str(entry.get("planned_file_path") or ""),
        "generation_status": generation_status or ("implemented" if passed else "failed"),
        "tdd_passed": False if generation_status == "retained_after_tdd_failure" else bool(passed if tdd_passed is None else tdd_passed),
        "tdd_final_pytest_rc": tdd_final_pytest_rc,
        "passed": passed,
        "reasons": reasons,
        "weak_responsibility_realization": weak_realization,
        "structured_contract_issues": structured_contract_issues,
        "syntax_postcheck": syntax_postcheck,
        "compile_postcheck": compile_postcheck,
        "import_postcheck": import_postcheck,
        "signature_feedback": signature_feedback,
        "compressed_feedback": compressed_feedback,
    }


def _persist_component_realization_report(output_dir: Path, generated_entries: Any) -> dict[str, Any]:
    entries = generated_entries if isinstance(generated_entries, list) else []
    evaluations = [
        _evaluate_generated_entry_status(entry)
        for entry in entries
        if isinstance(entry, dict)
    ]
    report = {
        "total_components": len(evaluations),
        "passed_components": sum(1 for item in evaluations if item.get("passed")),
        "failed_components": sum(1 for item in evaluations if not item.get("passed")),
        "components": evaluations,
    }
    save_json(report, output_dir / "component_realization_report.json")
    return report


def _persist_component_import_postcheck_report(output_dir: Path, generated_entries: Any) -> dict[str, Any]:
    entries = generated_entries if isinstance(generated_entries, list) else []
    evaluations = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        parent = _parent_from_generated_entry(entry) or str(entry.get("task") or "").strip()
        component = str(entry.get("component") or "").strip()
        postcheck = _evaluate_generated_entry_import_postcheck(entry)
        evaluations.append(
            {
                "task": parent,
                "component": component,
                "code_file": str(postcheck.get("code_file") or ""),
                "module": str(postcheck.get("module") or ""),
                "passed": bool(postcheck.get("passed")),
                "error": str(postcheck.get("error") or ""),
            }
        )
    report = {
        "total_files": len(evaluations),
        "passed_files": sum(1 for item in evaluations if item.get("passed")),
        "failed_files": sum(1 for item in evaluations if not item.get("passed")),
        "files": evaluations,
    }
    save_json(report, output_dir / "component_import_postcheck_report.json")
    return report


def _persist_component_syntax_postcheck_report(output_dir: Path, generated_entries: Any) -> dict[str, Any]:
    entries = generated_entries if isinstance(generated_entries, list) else []
    evaluations = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        parent = _parent_from_generated_entry(entry) or str(entry.get("task") or "").strip()
        component = str(entry.get("component") or "").strip()
        postcheck = _evaluate_generated_entry_syntax_postcheck(entry)
        evaluations.append(
            {
                "task": parent,
                "component": component,
                "code_file": str(postcheck.get("code_file") or ""),
                "passed": bool(postcheck.get("passed")),
                "error": str(postcheck.get("error") or ""),
            }
        )
    report = {
        "total_files": len(evaluations),
        "passed_files": sum(1 for item in evaluations if item.get("passed")),
        "failed_files": sum(1 for item in evaluations if not item.get("passed")),
        "files": evaluations,
    }
    save_json(report, output_dir / "component_syntax_postcheck_report.json")
    return report


def _persist_component_compile_postcheck_report(output_dir: Path, generated_entries: Any) -> dict[str, Any]:
    entries = generated_entries if isinstance(generated_entries, list) else []
    evaluations = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        parent = _parent_from_generated_entry(entry) or str(entry.get("task") or "").strip()
        component = str(entry.get("component") or "").strip()
        postcheck = _evaluate_generated_entry_compile_postcheck(entry)
        evaluations.append(
            {
                "task": parent,
                "component": component,
                "code_file": str(postcheck.get("code_file") or ""),
                "passed": bool(postcheck.get("passed")),
                "error": str(postcheck.get("error") or ""),
            }
        )
    report = {
        "total_files": len(evaluations),
        "passed_files": sum(1 for item in evaluations if item.get("passed")),
        "failed_files": sum(1 for item in evaluations if not item.get("passed")),
        "files": evaluations,
    }
    save_json(report, output_dir / "component_compile_postcheck_report.json")
    return report


def _persist_component_lint_report(
    output_dir: Path,
    generated_entries: Any,
    lint_report: dict[str, Any],
) -> dict[str, Any]:
    entries = generated_entries if isinstance(generated_entries, list) else []
    files_by_path: dict[str, dict[str, Any]] = {}
    for item in lint_report.get("files", []) if isinstance(lint_report, dict) else []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("file") or "").strip()
        if path:
            files_by_path[path] = item

    evaluations = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        parent = _parent_from_generated_entry(entry) or str(entry.get("task") or "").strip()
        component = str(entry.get("component") or "").strip()
        files = entry.get("files", {})
        code_file = str(files.get("code") or "") if isinstance(files, dict) else ""
        test_file = str(files.get("test") or "") if isinstance(files, dict) else ""
        code_lint = files_by_path.get(code_file, {})
        test_lint = files_by_path.get(test_file, {}) if test_file else {}
        evaluations.append(
            {
                "task": parent,
                "component": component,
                "code_file": code_file,
                "test_file": test_file,
                "code_final_ok": bool(code_lint.get("final_ok", True)) if code_file else True,
                "test_final_ok": bool(test_lint.get("final_ok", True)) if test_file else True,
                "code_final_categories": code_lint.get("final_categories", []) if isinstance(code_lint, dict) else [],
                "test_final_categories": test_lint.get("final_categories", []) if isinstance(test_lint, dict) else [],
                "code_fixed_by": code_lint.get("fixed_by") if isinstance(code_lint, dict) else None,
                "test_fixed_by": test_lint.get("fixed_by") if isinstance(test_lint, dict) else None,
            }
        )
    report = {
        "total_components": len(evaluations),
        "checked_files": int(lint_report.get("checked_files", 0)) if isinstance(lint_report, dict) else 0,
        "files_with_issues": int(lint_report.get("files_with_issues", 0)) if isinstance(lint_report, dict) else 0,
        "unresolved": int(lint_report.get("unresolved", 0)) if isinstance(lint_report, dict) else 0,
        "issue_categories": dict(lint_report.get("issue_categories", {})) if isinstance(lint_report, dict) else {},
        "final_issue_categories": dict(lint_report.get("final_issue_categories", {})) if isinstance(lint_report, dict) else {},
        "static_preflight": dict(lint_report.get("static_preflight", {})) if isinstance(lint_report.get("static_preflight", {}), dict) else {},
        "components": evaluations,
    }
    save_json(report, output_dir / "component_lint_report.json")
    return report


def _persist_all_component_reports(output_dir: Path, generated_entries: Any) -> None:
    _persist_component_realization_report(output_dir, generated_entries)
    _persist_component_syntax_postcheck_report(output_dir, generated_entries)
    _persist_component_compile_postcheck_report(output_dir, generated_entries)
    _persist_component_import_postcheck_report(output_dir, generated_entries)


def _repair_existing_import_postcheck_failures(
    *,
    generated_entries: Any,
    code_generator: Any,
    code_output_dir: Path,
    implemented_components_context: str = "",
) -> tuple[list[dict[str, Any]], int]:
    entries = generated_entries if isinstance(generated_entries, list) else []
    repaired = 0
    updated_entries: list[dict[str, Any]] = []
    logging.info(
        "Starting existing generated import postcheck scan: entries=%d root=%s",
        len(entries),
        code_output_dir,
    )
    started_at = time.perf_counter()
    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            continue
        entry_copy = dict(entry)
        parent_name = _parent_from_generated_entry(entry_copy) or str(entry_copy.get("task") or "Unknown")
        component_name = str(entry_copy.get("component") or "Unknown")
        status = _evaluate_generated_entry_import_postcheck(entry_copy)
        logging.info(
            "Existing import postcheck %d/%d: %s::%s module=%s passed=%s",
            idx,
            len(entries),
            parent_name,
            component_name,
            str(status.get("module") or ""),
            bool(status.get("passed")),
        )
        if status.get("passed"):
            entry_copy["import_postcheck"] = status
            updated_entries.append(entry_copy)
            continue

        files = entry_copy.get("files", {})
        if not isinstance(files, dict) or not files.get("code"):
            entry_copy["import_postcheck"] = status
            updated_entries.append(entry_copy)
            continue

        planned_rel_path = str(entry_copy.get("planned_file_path") or "").strip()
        if not planned_rel_path:
            try:
                planned_rel_path = str(
                    Path(str(files.get("code"))).resolve().relative_to(code_output_dir.resolve())
                ).replace("\\", "/")
            except Exception:
                planned_rel_path = ""

        code_result = {
            "component_name": component_name,
            "file_path": planned_rel_path,
        }
        try:
            logging.info(
                "Invoking import-fix agent for existing file %d/%d: %s::%s module=%s",
                idx,
                len(entries),
                parent_name,
                component_name,
                str(status.get("module") or ""),
            )
            postcheck = code_generator.postcheck_saved_component(
                code_result=code_result,
                repo_root=code_output_dir,
                created_files={k: str(v) for k, v in files.items()},
                implemented_components_context=implemented_components_context,
            )
            entry_copy["import_postcheck"] = postcheck
            repaired += 1
            logging.info(
                "Repaired existing import postcheck failure for %s::%s via LLM patch",
                parent_name,
                component_name,
            )
        except Exception as exc:
            entry_copy["import_postcheck"] = {
                "passed": False,
                "code_file": str(files.get("code") or ""),
                "module": str(status.get("module") or ""),
                "error": str(exc),
            }
            logging.warning(
                "Failed to repair existing import postcheck for %s::%s: %s",
                parent_name,
                component_name,
                exc,
            )
        updated_entries.append(entry_copy)
    logging.info(
        "Completed existing generated import postcheck scan: repaired=%d total=%d elapsed=%.2fs",
        repaired,
        len(entries),
        time.perf_counter() - started_at,
    )
    return updated_entries, repaired


def _repair_existing_resume_postcheck_failures(
    *,
    generated_entries: Any,
    code_generator: Any,
    code_output_dir: Path,
    implemented_components_context: str = "",
) -> tuple[list[dict[str, Any]], int]:
    entries = generated_entries if isinstance(generated_entries, list) else []
    repaired = 0
    updated_entries: list[dict[str, Any]] = []
    logging.info(
        "Starting existing generated resume postcheck scan: entries=%d root=%s",
        len(entries),
        code_output_dir,
    )
    started_at = time.perf_counter()
    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            continue
        entry_copy = dict(entry)
        parent_name = _parent_from_generated_entry(entry_copy) or str(entry_copy.get("task") or "Unknown")
        component_name = str(entry_copy.get("component") or "Unknown")
        files = entry_copy.get("files", {})
        code_file = Path(str(files.get("code", ""))) if isinstance(files, dict) and files.get("code") else None

        syntax_status = _evaluate_generated_entry_syntax_postcheck(entry_copy)
        compile_status = _evaluate_generated_entry_compile_postcheck(entry_copy)
        import_status = _evaluate_generated_entry_import_postcheck(entry_copy)
        repaired_this_entry = False

        if code_file is not None and code_file.exists() and code_file.suffix == ".py":
            if not syntax_status.get("passed") or not compile_status.get("passed"):
                try:
                    rel_path = str(code_file.resolve().relative_to(code_output_dir.resolve())).replace("\\", "/")
                except Exception:
                    rel_path = str(entry_copy.get("planned_file_path") or code_file.name)
                original = code_file.read_text(encoding="utf-8")
                repaired_code = code_generator._autofix_python_syntax(  # noqa: SLF001
                    original,
                    component_name,
                    rel_path,
                )
                if repaired_code != original:
                    code_file.write_text(repaired_code, encoding="utf-8")
                    repaired += 1
                    repaired_this_entry = True
                syntax_status = _evaluate_generated_entry_syntax_postcheck(entry_copy)
                compile_status = _evaluate_generated_entry_compile_postcheck(entry_copy)

        entry_copy["syntax_postcheck"] = syntax_status
        entry_copy["compile_postcheck"] = compile_status

        if not import_status.get("passed") and code_file is not None and code_file.exists():
            planned_rel_path = str(entry_copy.get("planned_file_path") or "").strip()
            if not planned_rel_path:
                try:
                    planned_rel_path = str(code_file.resolve().relative_to(code_output_dir.resolve())).replace("\\", "/")
                except Exception:
                    planned_rel_path = ""
            try:
                import_status = code_generator.postcheck_saved_component(
                    code_result={
                        "component_name": component_name,
                        "file_path": planned_rel_path,
                    },
                    repo_root=code_output_dir,
                    created_files={k: str(v) for k, v in files.items()} if isinstance(files, dict) else {},
                    implemented_components_context=implemented_components_context,
                )
                repaired += 1
                repaired_this_entry = True
            except Exception as exc:
                import_status = {
                    "passed": False,
                    "code_file": str(code_file) if code_file is not None else "",
                    "module": str(import_status.get("module") or ""),
                    "error": str(exc),
                }
        entry_copy["import_postcheck"] = import_status

        logging.info(
            "Existing resume postcheck %d/%d: %s::%s syntax=%s compile=%s import=%s repaired=%s",
            idx,
            len(entries),
            parent_name,
            component_name,
            bool(entry_copy["syntax_postcheck"].get("passed")),
            bool(entry_copy["compile_postcheck"].get("passed")),
            bool(entry_copy["import_postcheck"].get("passed")),
            repaired_this_entry,
        )
        updated_entries.append(entry_copy)
    logging.info(
        "Completed existing generated resume postcheck scan: repaired=%d total=%d elapsed=%.2fs",
        repaired,
        len(entries),
        time.perf_counter() - started_at,
    )
    return updated_entries, repaired


def _log_existing_component_realization_summary(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        return
    failed_components = [
        item for item in report.get("components", [])
        if isinstance(item, dict) and not item.get("passed")
    ]
    logging.info(
        "Existing generated component status: passed=%d failed=%d total=%d",
        int(report.get("passed_components", 0)),
        int(report.get("failed_components", 0)),
        int(report.get("total_components", 0)),
    )
    if failed_components:
        preview = [
            f"{str(item.get('task') or 'Unknown')}::{str(item.get('component') or 'Unknown')}"
            for item in failed_components[:20]
        ]
        logging.info(
            "Components scheduled for regeneration due to failed realization checks (%d shown%s): %s",
            len(preview),
            " of " + str(len(failed_components)) if len(failed_components) > len(preview) else "",
            ", ".join(preview),
        )


def _generated_entry_has_usable_code(
    entry: dict[str, Any],
    *,
    rerun_retained_tdd_failures: bool = False,
) -> bool:
    """A generated entry is resumable if its code exists and file-level checks are healthy."""
    if not isinstance(entry, dict):
        return False
    files = entry.get("files", {})
    code_path = Path(str(files.get("code", ""))) if isinstance(files, dict) and files.get("code") else None
    if code_path is None or not code_path.exists():
        return False
    generation_status = str(entry.get("generation_status") or "").strip()
    if generation_status == "retained_after_tdd_failure" and rerun_retained_tdd_failures:
        return False
    syntax_postcheck = _evaluate_generated_entry_syntax_postcheck(entry)
    compile_postcheck = _evaluate_generated_entry_compile_postcheck(entry)
    import_postcheck = _evaluate_generated_entry_import_postcheck(entry)
    return bool(
        syntax_postcheck.get("passed")
        and compile_postcheck.get("passed")
        and import_postcheck.get("passed")
    )


def detect_completed_generated_parents(
    generated_entries: Any,
    architectures: Any,
    *,
    rerun_retained_tdd_failures: bool = False,
) -> set[str]:
    """
    Detect parents whose expected components already have generated code on disk.

    A parent is considered complete when every component declared by its current
    architecture has a corresponding generated entry with an existing code file.
    """
    if not isinstance(generated_entries, list) or not isinstance(architectures, list):
        return set()

    by_parent: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in generated_entries:
        if not isinstance(entry, dict):
            continue
        parent = _parent_from_generated_entry(entry)
        component = str(entry.get("component") or "").strip()
        if not parent or not component:
            continue
        if not _generated_entry_has_usable_code(
            entry,
            rerun_retained_tdd_failures=rerun_retained_tdd_failures,
        ):
            continue
        by_parent.setdefault(parent, {})[component] = entry

    completed: set[str] = set()
    for arch in architectures:
        if not isinstance(arch, dict):
            continue
        parent = _parent_from_architecture_entry(arch)
        if not parent:
            continue
        components = arch.get("architecture", {}).get("components", [])
        if not isinstance(components, list):
            continue
        expected = {
            str(comp.get("name") or "").strip()
            for comp in components
            if isinstance(comp, dict) and str(comp.get("name") or "").strip()
        }
        if not expected:
            continue
        existing = set(by_parent.get(parent, {}).keys())
        if expected.issubset(existing):
            completed.add(parent)
        else:
            missing = sorted(expected - existing)
            if missing:
                logging.info(
                    "Codegen resume: parent '%s' is incomplete; will regenerate missing/unusable components: %s",
                    parent,
                    ", ".join(missing[:20]),
                )
    return completed


def rehydrate_memory_from_generated_artifacts(
    *,
    memory_agent: Any,
    code_generator: Any,
    code_output_dir: Path,
    generated_entries: Any,
    module_registry: Optional[Dict[str, str]] = None,
    parent_scope: Optional[Set[str]] = None,
    rerun_retained_tdd_failures: bool = False,
) -> int:
    """Re-register implemented components from generated artifacts into memory.

    This makes resume robust even when ``memory.json`` is missing or stale.
    """
    if not isinstance(generated_entries, list):
        return 0

    entries = generated_entries
    if parent_scope:
        entries = select_generated_entries_for_parents(entries, parent_scope)

    restored = 0
    module_registry = module_registry if module_registry is not None else {}
    for entry in entries:
        if not _generated_entry_has_usable_code(
            entry,
            rerun_retained_tdd_failures=rerun_retained_tdd_failures,
        ):
            parent = _parent_from_generated_entry(entry)
            component_name = str(entry.get("component") or "").strip()
            if parent or component_name:
                logging.info(
                    "Skipping rehydrate for incomplete generated component '%s' under parent '%s'",
                    component_name or "Unknown",
                    parent or "Unknown",
                )
            continue
        parent = _parent_from_generated_entry(entry)
        component_name = str(entry.get("component") or "").strip()
        files = entry.get("files", {})
        if not parent or not component_name or not isinstance(files, dict):
            continue
        code_path = Path(str(files.get("code", "")))
        if not code_path.exists():
            continue
        try:
            code_content = code_path.read_text(encoding="utf-8")
            rel_path = str(code_path.relative_to(code_output_dir))
            metadata = code_generator.extract_component_metadata(
                {
                    "component_name": component_name,
                    "file_path": rel_path,
                    "code": code_content,
                },
                requirement_node=parent,
            )
            memory_agent.register_component_implementation(**metadata)
            module_path = _module_from_relative_py_path(rel_path)
            if module_path:
                module_registry[component_name] = module_path
            restored += 1
        except Exception as exc:
            logging.warning(
                "Failed to rehydrate generated component '%s' for parent '%s': %s",
                component_name,
                parent,
                exc,
            )
    if restored:
        logging.info(
            "Rehydrated %d implemented components into memory from generated artifacts",
            restored,
        )
    return restored


def apply_cross_requirement_component_merge(
    architectures: List[dict],
    component_merge_agent: Any,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Deduplicate architecture components across requirement parents."""
    parent_order: List[str] = []
    flat_components: List[dict] = []
    components_by_parent: Dict[str, List[dict]] = {}
    arch_by_parent: Dict[str, dict] = {}

    for arch_idx, arch in enumerate(architectures if isinstance(architectures, list) else []):
        if not isinstance(arch, dict):
            continue
        parent = _parent_from_architecture_entry(arch) or f"parent_{arch_idx}"
        parent_order.append(parent)
        arch_by_parent[parent] = arch
        architecture = arch.get("architecture", {})
        components = architecture.get("components", []) if isinstance(architecture, dict) else []
        if not isinstance(components, list):
            components = []
        components_by_parent[parent] = components
        for comp_idx, component in enumerate(components):
            if not isinstance(component, dict):
                continue
            row = dict(component)
            row["_cross_requirement_parent"] = parent
            row["_cross_requirement_component_index"] = comp_idx
            flat_components.append(row)

    if len(flat_components) < 2:
        return architectures, {
            "enabled": True,
            "scope": "cross_requirement",
            "input_count": len(flat_components),
            "output_count": len(flat_components),
            "applied": False,
            "merge_groups": [],
            "stats": {
                "components_before": len(flat_components),
                "components_after": len(flat_components),
                "merged_component_count": 0,
                "accepted_group_count": 0,
                "cross_parent_group_count": 0,
                "parents_with_merge": 0,
            },
        }

    merged_arch, flat_report = component_merge_agent.merge_architecture_components(
        parent_task="__cross_requirement_component_merge__",
        architecture={"components": flat_components},
    )
    merged_components = merged_arch.get("components", [])
    if not isinstance(merged_components, list):
        merged_components = flat_components

    redistributed: Dict[str, List[dict]] = {parent: [] for parent in parent_order}
    for component in merged_components:
        if not isinstance(component, dict):
            continue
        parent = str(component.get("_cross_requirement_parent") or "").strip()
        if parent not in redistributed:
            parent = parent_order[0] if parent_order else ""
        clean = dict(component)
        clean.pop("_cross_requirement_parent", None)
        clean.pop("_cross_requirement_component_index", None)
        if parent:
            redistributed.setdefault(parent, []).append(clean)

    updated: List[dict] = []
    parents_with_merge = 0
    for parent in parent_order:
        arch = dict(arch_by_parent[parent])
        architecture = dict(arch.get("architecture", {}) or {})
        before_count = len(components_by_parent.get(parent, []))
        after_components = redistributed.get(parent, [])
        architecture["components"] = after_components
        architecture["component_count"] = len(after_components)
        arch["architecture"] = architecture
        updated.append(arch)
        if len(after_components) < before_count:
            parents_with_merge += 1

    id_to_parent = {
        f"C{idx + 1}": str(component.get("_cross_requirement_parent") or "").strip()
        for idx, component in enumerate(flat_components)
    }
    cross_parent_group_count = 0
    merge_groups = flat_report.get("merge_groups", []) if isinstance(flat_report, dict) else []
    for group in merge_groups if isinstance(merge_groups, list) else []:
        if not isinstance(group, dict):
            continue
        parents = {
            id_to_parent.get(str(source_id).strip(), "")
            for source_id in group.get("source_ids", [])
        }
        parents.discard("")
        group["source_parents"] = sorted(parents)
        if len(parents) > 1:
            cross_parent_group_count += 1

    stats = dict(flat_report.get("stats", {}) if isinstance(flat_report, dict) else {})
    stats.update(
        {
            "components_before": len(flat_components),
            "components_after": len(merged_components),
            "merged_component_count": max(0, len(flat_components) - len(merged_components)),
            "cross_parent_group_count": cross_parent_group_count,
            "parents_with_merge": parents_with_merge,
        }
    )
    report = dict(flat_report) if isinstance(flat_report, dict) else {}
    report.update(
        {
            "enabled": True,
            "scope": "cross_requirement",
            "parent_count": len(parent_order),
            "stats": stats,
        }
    )
    return updated, report


def apply_component_merge_to_architectures(
    architectures: List[dict],
    component_merge_agent: Optional[Any],
    component_split_agent: Optional[Any],
    *,
    apply_merge: bool,
    enable_embedding_analysis: bool,
    enable_cross_requirement_merge: bool = False,
) -> Tuple[List[dict], Dict[str, Any], Optional[Dict[str, Any]]]:
    """Apply component-level merge (LLM) and optional embedding diagnostics on architectures."""
    if not isinstance(architectures, list):
        empty = {
            "enabled": component_merge_agent is not None,
            "parents": [],
            "stats": {
                "parent_count": 0,
                "components_before": 0,
                "components_after": 0,
                "merged_components": 0,
                "parents_with_merge": 0,
            },
        }
        emb_empty = {"enabled": bool(enable_embedding_analysis), "parents": [], "stats": {"parent_count": 0}}
        return [], empty, emb_empty if enable_embedding_analysis else None

    merge_report: Dict[str, Any] = {
        "enabled": bool(apply_merge and component_merge_agent is not None),
        "parents": [],
        "stats": {
            "parent_count": len(architectures),
            "components_before": 0,
            "components_after": 0,
            "merged_components": 0,
            "split_components": 0,
            "parents_with_merge": 0,
            "parents_with_split": 0,
        },
    }
    emb_report: Optional[Dict[str, Any]] = None
    if enable_embedding_analysis:
        emb_report = {
            "enabled": True,
            "parents": [],
            "stats": {
                "parent_count": len(architectures),
                "parents_with_candidates": 0,
                "pair_count": 0,
            },
        }

    if component_merge_agent is None and component_split_agent is None and not enable_embedding_analysis:
        return architectures, merge_report, emb_report

    cross_merge_applied = False
    cross_components_before = 0
    cross_merged_components = 0
    cross_parents_with_merge = 0
    if apply_merge and enable_cross_requirement_merge and component_merge_agent is not None:
        architectures, cross_report = apply_cross_requirement_component_merge(
            architectures,
            component_merge_agent,
        )
        cross_merge_applied = True
        merge_report["cross_requirement_merge"] = cross_report
        cross_components_before = int(cross_report.get("stats", {}).get("components_before", 0) or 0)
        cross_merged_components = int(cross_report.get("stats", {}).get("merged_component_count", 0) or 0)
        cross_parents_with_merge = int(cross_report.get("stats", {}).get("parents_with_merge", 0) or 0)
        apply_merge = False

    updated: List[dict] = []
    final_components_after = 0
    split_components_total = 0
    parents_with_split = 0
    total = len(architectures)
    for idx, arch in enumerate(architectures, start=1):
        if not isinstance(arch, dict):
            continue
        parent_task = _parent_from_architecture_entry(arch) or "Unknown"
        started_at = time.perf_counter()
        architecture_payload = arch.get("architecture", {})
        if not isinstance(architecture_payload, dict):
            architecture_payload = {}
        components_before = architecture_payload.get("components", [])
        before_count = len(components_before) if isinstance(components_before, list) else 0
        if not cross_merge_applied:
            merge_report["stats"]["components_before"] += before_count
        logging.info(
            "Component merge parent %d/%d: %s (components=%d)",
            idx,
            total,
            parent_task,
            before_count,
        )

        arch_copy = dict(arch)
        parent_merge_detail: Dict[str, Any] = {
            "parent_task": parent_task,
            "input_count": before_count,
            "output_count": before_count,
            "applied": False,
            "merge_groups": [],
            "stats": {"merged_component_count": 0},
        }

        if component_merge_agent is not None and apply_merge:
            merged_arch, parent_merge_detail = component_merge_agent.merge_architecture_components(
                parent_task=parent_task,
                architecture=architecture_payload,
            )
            arch_copy["architecture"] = merged_arch
        else:
            arch_copy["architecture"] = architecture_payload

        if component_split_agent is not None:
            split_arch, split_detail = component_split_agent.split_architecture_components(
                parent_task=parent_task,
                architecture=arch_copy.get("architecture", {}),
            )
            arch_copy["architecture"] = split_arch
            if isinstance(split_detail, dict):
                parent_merge_detail["split_groups"] = split_detail.get("split_groups", [])
                parent_merge_detail["split_summary"] = split_detail
                stats = parent_merge_detail.setdefault("stats", {})
                if isinstance(stats, dict):
                    stats["split_group_count"] = split_detail.get("stats", {}).get("split_group_count", 0)
                    stats["split_component_count"] = split_detail.get("stats", {}).get("split_component_count", 0)

        after_components = arch_copy.get("architecture", {}).get("components", [])
        after_count = len(after_components) if isinstance(after_components, list) else 0
        final_components_after += after_count
        merged_num = 0 if cross_merge_applied else max(0, before_count - after_count)
        split_num = max(0, after_count - before_count)
        if not cross_merge_applied:
            merge_report["stats"]["merged_components"] += merged_num
        split_components_total += split_num
        if merged_num > 0:
            merge_report["stats"]["parents_with_merge"] += 1
        if split_num > 0:
            parents_with_split += 1
        merge_report["parents"].append(parent_merge_detail)
        logging.info(
            "Component normalize completed for %s: before=%d after=%d merged=%d split=%d elapsed=%.2fs",
            parent_task,
            before_count,
            after_count,
            merged_num,
            split_num,
            time.perf_counter() - started_at,
        )

        if enable_embedding_analysis and emb_report is not None and component_merge_agent is not None:
            diag = component_merge_agent.embedding_diagnostic(
                parent_task=parent_task,
                architecture=arch_copy.get("architecture", {}),
            )
            emb_report["parents"].append(diag)
            pair_scores = diag.get("pair_scores", [])
            emb_report["stats"]["pair_count"] += len(pair_scores) if isinstance(pair_scores, list) else 0
            clusters = diag.get("clusters_after_policy", [])
            if isinstance(clusters, list) and any(isinstance(group, list) and len(group) >= 2 for group in clusters):
                emb_report["stats"]["parents_with_candidates"] += 1

        updated.append(arch_copy)

    if cross_merge_applied:
        merge_report["stats"]["components_before"] = cross_components_before
        merge_report["stats"]["merged_components"] = cross_merged_components
        merge_report["stats"]["parents_with_merge"] = cross_parents_with_merge
    merge_report["stats"]["components_after"] = final_components_after
    merge_report["stats"]["split_components"] = split_components_total
    merge_report["stats"]["parents_with_split"] = parents_with_split

    return updated, merge_report, emb_report


def group_tasks_by_parent(original_dag: RequirementDAG, decomposed_dag: RequirementDAG, plan: List[dict]) -> List[Tuple[dict, List[dict]]]:
    """
    Group decomposed tasks by their original parent requirements.
    
    Args:
        original_dag: The original requirement DAG (before decomposition)
        decomposed_dag: The decomposed requirement DAG
        plan: List of tasks from the planner
    
    Returns:
        List of (parent_task, [sub_tasks]) tuples where parent_task is from original_dag
        and sub_tasks are decomposed children from decomposed_dag
    """
    # Build mapping from task name to task dict
    task_by_name = {task.get('name', f'task_{i}'): task for i, task in enumerate(plan)}
    
    # Build mapping from original parent name to its decomposed sub-tasks
    # Each decomposed node has metadata['parent'] pointing to its original parent
    parent_to_subtasks = {}
    
    for decomposed_name, decomposed_node in decomposed_dag.nodes.items():
        # Check if this decomposed node has a parent in metadata
        parent_name = decomposed_node.metadata.get('parent')

        if parent_name:
            # This is a decomposed sub-node, group it under its parent
            if parent_name not in parent_to_subtasks:
                parent_to_subtasks[parent_name] = []
            
            # Find corresponding task in plan
            if decomposed_name in task_by_name:
                parent_to_subtasks[parent_name].append(task_by_name[decomposed_name])
        else:
            # This node has no parent metadata, it's likely an original node that wasn't decomposed
            # Initialize it in the mapping to ensure it appears in final groups
            if decomposed_name not in parent_to_subtasks:
                parent_to_subtasks[decomposed_name] = []
    
    # Build parent groups
    parent_groups = []
    processed_task_names = set()
    
    # Process each parent from original_dag
    for parent_name in original_dag.nodes.keys():
        subtasks = parent_to_subtasks.get(parent_name, [])
        
        # Sort subtasks by their order metadata if available
        subtasks_sorted = sorted(
            subtasks, 
            key=lambda t: decomposed_dag.nodes[t.get('name', '')].metadata.get('order', 0) 
            if t.get('name', '') in decomposed_dag.nodes else 0
        )
        
        # Get parent task info from original_dag
        parent_node = original_dag.nodes[parent_name]
        parent_task = task_by_name.get(parent_name)
        
        if not parent_task:
            # Parent not in plan, create task from original node
            parent_task = {
                'name': parent_name,
                'description': parent_node.description,
                'metadata': parent_node.metadata
            }
        
        # Add to groups
        parent_groups.append((parent_task, subtasks_sorted))
        processed_task_names.add(parent_name)
        for subtask in subtasks_sorted:
            processed_task_names.add(subtask.get('name', ''))
    
    # Handle any remaining tasks that weren't grouped (edge case)
    for task in plan:
        task_name = task.get('name', '')
        if task_name not in processed_task_names:
            parent_groups.append((task, []))
    
    # Log statistics
    total_parents = len(parent_groups)
    parents_with_subs = sum(1 for _, subs in parent_groups if len(subs) > 0)
    total_subs = sum(len(subs) for _, subs in parent_groups)
    
    # Print aggregated parent node names
    parent_names = [parent.get('name', 'Unknown') for parent, _ in parent_groups]
    logging.info(f"Aggregated parent node names ({len(parent_names)} total):")
    for i, name in enumerate(parent_names, 1):
        logging.info(f"  {i:2d}. {name}")
    
    # logging.info(parent_groups)
    logging.info(f"Grouped {len(plan)} tasks into {total_parents} parent groups:")
    logging.info(f"  - {parents_with_subs} parents with decomposed sub-tasks ({total_subs} sub-tasks)")
    logging.info(f"  - {total_parents - parents_with_subs} parents without decomposition")
    
    return parent_groups


def get_task_parent_name(task_name: str, decomposed_dag: RequirementDAG) -> str:
    if task_name in decomposed_dag.nodes:
        parent = decomposed_dag.nodes[task_name].metadata.get("parent")
        if parent:
            return parent
    if "::" in task_name:
        return task_name.split("::", 1)[0]
    return task_name


def build_parent_filtered_dag(decomposed_dag: RequirementDAG, parents: set[str]) -> RequirementDAG:
    nodes = {
        name: node
        for name, node in decomposed_dag.nodes.items()
        if node.metadata.get("parent") in parents or name in parents
    }
    adjacency = {
        name: {child for child in decomposed_dag.adjacency.get(name, set()) if child in nodes}
        for name in nodes
    }
    return RequirementDAG(nodes, adjacency)


def build_parent_codegen_layers(
    dag: RequirementDAG,
    ordered_parents: List[str],
) -> List[List[str]]:
    """Build conservative topological layers for parent-level code generation.

    Parents in the same layer are guaranteed to have no dependency edges between
    them inside the parent requirement DAG. Layers are emitted in original order
    as much as possible. Any cycle falls back to a final sequential layer.
    """
    ordered = [str(parent).strip() for parent in ordered_parents if str(parent).strip()]
    if not ordered:
        return []

    order_index = {name: idx for idx, name in enumerate(ordered)}
    parent_scope = set(ordered)
    filtered_dag = build_parent_filtered_dag(dag, parent_scope)

    adjacency: Dict[str, Set[str]] = {name: set() for name in ordered}
    indegree: Dict[str, int] = {name: 0 for name in ordered}

    for source in ordered:
        for target in filtered_dag.adjacency.get(source, set()):
            if target not in parent_scope:
                continue
            if target in adjacency[source]:
                continue
            adjacency[source].add(target)
            indegree[target] += 1

    layers: List[List[str]] = []
    processed: Set[str] = set()
    ready = sorted(
        [name for name, degree in indegree.items() if degree == 0],
        key=lambda name: order_index[name],
    )

    while ready:
        layer = list(ready)
        layers.append(layer)
        next_ready: Set[str] = set()
        for source in layer:
            processed.add(source)
            for target in adjacency.get(source, set()):
                indegree[target] -= 1
                if indegree[target] == 0 and target not in processed:
                    next_ready.add(target)
        ready = sorted(next_ready, key=lambda name: order_index[name])

    if len(processed) < len(ordered):
        remaining = [name for name in ordered if name not in processed]
        logging.warning(
            "Parent requirement DAG contains a cycle or unresolved ordering for %d parents; "
            "falling back to a final sequential layer: %s",
            len(remaining),
            ", ".join(remaining[:20]),
        )
        layers.extend([[name] for name in remaining])

    return layers


def merge_plans(
    existing_plan: List[dict],
    incremental_plan: List[dict],
    affected_parents: set[str],
    decomposed_dag: RequirementDAG,
) -> List[dict]:
    kept = [
        task for task in existing_plan
        if get_task_parent_name(task.get("name", ""), decomposed_dag) not in affected_parents
    ]
    merged = kept + incremental_plan
    for i, task in enumerate(merged):
        task["order"] = i
    return merged


def collect_dependency_nodes(dag: RequirementDAG, node_name: str) -> List[str]:
    """Collect all prerequisite nodes for the given node name."""
    if not node_name or node_name not in dag.nodes:
        return []
    visited: set[str] = set()
    stack = list(dag.reverse_adjacency.get(node_name, set()))
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(dag.reverse_adjacency.get(current, set()))
    return sorted(visited)


def build_requirement_edges(dag: RequirementDAG) -> List[dict]:
    """Flatten DAG adjacency into a list of requirement edge dicts."""
    edges: List[dict] = []
    for source, targets in dag.adjacency.items():
        for target in targets:
            edges.append({"source": source, "target": target})
    return edges


def build_codegen_parent_dag(
    dag_source: str,
    requirement_dag: RequirementDAG,
    dependency_graph_payload: Any,
) -> RequirementDAG:
    """Build the DAG used for parent-level codegen ordering and dependency context."""
    source = str(dag_source or "requirement").strip().lower()
    if source == "none":
        adjacency: Dict[str, Set[str]] = {
            name: set() for name in requirement_dag.nodes
        }
        logging.info(
            "Using graph-free parent codegen ordering/context: nodes=%d edges=0",
            len(adjacency),
        )
        return RequirementDAG(dict(requirement_dag.nodes), adjacency)
    if source != "dependency":
        return requirement_dag

    raw_graph = (
        dependency_graph_payload.get("dependency_graph", {})
        if isinstance(dependency_graph_payload, dict)
        else {}
    )
    raw_edges = raw_graph.get("edges", []) if isinstance(raw_graph, dict) else []
    adjacency: Dict[str, Set[str]] = {
        name: set() for name in requirement_dag.nodes
    }
    for row in raw_edges if isinstance(raw_edges, list) else []:
        if not isinstance(row, dict):
            continue
        source_name = str(row.get("source") or "").strip()
        target_name = str(row.get("target") or "").strip()
        if source_name in adjacency and target_name in requirement_dag.nodes:
            adjacency[source_name].add(target_name)

    edge_count = sum(len(targets) for targets in adjacency.values())
    logging.info(
        "Using dependency graph for parent codegen DAG: nodes=%d edges=%d",
        len(adjacency),
        edge_count,
    )
    return RequirementDAG(dict(requirement_dag.nodes), adjacency)


def clone_requirement_dag(dag: RequirementDAG) -> RequirementDAG:
    """Clone a requirement DAG without introducing decomposition metadata."""
    return RequirementDAG(
        dict(dag.nodes),
        {
            name: set(targets)
            for name, targets in dag.adjacency.items()
        },
    )


def build_empty_actions_for_architectures(architectures: List[dict]) -> List[dict]:
    """Return an empty action list aligned to architecture parent order."""
    empty_actions: List[dict] = []
    for arch in architectures:
        parent = _parent_from_architecture_entry(arch)
        if not parent:
            continue
        empty_actions.append({"task": parent, "actions": []})
    return empty_actions


def build_codegen_parent_order(
    dag_source: str,
    ordered_parents: List[str],
    no_graph_seed: int,
) -> List[str]:
    """Build parent order for codegen, with seeded shuffle for graph-free mode."""
    ordered = [str(parent).strip() for parent in ordered_parents if str(parent).strip()]
    if str(dag_source or "").strip().lower() != "none":
        return ordered
    shuffled = list(ordered)
    random.Random(int(no_graph_seed)).shuffle(shuffled)
    return shuffled


def build_no_graph_plan(requirement_items: List[dict]) -> List[dict]:
    """Build a graph-free execution plan directly from requirement items."""
    plan: List[dict] = []
    for idx, item in enumerate(requirement_items):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        plan.append(
            {
                "name": name,
                "description": str(item.get("description") or "").strip(),
                "order": len(plan),
                "metadata": {},
            }
        )
    return plan


def build_no_graph_parent_groups(plan: List[dict]) -> List[Tuple[dict, List[dict]]]:
    """Build parent groups without DAG/decomposition, one parent per plan entry."""
    parent_groups: List[Tuple[dict, List[dict]]] = []
    for task in plan:
        if not isinstance(task, dict):
            continue
        parent_groups.append((task, []))
    return parent_groups


def apply_action_guided_structure_refinement(
    architectures: List[dict],
    component_merge_agent: Optional[Any],
    component_split_agent: Optional[Any],
) -> Tuple[List[dict], Dict[str, Any]]:
    """Apply action-guided split refinement without changing the earlier normalize stage."""
    refined_architectures: List[dict] = []
    split_groups_added = 0
    parent_reports: List[dict] = []
    total_arches = len(architectures)

    for refinement_idx, arch_entry in enumerate(architectures, start=1):
        if not isinstance(arch_entry, dict):
            continue
        parent_task = _parent_from_architecture_entry(arch_entry)
        architecture = arch_entry.get("architecture", {})
        if not parent_task or not isinstance(architecture, dict):
            refined_architectures.append(arch_entry)
            continue

        logging.info(
            "Refinement parent %d/%d: %s",
            refinement_idx,
            total_arches,
            parent_task,
        )

        current_architecture = architecture
        components = current_architecture.get("components", [])
        parent_report: Dict[str, Any] = {
            "parent_task": parent_task,
            "components_before": len(components) if isinstance(components, list) else 0,
        }

        should_merge = False
        if component_merge_agent is not None and isinstance(components, list):
            should_merge = any(
                isinstance(component, dict)
                and str(component.get("recommended_action") or "").strip().lower() == "merge"
                for component in components
            )

        if should_merge and component_merge_agent is not None:
            merged_architecture, merge_report = component_merge_agent.merge_architecture_components(
                parent_task,
                current_architecture,
            )
            current_architecture = merged_architecture
            if isinstance(merge_report, dict):
                parent_report["merge_report"] = merge_report

        if component_split_agent is not None:
            split_architecture, split_report = component_split_agent.split_architecture_components(
                parent_task,
                current_architecture,
            )
            current_architecture = split_architecture
            if isinstance(split_report, dict):
                parent_report["split_report"] = split_report
                split_groups_added += int(
                    split_report.get("stats", {}).get("split_group_count", 0) or 0
                )

        split_group_count = int(
            ((parent_report.get("split_report", {}) or {}).get("stats", {}) or {}).get("split_group_count", 0) or 0
        )
        if component_merge_agent is not None and split_group_count > 0:
            post_split_merged_architecture, post_split_merge_report = component_merge_agent.merge_architecture_components(
                parent_task,
                current_architecture,
                require_split_origin=True,
                include_rule_candidates=True,
            )
            current_architecture = post_split_merged_architecture
            if isinstance(post_split_merge_report, dict):
                parent_report["post_split_merge_report"] = post_split_merge_report

        arch_copy = dict(arch_entry)
        arch_copy["architecture"] = current_architecture
        refined_architectures.append(arch_copy)
        refined_components = current_architecture.get("components", [])
        parent_report["components_after"] = (
            len(refined_components) if isinstance(refined_components, list) else 0
        )
        parent_reports.append(parent_report)

    components_after = sum(
        len((arch.get("architecture", {}) or {}).get("components", []) or [])
        for arch in refined_architectures
        if isinstance(arch, dict)
    )
    report = {
        "stats": {
            "components_after": components_after,
            "merge_group_count": sum(
                int(
                    (merge_report.get("stats", {}) or {}).get("accepted_group_count", 0)
                    or len((merge_report.get("merge_groups", []) or []))
                )
                for report in parent_reports
                if isinstance(report, dict)
                for merge_report in [
                    (report.get("merge_report", {}) or {}),
                    (report.get("post_split_merge_report", {}) or {}),
                ]
                if isinstance(merge_report, dict) and merge_report
            ),
            "split_group_count": split_groups_added,
        },
        "parents": parent_reports,
    }
    return refined_architectures, report


def apply_pre_action_component_merge(
    architectures: List[dict],
    component_merge_agent: Optional[Any],
    *,
    enable_cross_requirement_merge: bool = False,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Run component merge before action selection for a refinement round."""
    if component_merge_agent is None:
        return architectures, {
            "stats": {
                "components_before": count_architecture_components(architectures),
                "components_after": count_architecture_components(architectures),
                "merge_group_count": 0,
            },
            "parents": [],
        }

    if enable_cross_requirement_merge:
        return apply_cross_requirement_component_merge(
            architectures,
            component_merge_agent,
        )

    merged_architectures: List[dict] = []
    parent_reports: List[dict] = []
    merge_groups_added = 0
    components_before = count_architecture_components(architectures)

    for arch_entry in architectures:
        if not isinstance(arch_entry, dict):
            continue
        parent_task = _parent_from_architecture_entry(arch_entry)
        architecture = arch_entry.get("architecture", {})
        if not parent_task or not isinstance(architecture, dict):
            merged_architectures.append(arch_entry)
            continue

        merged_architecture, merge_report = component_merge_agent.merge_architecture_components(
            parent_task,
            architecture,
        )
        arch_copy = dict(arch_entry)
        arch_copy["architecture"] = merged_architecture
        merged_architectures.append(arch_copy)
        if isinstance(merge_report, dict):
            parent_reports.append(merge_report)
            merge_groups_added += int(
                merge_report.get("stats", {}).get("accepted_group_count", 0)
                or len(merge_report.get("merge_groups", []) or [])
            )

    return merged_architectures, {
        "stats": {
            "components_before": components_before,
            "components_after": count_architecture_components(merged_architectures),
            "merge_group_count": merge_groups_added,
        },
        "parents": parent_reports,
    }


def choose_actions_for_architectures(
    architectures: List[dict],
    api_config: dict,
    output_dir: str,
    max_workers: int,
) -> List[dict]:
    """Run strategist action selection for architecture parents."""
    if not architectures:
        return []
    logging.info(
        "Choosing actions for %d architectures (parallel workers: %d)",
        len(architectures),
        max_workers,
    )
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        future_to_idx = {
            executor.submit(
                process_action_task,
                arch_info,
                api_config,
                output_dir,
            ): i
            for i, arch_info in enumerate(architectures)
        }
        actions: List[Any] = [None] * len(architectures)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                action_result = future.result()
                actions[idx] = action_result
                logging.info(
                    "  ✓ Completed actions for architecture %d/%d",
                    idx + 1,
                    len(architectures),
                )
            except Exception as exc:
                logging.error("Failed to choose actions for architecture %d: %s", idx, exc)
                raise
    return merge_actions_for_architectures(
        [],
        [entry for entry in actions if isinstance(entry, dict)],
        architectures,
    )


def _adaptive_metric_split_min_subrequirements(
    base_min_subrequirements: int,
    round_idx: int,
) -> int:
    """Tighten split triggering across rounds to reduce recursive over-splitting.

    Round 1 uses the configured base threshold, round 2 requires one more served
    subrequirement, and round 3+ requires two more, capped at 5.
    """
    base = max(1, int(base_min_subrequirements))
    round_number = max(1, int(round_idx))
    adaptive = base + min(round_number - 1, 2)
    return min(adaptive, 5)


def build_gap_add_stage_inputs(
    *,
    requirements_file: Path,
    req_path: Path | None,
    generated_files_path: Path,
    realization_report_path: Path,
) -> Dict[str, Any]:
    input_path = req_path if req_path and req_path.exists() else requirements_file
    input_text = input_path.read_text(encoding="utf-8") if input_path.exists() else ""
    requirements_payload = load_json_if_exists(requirements_file, False) or {}
    generated_entries = load_json_if_exists(generated_files_path, False) or []
    realization_report = load_json_if_exists(realization_report_path, False) or {}
    return {
        "input_text": input_text,
        "requirements_payload": requirements_payload if isinstance(requirements_payload, dict) else {},
        "generated_entries": generated_entries if isinstance(generated_entries, list) else [],
        "realization_report": realization_report if isinstance(realization_report, dict) else {},
    }


def run_gap_addition_stage(
    *,
    architectures: List[dict],
    args: argparse.Namespace,
    output_dir: Path,
    input_text: str,
    requirements_payload: Dict[str, Any],
    generated_entries: Any,
    realization_report: Dict[str, Any],
    component_merge_agent: Any,
    component_split_agent: Any,
    propose_gap_candidate_for_parent_fn=propose_gap_candidate_for_parent,
    judge_gap_candidate_fn=None,
    run_local_gap_cleanup_fn=run_local_gap_cleanup,
) -> Tuple[List[dict], Dict[str, Any]]:
    report_path = output_dir / "gap_addition_report.json"
    if not bool(getattr(args, "enable_gap_add_actions", False)):
        report = {"enabled": False, "accepted_count": 0, "parents": []}
        save_json(report, report_path)
        return architectures, report

    judge = judge_gap_candidate_fn
    if judge is None:
        judge = GapAdditionJudge(
            api_config=getattr(args, "api_config", {}) if hasattr(args, "api_config") else {},
            output_dir=str(output_dir),
        ).judge_candidate

    updated_architectures: List[dict] = []
    parent_reports: List[dict] = []
    accepted_count = 0
    for parent_entry in architectures:
        candidate = propose_gap_candidate_for_parent_fn(
            parent_entry=parent_entry,
            input_text=input_text,
            requirements_payload=requirements_payload,
            generated_entries=generated_entries if isinstance(generated_entries, list) else [],
            realization_report=realization_report if isinstance(realization_report, dict) else {},
            proposal_threshold=float(getattr(args, "gap_add_proposal_threshold", 0.55)),
            max_candidates_per_parent=1,
        )
        if candidate is None:
            updated_architectures.append(parent_entry)
            parent_reports.append(
                {
                    "parent_requirement": _parent_from_architecture_entry(parent_entry),
                    "accepted": False,
                    "reason": "no_candidate",
                }
            )
            continue

        decision = judge(
            candidate=candidate,
            parent_entry=parent_entry,
            generated_entries=generated_entries if isinstance(generated_entries, list) else [],
            realization_report=realization_report if isinstance(realization_report, dict) else {},
        )
        if decision.action == "add_requirement_and_component" and decision.final_confidence < float(
            getattr(args, "gap_add_requirement_threshold", 0.82)
        ):
            decision = GapAdditionDecision(
                **{**decision.__dict__, "decision": "reject", "action": "none", "reason": "below_threshold"}
            )
        elif decision.action == "add_component" and decision.final_confidence < float(
            getattr(args, "gap_add_component_threshold", 0.74)
        ):
            decision = GapAdditionDecision(
                **{**decision.__dict__, "decision": "reject", "action": "none", "reason": "below_threshold"}
            )

        if decision.action not in {"add_component", "add_requirement_and_component"}:
            updated_architectures.append(parent_entry)
            parent_reports.append(
                {
                    "parent_requirement": candidate.parent_requirement,
                    "accepted": False,
                    "candidate_type": candidate.candidate_type,
                    "reason": decision.reason,
                }
            )
            continue

        mutated = apply_gap_addition_decision(parent_entry=parent_entry, decision=decision)
        cleaned, cleanup_report = run_local_gap_cleanup_fn(
            parent_entry=mutated,
            component_merge_agent=component_merge_agent,
            component_split_agent=component_split_agent,
            augment_actions_with_component_metrics_fn=augment_actions_with_component_metrics,
            apply_action_guided_structure_refinement_fn=apply_action_guided_structure_refinement,
            split_cohesion_threshold=float(getattr(args, "component_metric_split_cohesion_threshold", 2.0 / 3.0)),
            split_min_subrequirements=max(4, int(getattr(args, "component_metric_split_min_subrequirements", 3))),
            merge_max_small_subrequirements=int(getattr(args, "component_metric_merge_max_small_subrequirements", 1)),
        )
        updated_architectures.append(cleaned)
        accepted_count += 1
        parent_reports.append(
            {
                "parent_requirement": candidate.parent_requirement,
                "accepted": True,
                "decision": decision.action,
                "final_confidence": decision.final_confidence,
                "cleanup_report": cleanup_report,
            }
        )

    report = {"enabled": True, "accepted_count": accepted_count, "parents": parent_reports}
    save_json(report, report_path)
    return updated_architectures, report


def run_action_feedback_rounds(
    *,
    architectures: List[dict],
    initial_actions: List[dict],
    component_merge_agent: Optional[Any],
    component_split_agent: Optional[Any],
    rounds: int,
    api_config: dict,
    output_dir: Path,
    max_workers: int,
    stop_on_stable: bool = False,
    save_stops_component: bool = False,
    enable_cross_requirement_merge: bool = False,
    save_round_artifacts: bool = True,
    action_selector: Optional[Callable[[List[dict], int], List[dict]]] = None,
    existing_generated_entries: Any = None,
    tdd_revise_failure_threshold: int = 3,
    decomposed_dag: RequirementDAG | Dict[str, Any] | None = None,
    enable_metric_actions: bool = False,
    metric_split_cohesion_threshold: float = 2.0 / 3.0,
    metric_split_min_subrequirements: int = 3,
    metric_merge_max_small_subrequirements: int = 1,
    metric_merge_judge: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    """Run metrics-driven structural feedback before module planning."""

    def _log_round_proposed_actions(round_number: int, actions_payload: List[dict]) -> None:
        for entry in actions_payload if isinstance(actions_payload, list) else []:
            if not isinstance(entry, dict):
                continue
            parent = _parent_from_action_entry(entry)
            rows = entry.get("actions", [])
            if not parent or not isinstance(rows, list) or not rows:
                continue
            logging.info("Round %d proposed actions for parent '%s':", round_number, parent)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                component = str(row.get("component") or "").strip()
                action = str(row.get("action") or "").strip()
                origin = str(row.get("action_origin") or "").strip()
                rationale = str(row.get("rationale") or "").strip()
                suffix = f" [{origin}]" if origin else ""
                logging.info("  %s -> %s%s", component, action, suffix)
                if rationale:
                    logging.info("    rationale: %s", rationale)

    def _log_round_accepted_actions(round_number: int, refinement_report: Dict[str, Any]) -> None:
        for parent_report in refinement_report.get("parents", []) if isinstance(refinement_report, dict) else []:
            if not isinstance(parent_report, dict):
                continue
            parent = str(parent_report.get("parent_task") or "").strip()
            merge_report = parent_report.get("merge_report") or {}
            split_report = parent_report.get("split_report") or {}
            merge_groups = merge_report.get("merge_groups", []) if isinstance(merge_report, dict) else []
            split_groups = split_report.get("split_groups", []) if isinstance(split_report, dict) else []

            for group in merge_groups if isinstance(merge_groups, list) else []:
                merged_name = str(group.get("merged_name") or "").strip()
                sources = [
                    str(item.get("source_name") or "").strip()
                    for item in group.get("sources", [])
                    if isinstance(item, dict)
                ]
                logging.info(
                    "Round %d accepted merge for parent '%s': %s -> %s",
                    round_number,
                    parent,
                    sources,
                    merged_name,
                )

            for group in split_groups if isinstance(split_groups, list) else []:
                component_name = str(group.get("component_name") or "").strip()
                split_into = [str(item).strip() for item in group.get("split_into", []) if str(item).strip()]
                confidence = group.get("confidence")
                logging.info(
                    "Round %d accepted split for parent '%s': %s -> %s (confidence=%s)",
                    round_number,
                    parent,
                    component_name,
                    split_into,
                    confidence,
                )
                reason = str(group.get("reason") or "").strip()
                if reason:
                    logging.info("  reason: %s", reason)

    total_rounds = max(1, int(rounds))
    current_architectures = list(architectures)
    last_actions: List[dict] = []
    round_reports: List[dict] = []
    total_merge_groups = 0
    total_split_groups = 0
    total_action_counts: Dict[str, int] = {}
    stopped_components: Dict[str, Set[str]] = {}
    revise_fallback_report: Dict[str, Any] | None = None

    for round_idx in range(1, total_rounds + 1):
        round_components_before = count_architecture_components(current_architectures)
        if save_round_artifacts:
            save_json(
                current_architectures,
                output_dir / f"architectures_round_{round_idx}_before_pre_action_merge.json",
            )
        current_architectures, pre_action_merge_report = apply_pre_action_component_merge(
            current_architectures,
            component_merge_agent,
            enable_cross_requirement_merge=enable_cross_requirement_merge,
        )
        pre_action_merge_groups = int(
            pre_action_merge_report.get("stats", {}).get("merge_group_count", 0) or 0
        )
        components_after_merge = count_architecture_components(current_architectures)
        if save_round_artifacts:
            save_json(
                current_architectures,
                output_dir / f"architectures_round_{round_idx}_after_pre_action_merge.json",
            )
            save_json(
                pre_action_merge_report,
                output_dir / f"component_merge_report_round_{round_idx}_pre_action.json",
            )

        selection_architectures = current_architectures
        if save_stops_component and stopped_components:
            selection_architectures = filter_architectures_for_unstopped_components(
                current_architectures,
                stopped_components,
            )

        if round_idx == 1:
            round_actions = merge_actions_for_architectures(
                [],
                initial_actions,
                current_architectures,
            )
        elif action_selector is not None:
            round_actions = action_selector(selection_architectures, round_idx)
        elif enable_metric_actions:
            effective_split_min_subrequirements = _adaptive_metric_split_min_subrequirements(
                int(metric_split_min_subrequirements),
                round_idx,
            )
            round_actions = _build_default_empty_actions(current_architectures)
            round_actions, _ = augment_actions_with_component_metrics(
                architectures=current_architectures,
                actions=round_actions,
                decomposed_dag=decomposed_dag,
                split_cohesion_threshold=float(metric_split_cohesion_threshold),
                split_min_subrequirements=effective_split_min_subrequirements,
                merge_judge=metric_merge_judge,
                merge_max_small_subrequirements=int(metric_merge_max_small_subrequirements),
            )
        else:
            round_actions = _build_default_empty_actions(current_architectures)
        if save_stops_component:
            round_actions = add_saved_component_actions(
                round_actions,
                current_architectures,
                stopped_components,
            )

        round_actions = merge_actions_for_architectures(
            [],
            round_actions,
            current_architectures,
        )
        _log_round_proposed_actions(round_idx, round_actions)
        last_actions = round_actions
        action_counts = count_action_types(round_actions)
        if save_stops_component:
            for parent, components in collect_saved_component_actions(round_actions).items():
                stopped_components.setdefault(parent, set()).update(components)
        for action, count in action_counts.items():
            total_action_counts[action] = total_action_counts.get(action, 0) + count
        if save_round_artifacts:
            save_json(round_actions, output_dir / f"actions_round_{round_idx}.json")

        hinted_architectures = apply_action_hints_to_architectures(
            current_architectures,
            round_actions,
        )
        components_before = count_architecture_components(hinted_architectures)
        if component_split_agent is not None:
            refined_architectures, refinement_report = apply_action_guided_structure_refinement(
                architectures=hinted_architectures,
                component_merge_agent=component_merge_agent,
                component_split_agent=component_split_agent,
            )
        else:
            refined_architectures = hinted_architectures
            refinement_report = {
                "stats": {
                    "components_after": components_before,
                    "merge_group_count": 0,
                    "split_group_count": 0,
                }
            }
        if save_round_artifacts:
            save_json(
                refinement_report,
                output_dir / f"component_refinement_report_round_{round_idx}.json",
            )
        _log_round_accepted_actions(round_idx, refinement_report)
        components_after = count_architecture_components(refined_architectures)
        merge_groups = int(refinement_report.get("stats", {}).get("merge_group_count", 0) or 0)
        split_groups = int(refinement_report.get("stats", {}).get("split_group_count", 0) or 0)
        merge_groups += pre_action_merge_groups
        total_merge_groups += merge_groups
        total_split_groups += split_groups
        round_report = {
            "round": round_idx,
            "stats": {
                "components_before": round_components_before,
                "components_after_merge": components_after_merge,
                "components_before_split": components_before,
                "components_after": components_after,
                "merge_group_count": merge_groups,
                "split_group_count": split_groups,
                "action_counts": action_counts,
                "stopped_component_count": count_stopped_components(stopped_components),
            },
            "pre_action_merge_report": pre_action_merge_report,
            "refinement_report": refinement_report,
        }
        round_reports.append(round_report)

        logging.info(
            "Action feedback round %d/%d completed: components_before=%d components_after=%d merge_groups=%d split_groups=%d actions=%s",
            round_idx,
            total_rounds,
            components_before,
            components_after,
            merge_groups,
            split_groups,
            action_counts,
        )
        current_architectures = refined_architectures
        active_components_after = count_unstopped_architecture_components(
            current_architectures,
            stopped_components,
        )
        round_report["stats"]["active_component_count"] = active_components_after
        if save_round_artifacts:
            save_json(round_report, output_dir / f"action_refinement_round_{round_idx}.json")
        if save_stops_component and active_components_after == 0:
            logging.info(
                "Stopping action feedback early at round %d because all components were saved",
                round_idx,
            )
            break
        if stop_on_stable and merge_groups == 0 and split_groups == 0:
            logging.info(
                "Structural convergence reached at round %d; running revise-only fallback",
                round_idx,
            )
            strategist_revise_actions: List[dict] = []
            if action_selector is None:
                strategist_actions = choose_actions_for_architectures(
                    selection_architectures,
                    api_config,
                    str(output_dir),
                    max_workers,
                )
                strategist_revise_actions = _filter_revise_only_actions(strategist_actions)
            tdd_revise_report = build_tdd_revise_action_report(
                existing_generated_entries,
                failure_threshold=tdd_revise_failure_threshold,
            )
            tdd_revise_actions = tdd_revise_report.get("actions", [])
            combined_revise_actions = _combine_actions_for_architectures(
                current_architectures,
                strategist_revise_actions,
                tdd_revise_actions,
            )
            combined_revise_actions = [
                entry
                for entry in combined_revise_actions
                if isinstance(entry, dict) and isinstance(entry.get("actions"), list) and entry.get("actions")
            ]
            last_actions = combined_revise_actions
            revise_fallback_report = {
                "trigger_round": round_idx,
                "strategist_revise_count": sum(
                    len(entry.get("actions", []))
                    for entry in strategist_revise_actions
                    if isinstance(entry, dict)
                ),
                "tdd_revise_count": sum(
                    len(entry.get("actions", []))
                    for entry in tdd_revise_actions
                    if isinstance(entry, dict)
                ),
                "combined_revise_count": sum(
                    len(entry.get("actions", []))
                    for entry in combined_revise_actions
                    if isinstance(entry, dict)
                ),
                "strategist_revises": strategist_revise_actions,
                "tdd_revise_report": tdd_revise_report,
                "combined_revises": combined_revise_actions,
            }
            if save_round_artifacts:
                save_json(revise_fallback_report, output_dir / "revise_fallback_report.json")
            logging.info(
                "Revise-only fallback retained %d strategist revise actions",
                int(revise_fallback_report["strategist_revise_count"]),
            )
            logging.info(
                "TDD revise trigger contributed %d actions",
                int(revise_fallback_report["tdd_revise_count"]),
            )
            logging.info(
                "Combined revise action count after convergence: %d",
                int(revise_fallback_report["combined_revise_count"]),
            )
            logging.info(
                "Stopping action feedback early at round %d because no structural changes were applied",
                round_idx,
            )
            break

    summary = {
        "stats": {
            "rounds_requested": total_rounds,
            "rounds_completed": len(round_reports),
            "components_before": count_architecture_components(architectures),
            "components_after": count_architecture_components(current_architectures),
            "merge_group_count": total_merge_groups,
            "split_group_count": total_split_groups,
            "action_counts": total_action_counts,
            "stopped_component_count": count_stopped_components(stopped_components),
            "stopped_components": {
                parent: sorted(components)
                for parent, components in sorted(stopped_components.items())
            },
        },
        "rounds": round_reports,
    }
    if revise_fallback_report is not None:
        summary["revise_fallback"] = revise_fallback_report
    return current_architectures, last_actions, summary


def build_components_from_memory(memory_agent: Any) -> List[dict]:
    """Create component list for dependency inference from MemoryAgent snapshot."""
    snapshot = getattr(memory_agent, "snapshot", None)
    if snapshot is None:
        return []
    components: List[dict] = []
    for component_key, impl in snapshot.implemented_components.items():
        components.append(
            {
                "id": component_key,
                "name": impl.component_name,
                "requirement_node": impl.requirement_node,
                "parent_requirement": impl.metadata.get("parent_requirement", impl.requirement_node),
                "file_path": impl.file_path,
                "responsibilities": list(impl.metadata.get("responsibilities", []) or []),
                "serves_subrequirements": list(impl.metadata.get("serves_subrequirements", []) or []),
                "exports": impl.exports,
                "dependencies": impl.dependencies,
                "class_names": impl.class_names,
                "function_signatures": impl.function_signatures,
                "status": getattr(impl, "status", ""),
                "metadata": impl.metadata,
            }
        )
    return components


def register_architecture_components(
    memory_agent: Any,
    architectures: List[dict],
) -> None:
    """Register planned components from architectures into memory."""
    for arch_result in architectures:
        parent_task = arch_result.get("parent_task") or arch_result.get("task")
        if not parent_task:
            parent_task = (
                arch_result.get("architecture", {})
                .get("requirement", {})
                .get("name", "Unknown")
            )
        components = arch_result.get("architecture", {}).get("components", [])
        for comp in components:
            comp_name = comp.get("name", "UnknownComponent")
            serves_subreqs = comp.get("serves_subrequirements", [])
            memory_agent.register_component_implementation(
                component_name=comp_name,
                requirement_node=parent_task,
                file_path=f"planned/{comp_name}.py",
                class_names=[comp_name],
                function_signatures=[],
                dependencies=[],
                exports=comp.get("responsibilities", []),
                status="planned",
                responsibilities=comp.get("responsibilities", []),
                serves_subrequirements=serves_subreqs,
                parent_requirement=parent_task,
            )


def generate_dependency_graph_artifacts(
    api_config: dict,
    output_dir: Path,
    memory_agent: Any,
    dag: RequirementDAG,
) -> dict:
    """Generate dependency graph artifacts and persist to disk with debug logging."""
    components = build_components_from_memory(memory_agent)
    requirement_edges = build_requirement_edges(dag)
    logging.info(
        "Dependency graph generation starting: components=%d requirement_edges=%d",
        len(components),
        len(requirement_edges),
    )
    logging.debug(
        "Dependency graph input summary: components_preview=%s requirement_edges_preview=%s",
        components[:5],
        requirement_edges[:10],
    )
    dependency_agent = DependencyGraphAgent(api_config=api_config, output_dir=str(output_dir))
    logging.info("Dependency graph LLM inference starting")
    dependency_graph = dependency_agent.build_requirement_dependency_edges(
        components,
        constraints={
            "must_use_only_component_ids": True,
            "disallow_self_dependency": True,
            "allow_same_requirement_edges": False,
        },
        requirement_edges=requirement_edges,
    )
    dep_edges = dependency_graph.get("edges", []) if isinstance(dependency_graph, dict) else []
    comp_edges = dependency_graph.get("component_edges", []) if isinstance(dependency_graph, dict) else []
    unresolved = dependency_graph.get("unresolved", []) if isinstance(dependency_graph, dict) else []
    logging.info(
        "Dependency graph generation completed: requirement_edges=%d component_edges=%d unresolved=%d",
        len(dep_edges) if isinstance(dep_edges, list) else 0,
        len(comp_edges) if isinstance(comp_edges, list) else 0,
        len(unresolved) if isinstance(unresolved, list) else 0,
    )
    logging.debug("Dependency graph result summary: edges_preview=%s unresolved_preview=%s", dep_edges[:10] if isinstance(dep_edges, list) else [], unresolved[:10] if isinstance(unresolved, list) else [])
    dependency_graph_payload = {
        "components": components,
        "requirement_edges": requirement_edges,
        "dependency_graph": dependency_graph,
    }
    dependency_graph_path = output_dir / "dependency_graph.json"
    save_json(dependency_graph_payload, dependency_graph_path)
    logging.info("Dependency graph persisted to %s", dependency_graph_path)
    return dependency_graph_payload


def process_parent_architecture_task(
    parent_info: Tuple[int, dict, List[dict]], 
    api_config: dict, 
    output_dir: str, 
    memory_agent: Any,
    existing_modules: List[dict],
    dag_summary: dict
) -> Tuple[int, dict]:
    """Process architecture generation for a parent node and all its sub-nodes."""
    i, parent_task, sub_tasks = parent_info
    from agents import ArchitectAgent
    architect = ArchitectAgent(api_config=api_config, output_dir=output_dir)
    
    parent_name = parent_task.get('name', 'Unknown')
    logging.info(f"Generating unified architecture for parent {i+1}: {parent_name} with {len(sub_tasks)} sub-tasks")
    
    # Get memory context including all previously implemented components
    implemented_components_desc = memory_agent.format_implementations_for_prompt()
    
    # Combine environment and implementation context
    full_context = f"{implemented_components_desc}"

    logging.debug(implemented_components_desc)
    
    architecture = architect.generate_parent_architecture(
        parent_requirement=parent_task,
        sub_requirements=sub_tasks,
        environment_feedback=full_context,
        existing_modules=existing_modules,
        dag_summary=dag_summary
    )
    
    return i, {
        "parent_task": parent_task.get("name", f"parent_{i}"),
        "parent_node": parent_task,
        "sub_tasks": [sub.get("name") for sub in sub_tasks],
        "architecture": architecture
    }

def process_architecture_task(task_info: Tuple[int, dict], api_config: dict, output_dir: str, 
                              memory_desc: str, dag_summary: dict) -> Tuple[int, dict]:
    """Process a single architecture generation task (backward compatibility)."""
    i, task = task_info
    from agents import ArchitectAgent
    architect = ArchitectAgent(api_config=api_config, output_dir=output_dir)
    logging.info(f"Generating architecture for task {i+1}: {task.get('name', 'Unknown')}")
    architecture = architect.generate_architecture(task, memory_desc, dag_summary)
    return i, {
        "parent_task": task.get("name", f"task_{i}"),
        "parent_node": task,
        "task": task.get("name", f"task_{i}"),
        "sub_tasks": [],
        "architecture": architecture
    }


def process_action_task(arch_info: dict, api_config: dict, output_dir: str) -> dict:
    """Process a single action selection task."""
    from agents import StrategistAgent
    strategist = StrategistAgent(api_config=api_config, output_dir=output_dir)
    task_name = arch_info.get("parent_task") or arch_info.get("task")
    logging.info(f"Choosing actions for task: {task_name or 'Unknown'}")
    actions = strategist.choose_actions(arch_info["architecture"])
    return {
        "task": task_name,
        "actions": actions
    }


def process_code_generation_task(
    arch_info: Tuple[int, dict],
    api_config: dict,
    output_dir: str,
    code_output_dir: str,
    memory_agent: Any,
    dependency_dag: RequirementDAG,
    layout_policy: Dict[str, Any],
    module_registry: Dict[str, str] | None = None,
    component_plan_index: Dict[str, Dict[str, Any]] | None = None,
    existing_generated_index: Dict[str, Dict[str, Any]] | None = None,
    source_context_parents: List[str] | None = None,
    implemented_context_override: Optional[str] = None,
    register_in_memory: bool = True,
    postcheck_max_workers: int = 1,
    rerun_retained_tdd_failures: bool = False,
) -> Tuple[int, List[dict]]:
    """Process code generation for a parent-level architecture."""
    from agents import CodeGeneratorAgent

    parent_started_at = time.perf_counter()
    i, arch_result = arch_info
    code_generator = CodeGeneratorAgent(api_config=api_config, output_dir=output_dir)
    
    parent_task = arch_result.get("parent_task", "Unknown")
    sub_tasks = arch_result.get("sub_tasks", [])
    architecture = arch_result.get("architecture", {})
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Generating code for parent {i+1}: {parent_task}")
    logging.info(f"  Covers {len(sub_tasks)} sub-tasks")
    logging.info(f"  Components to generate: {len(architecture.get('components', []))}")
    logging.info(f"{'='*60}\n")
    
    dependency_nodes = collect_dependency_nodes(dependency_dag, parent_task)
    mapped_sources = [name for name in (source_context_parents or []) if name]
    context_nodes = list(dict.fromkeys(dependency_nodes + mapped_sources))
    logging.debug("Codegen dependency nodes for %s: %s", parent_task, context_nodes)
    if implemented_context_override is not None:
        implemented_context = implemented_context_override
    elif memory_agent is not None:
        implemented_context = memory_agent.format_implementations_for_prompt(
            filter_nodes=context_nodes,
            status_filter="implemented"
        )
    else:
        implemented_context = ""
    logging.debug(implemented_context)

        
    # Create a unified task description for code generation
    unified_task = {
        "name": parent_task,
        "description": architecture.get("requirement", {}).get("description", ""),
        "sub_requirements": sub_tasks,
        "parent_node": arch_result.get("parent_node"),
        "parent_prev_node": arch_result.get("parent_prev_node"),
    }

    logging.debug(unified_task)
    
    components = architecture.get("components", [])
    if not isinstance(components, list):
        components = []
    components = components[: architecture.get("component_count", 999)]

    planned_paths = _build_component_file_plan(architecture, unified_task, layout_policy)
    logging.info(
        "Layout schema plan for '%s': root=%s, components=%d",
        parent_task,
        layout_policy.get("layout_root"),
        len(planned_paths),
    )
    if planned_paths:
        planned_lines = [
            f"- {name}: {path}"
            for name, path in sorted(planned_paths.items())
        ]
        implemented_context = (
            f"{implemented_context}\n\n"
            "=== FILE PLAN SCHEMA (must follow exactly) ===\n"
            f"Top-level whitelist: {layout_policy.get('top_whitelist', [])}\n"
            "Component -> file path:\n"
            + "\n".join(planned_lines)
            + "\n\nImport policy:\n"
            f"1) Prefer imports under '{layout_policy.get('layout_root')}'.\n"
            "2) Reuse existing modules from MODULE REGISTRY.\n"
            "3) Avoid introducing new top-level package roots."
        )

    local_module_registry = dict(module_registry or {})
    if local_module_registry:
        module_hint_lines = [
            f"- {name}: {module}"
            for name, module in sorted(local_module_registry.items())[:200]
            if name and module
        ]
        if module_hint_lines:
            implemented_context = (
                f"{implemented_context}\n\n"
                "=== MODULE REGISTRY (import from these modules first, avoid creating new top-level packages) ===\n"
                + "\n".join(module_hint_lines)
            )

    created_files = []
    pending_postchecks: List[Dict[str, Any]] = []
    component_plan_index = component_plan_index or {}
    existing_generated_index = existing_generated_index or {}
    for component in components:
        component_started_at = time.perf_counter()
        component_name = str(component.get("name", "")).strip()
        plan_key = f"{parent_task}::{component_name}"
        planned_entry = component_plan_index.get(plan_key, {})
        planned_rel_path = str(planned_entry.get("planned_file_path", "")).strip() or planned_paths.get(component_name)
        if layout_policy.get("enabled"):
            planned_rel_path = _normalize_layout_file_path(
                planned_rel_path or "",
                layout_policy,
                fallback_rel_path=f"{layout_policy.get('layout_root')}/generated/{_to_snake_case(component_name)}.py",
            )

        planned_export_symbols = planned_entry.get("export_symbols", []) if isinstance(planned_entry, dict) else []
        if not isinstance(planned_export_symbols, list) or not planned_export_symbols:
            planned_export_symbols = _derive_component_export_symbols(
                component_name=component_name,
                responsibilities=component.get("responsibilities", []) if isinstance(component, dict) else [],
                planned_file_path=str(planned_rel_path or ""),
            )
        previous_attempt_feedback = ""
        previous_entry = existing_generated_index.get(plan_key, {})
        if isinstance(previous_entry, dict) and not _generated_entry_has_usable_code(
            previous_entry,
            rerun_retained_tdd_failures=bool(rerun_retained_tdd_failures),
        ):
            previous_attempt_feedback = str(
                _evaluate_generated_entry_status(previous_entry).get("compressed_feedback") or ""
            ).strip()
            if previous_attempt_feedback:
                logging.info(
                    "  Retrying %s with compressed previous-attempt feedback",
                    component_name,
                )

        code_result = code_generator.generate_code(
            component,
            unified_task,
            architecture,
            language="python",
            implemented_components_context=implemented_context,
            planned_file_path=planned_rel_path,
            previous_attempt_feedback=previous_attempt_feedback,
        )
        code_result, layout_meta = _enforce_layout_with_oov_retry(
            code_generator=code_generator,
            code_result=code_result,
            component=component if isinstance(component, dict) else {"name": component_name},
            unified_task=unified_task,
            architecture=architecture,
            implemented_context=implemented_context,
            layout_policy=layout_policy,
            planned_rel_path=str(planned_rel_path or ""),
        )

        files = code_generator.save_generated_code(code_result, code_output_dir)
        comp_name = code_result["component_name"]
        serves_subreqs = code_result.get("serves_subrequirements", sub_tasks)
        init_files: List[str] = []
        code_file_path = files.get("code")
        if code_file_path:
            init_files = _ensure_package_inits(
                Path(code_file_path),
                Path(code_output_dir),
                    str(layout_policy.get("layout_root") or ""),
                )
            module_path = _module_from_relative_py_path(code_result.get("file_path", ""))
            if module_path:
                local_module_registry[comp_name] = module_path

        allowed_write_rel_paths: Set[str] = set()
        for key in ("code", "test"):
            file_path = files.get(key)
            if not file_path:
                continue
            try:
                rel_file = str(Path(file_path).resolve().relative_to(Path(code_output_dir).resolve())).replace("\\", "/")
            except Exception:
                continue
            if rel_file:
                allowed_write_rel_paths.add(rel_file)
        for init_file in init_files:
            try:
                rel_init = str(Path(init_file).resolve().relative_to(Path(code_output_dir).resolve())).replace("\\", "/")
            except Exception:
                continue
            if rel_init:
                allowed_write_rel_paths.add(rel_init)

        if not files or not code_result.get("save_succeeded", bool(files)):
            created_files.append({
                "component": comp_name,
                "parent_task": parent_task,
                "sub_tasks": serves_subreqs,
                "component_responsibilities": component.get("responsibilities", []) if isinstance(component, dict) else [],
                "component_export_symbols": planned_export_symbols,
                "files": files,
                "planned_file_path": planned_rel_path,
                "module_path": local_module_registry.get(comp_name, ""),
                "init_files": init_files,
                "layout_enforcement": layout_meta,
                "import_postcheck": {
                    "passed": False,
                    "skipped": True,
                    "reason": "save_failed",
                    "error": str(code_result.get("save_error") or "generated files were not saved"),
                },
                "syntax_postcheck": code_result.get("syntax_postcheck", {}),
                "compile_postcheck": code_result.get("compile_postcheck", {}),
                "generation_status": str(code_result.get("generation_status") or "save_failed"),
                "tdd_passed": code_result.get("tdd_passed"),
                "tdd_final_pytest_rc": code_result.get("tdd_final_pytest_rc"),
                "save_succeeded": False,
                "save_error": str(code_result.get("save_error") or "generated files were not saved"),
            })
            logging.error(
                "Component '%s' marked save_failed; skipping component postcheck because generated files were not saved: %s",
                comp_name,
                str(code_result.get("save_error") or "generated files were not saved"),
            )
            continue

        pending_postchecks.append({
            "component_name": comp_name,
            "code_result": code_result,
            "files": files,
            "serves_subreqs": serves_subreqs,
            "init_files": init_files,
            "planned_export_symbols": planned_export_symbols,
            "planned_file_path": planned_rel_path,
            "layout_meta": layout_meta,
            "component_responsibilities": component.get("responsibilities", []) if isinstance(component, dict) else [],
            "module_path": local_module_registry.get(comp_name, ""),
            "allowed_write_rel_paths": allowed_write_rel_paths,
            "started_at": component_started_at,
        })

    def _run_component_postcheck_task(task: Dict[str, Any]) -> Dict[str, Any]:
        import_postcheck = code_generator.postcheck_saved_component(
            code_result=task["code_result"],
            repo_root=code_output_dir,
            created_files=task["files"],
            implemented_components_context=implemented_context,
            allowed_write_rel_paths=set(task.get("allowed_write_rel_paths", set())),
        )
        result_entry = {
            "component": str(task.get("component_name", "")).strip(),
            "parent_task": parent_task,
            "sub_tasks": task["serves_subreqs"],
            "component_responsibilities": task["component_responsibilities"],
            "component_export_symbols": task["planned_export_symbols"],
            "files": task["files"],
            "planned_file_path": task["planned_file_path"],
            "module_path": task["module_path"],
            "init_files": task["init_files"],
            "layout_enforcement": task["layout_meta"],
            "import_postcheck": import_postcheck,
            "syntax_postcheck": task["code_result"].get("syntax_postcheck", {}),
            "compile_postcheck": task["code_result"].get("compile_postcheck", {}),
            "generation_status": str(task["code_result"].get("generation_status") or ""),
            "tdd_passed": task["code_result"].get("tdd_passed"),
            "tdd_final_pytest_rc": task["code_result"].get("tdd_final_pytest_rc"),
        }
        return {"task": task, "entry": result_entry}

    def _select_conflict_free_batch(
        tasks: List[Dict[str, Any]],
        max_batch_workers: int,
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        used_paths: Set[str] = set()
        for task in tasks:
            write_set = set(task.get("allowed_write_rel_paths", set()))
            if used_paths.intersection(write_set):
                continue
            selected.append(task)
            used_paths.update(write_set)
            if len(selected) >= max_batch_workers:
                break
        return selected

    logging.info(
        "Component postcheck task pool starting for '%s': tasks=%d max_workers=%d",
        parent_task,
        len(pending_postchecks),
        max(1, int(postcheck_max_workers)),
    )
    remaining_postchecks = list(pending_postchecks)
    postcheck_batch_idx = 0
    while remaining_postchecks:
        postcheck_batch_idx += 1
        batch = _select_conflict_free_batch(
            remaining_postchecks,
            max_batch_workers=max(1, int(postcheck_max_workers)),
        )
        if not batch:
            batch = [remaining_postchecks[0]]
        batch_keys = {id(task) for task in batch}
        remaining_postchecks = [task for task in remaining_postchecks if id(task) not in batch_keys]
        logging.info(
            "Component postcheck batch %d for '%s': %d task(s) with disjoint write sets: %s",
            postcheck_batch_idx,
            parent_task,
            len(batch),
            ", ".join(str(task.get("component_name", "")) for task in batch),
        )
        if len(batch) == 1:
            batch_results = [_run_component_postcheck_task(batch[0])]
        else:
            batch_results = []
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                future_to_task = {
                    executor.submit(_run_component_postcheck_task, task): task
                    for task in batch
                }
                for future in as_completed(future_to_task):
                    batch_results.append(future.result())

        for result in batch_results:
            task = result["task"]
            entry = result["entry"]
            created_files.append(entry)
            comp_name = str(task.get("component_name", ""))
            files = task.get("files", {})
            serves_subreqs = task.get("serves_subreqs", [])
            init_files = task.get("init_files", [])
            code_result = task.get("code_result", {})

            if register_in_memory and memory_agent is not None:
                try:
                    metadata = code_generator.extract_component_metadata(
                        code_result,
                        requirement_node=parent_task
                    )
                    memory_agent.register_component_implementation(**metadata)
                    logging.info(
                        "  ✓ %s -> %s (serves %d sub-reqs, module=%s, init_created=%d, elapsed=%.2fs)",
                        comp_name,
                        files.get("code", "N/A"),
                        len(serves_subreqs),
                        task.get("module_path", ""),
                        len(init_files),
                        time.perf_counter() - float(task.get("started_at") or parent_started_at),
                    )
                except Exception as e:
                    logging.warning(f"  ✗ Failed to register {comp_name}: {e}")
            else:
                logging.info(
                    "  ✓ %s -> %s (serves %d sub-reqs, module=%s, init_created=%d, elapsed=%.2fs)",
                    comp_name,
                    files.get("code", "N/A"),
                    len(serves_subreqs),
                    task.get("module_path", ""),
                    len(init_files),
                    time.perf_counter() - float(task.get("started_at") or parent_started_at),
                )

    package_modules: List[str] = []
    for entry in created_files:
        if not isinstance(entry, dict):
            continue
        module_name = str(entry.get("module_path") or "").strip()
        if module_name:
            parts = module_name.split(".")
            for idx in range(1, len(parts)):
                package_name = ".".join(parts[:idx])
                if package_name and package_name not in package_modules:
                    package_modules.append(package_name)
        for init_file in entry.get("init_files", []) if isinstance(entry.get("init_files", []), list) else []:
            try:
                rel_init = str(Path(init_file).resolve().relative_to(Path(code_output_dir).resolve())).replace("\\", "/")
            except Exception:
                continue
            package_name = _module_from_relative_py_path(rel_init)
            if package_name and package_name not in package_modules:
                package_modules.append(package_name)

    if package_modules:
        package_postcheck = code_generator.postcheck_package_modules(
            package_modules=package_modules,
            repo_root=code_output_dir,
            implemented_components_context=implemented_context,
        )
        logging.info(
            "Parent/package import postcheck passed for '%s': modules=%d",
            parent_task,
            len(package_postcheck.get("modules", [])) if isinstance(package_postcheck, dict) else 0,
        )

    top_level_package_modules = _discover_top_level_package_modules(Path(code_output_dir))
    if top_level_package_modules:
        repo_package_postcheck = code_generator.postcheck_package_modules(
            package_modules=top_level_package_modules,
            repo_root=code_output_dir,
            implemented_components_context=implemented_context,
            max_fix_attempts=int(code_generator.package_postcheck_max_fix_attempts),
        )
        logging.info(
            "Repo-level top package import postcheck completed for '%s': modules=%d passed=%s",
            parent_task,
            len(repo_package_postcheck.get("modules", [])) if isinstance(repo_package_postcheck, dict) else 0,
            bool(repo_package_postcheck.get("passed")) if isinstance(repo_package_postcheck, dict) else False,
        )

    logging.info(
        "Completed code generation for '%s': %d components in %.2fs\n",
        parent_task,
        len(created_files),
        time.perf_counter() - parent_started_at,
    )
    return i, created_files

def parse_log_level(log_level: str) -> int:
    """Parse log level from string."""
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    try:
        return levels.get(log_level.lower(), logging.INFO)
    except Exception:
        return logging.INFO
    

def build_single_run_command(
    base_args: argparse.Namespace,
    requirements_file: Path,
    evolve_requirements_file: Path | None,
    force_regenerate: bool,
) -> List[str]:
    """Build a command line for a single pipeline run."""
    script_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script_path),
        "--requirements-file", str(requirements_file),
        "--repo", str(base_args.repo),
        "--workspace", str(Path(base_args.workspace).resolve()),
        "--base-url", str(base_args.base_url),
        "--api-key", str(base_args.api_key),
        "--model", str(base_args.model),
        "--max-workers", str(base_args.max_workers),
        "--log-level", str(base_args.log_level),
        "--parent-codegen-dag-source", str(getattr(base_args, "parent_codegen_dag_source", "dependency")),
    ]
    if base_args.req_path:
        cmd.extend(["--req-path", str(Path(base_args.req_path).resolve())])
    if base_args.output:
        cmd.extend(["--output", str(Path(base_args.output).resolve())])
    if base_args.use_processes:
        cmd.append("--use-processes")
    if getattr(base_args, "resume_rerun_retained_tdd_failures", False):
        cmd.append("--resume-rerun-retained-tdd-failures")
    if getattr(base_args, "disable_graph_module", False):
        cmd.append("--disable-graph-module")
    if getattr(base_args, "disable_dependency_graph", False):
        cmd.append("--disable-dependency-graph")
    if getattr(base_args, "disable_graph_module", False) or getattr(base_args, "parent_codegen_dag_source", "dependency") == "none":
        cmd.extend(["--no-graph-seed", str(getattr(base_args, "no_graph_seed", 42))])
    if getattr(base_args, "disable_decomposition", False):
        cmd.append("--disable-decomposition")
    if getattr(base_args, "disable_structure_refinement", False):
        cmd.append("--disable-structure-refinement")
    if getattr(base_args, "disable_strategist", False):
        cmd.append("--disable-strategist")
    if getattr(base_args, "enable_component_metric_actions", False):
        cmd.append("--enable-component-metric-actions")
    if float(getattr(base_args, "component_metric_split_cohesion_threshold", 2.0 / 3.0)) != (2.0 / 3.0):
        cmd.extend(
            [
                "--component-metric-split-cohesion-threshold",
                str(base_args.component_metric_split_cohesion_threshold),
            ]
        )
    if int(getattr(base_args, "component_metric_split_min_subrequirements", 3) or 3) != 3:
        cmd.extend(
            [
                "--component-metric-split-min-subrequirements",
                str(base_args.component_metric_split_min_subrequirements),
            ]
        )
    if float(getattr(base_args, "component_split_min_confidence", 0.70)) != 0.70:
        cmd.extend(
            [
                "--component-split-min-confidence",
                str(base_args.component_split_min_confidence),
            ]
        )
    if getattr(base_args, "enable_component_metric_merge_judge", False):
        cmd.append("--enable-component-metric-merge-judge")
    if int(getattr(base_args, "component_metric_merge_max_small_subrequirements", 1) or 1) != 1:
        cmd.extend(
            [
                "--component-metric-merge-max-small-subrequirements",
                str(base_args.component_metric_merge_max_small_subrequirements),
            ]
        )
    if int(getattr(base_args, "tdd_revise_failure_threshold", 2) or 2) != 2:
        cmd.extend(
            [
                "--tdd-revise-failure-threshold",
                str(base_args.tdd_revise_failure_threshold),
            ]
        )
    if int(getattr(base_args, "action_refinement_rounds", 1) or 1) != 1:
        cmd.extend(["--action-refinement-rounds", str(getattr(base_args, "action_refinement_rounds", 1))])
    if getattr(base_args, "action_refinement_stop_on_stable", False):
        cmd.append("--action-refinement-stop-on-stable")
    if getattr(base_args, "action_refinement_save_stops_component", False):
        cmd.append("--action-refinement-save-stops-component")
    if getattr(base_args, "enable_gap_add_actions", False):
        cmd.append("--enable-gap-add-actions")
    if float(getattr(base_args, "gap_add_proposal_threshold", 0.55)) != 0.55:
        cmd.extend(["--gap-add-proposal-threshold", str(base_args.gap_add_proposal_threshold)])
    if float(getattr(base_args, "gap_add_component_threshold", 0.74)) != 0.74:
        cmd.extend(["--gap-add-component-threshold", str(base_args.gap_add_component_threshold)])
    if float(getattr(base_args, "gap_add_requirement_threshold", 0.82)) != 0.82:
        cmd.extend(["--gap-add-requirement-threshold", str(base_args.gap_add_requirement_threshold)])
    if getattr(base_args, "stop_after_architecture_refinement", False):
        cmd.append("--stop-after-architecture-refinement")
    if getattr(base_args, "component_merge_admission_mode", "strict") != "strict":
        cmd.extend(["--component-merge-admission-mode", str(base_args.component_merge_admission_mode)])
    if float(getattr(base_args, "component_merge_relaxed_best", 0.30)) != 0.30:
        cmd.extend(["--component-merge-relaxed-best", str(base_args.component_merge_relaxed_best)])
    if float(getattr(base_args, "component_merge_relaxed_avg", 0.26)) != 0.26:
        cmd.extend(["--component-merge-relaxed-avg", str(base_args.component_merge_relaxed_avg)])
    if float(getattr(base_args, "component_merge_relaxed_min_pair", 0.20)) != 0.20:
        cmd.extend(["--component-merge-relaxed-min-pair", str(base_args.component_merge_relaxed_min_pair)])
    if float(getattr(base_args, "component_merge_relaxed_dominance_gap", 0.28)) != 0.28:
        cmd.extend(["--component-merge-relaxed-dominance-gap", str(base_args.component_merge_relaxed_dominance_gap)])
    if force_regenerate:
        cmd.append("--force-regenerate")
    if evolve_requirements_file:
        cmd.extend(["--evolve-requirements-file", str(evolve_requirements_file)])
    return cmd


def _prompt_existing_path(
    prompt: str,
    default_path: Path | None = None,
) -> Path | None:
    default_hint = f" [default: {default_path}]" if default_path else ""
    raw = input(f"{prompt}{default_hint}: ").strip()
    if not raw:
        candidate = default_path
    else:
        candidate = Path(raw).expanduser().resolve()
    if candidate is None:
        return None
    if not candidate.exists():
        print(f"Path does not exist: {candidate}")
        return None
    return candidate


def run_session_loop(base_args: argparse.Namespace) -> None:
    """Run an interactive loop for repeated incremental or fresh runs."""
    if not sys.stdin.isatty():
        raise RuntimeError("`--session-loop` requires an interactive terminal.")

    default_requirements = Path(base_args.requirements_file).expanduser().resolve()
    default_evolve = (
        Path(base_args.evolve_requirements_file).expanduser().resolve()
        if base_args.evolve_requirements_file
        else None
    )
    max_rounds = max(0, int(base_args.session_max_rounds or 0))

    print("Repo0 session loop started.")
    print("Choose per round: [e] evolve existing graph, [i] init from scratch, [q] quit.")

    round_index = 0
    while True:
        if max_rounds and round_index >= max_rounds:
            print(f"Reached max rounds: {max_rounds}.")
            break

        round_index += 1
        choice = input(f"\nRound {round_index} action [e/i/q] (default e): ").strip().lower() or "e"
        if choice == "q":
            break
        if choice not in {"e", "i"}:
            print("Invalid action, use e/i/q.")
            round_index -= 1
            continue

        if choice == "i":
            req_file = _prompt_existing_path("Path to base requirements file", default_requirements)
            if req_file is None:
                round_index -= 1
                continue
            default_requirements = req_file
            evolve_file = None
            force_regenerate = True
        else:
            evolve_file = _prompt_existing_path("Path to new evolve requirements file", default_evolve)
            if evolve_file is None:
                round_index -= 1
                continue
            default_evolve = evolve_file
            req_file = default_requirements
            force_regenerate = False

        cmd = build_single_run_command(
            base_args=base_args,
            requirements_file=req_file,
            evolve_requirements_file=evolve_file,
            force_regenerate=force_regenerate,
        )
        print("Executing:", " ".join(shlex.quote(part) for part in cmd))
        completed = subprocess.run(cmd)
        if completed.returncode == 0:
            print(f"Round {round_index} completed successfully.")
        else:
            print(f"Round {round_index} failed with code {completed.returncode}.")
            retry = input("Continue session anyway? [y/N]: ").strip().lower()
            if retry not in {"y", "yes"}:
                break


def _ensure_codegen_tdd_docker_image(
    *, disable_docker: bool, repo0_root: Path, network_host: bool
) -> None:
    """Build DEFAULT_TDD_DOCKER_IMAGE if Docker is used and the image is missing."""
    if disable_docker:
        return
    from agents.coding.code_generator import DEFAULT_TDD_DOCKER_IMAGE

    image = DEFAULT_TDD_DOCKER_IMAGE
    if not shutil.which("docker"):
        logging.warning(
            "TDD uses Docker image %s but docker CLI not found; host must install Docker or pass "
            "--codegen-tdd-disable-docker.",
            image,
        )
        return
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if inspect.returncode == 0:
        logging.info(
            "TDD Docker image %s already exists locally; skipping auto-build.",
            image,
        )
        return
    dockerfile = repo0_root / "docker" / "codegen-tdd" / "Dockerfile"
    if not dockerfile.is_file():
        logging.warning("Cannot auto-build %s: missing %s", image, dockerfile)
        return
    logging.info("TDD Docker image %s not found; building from %s …", image, dockerfile)
    build_cmd = ["docker", "build"]
    if network_host:
        build_cmd.extend(["--network", "host"])
    build_cmd.extend(["-f", str(dockerfile), "-t", image, "."])
    build = subprocess.run(
        build_cmd,
        cwd=str(repo0_root),
        timeout=600,
    )
    if build.returncode != 0:
        logging.warning("docker build for %s exited with code %s", image, build.returncode)


def main() -> None:
    # Configure logging
    args = parse_args()
    if args.session_loop:
        run_session_loop(args)
        return
    workspace = args.workspace.resolve()
    output_dir = (args.output or (workspace / "agents_output")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_code_root = (output_dir / "generated_code").resolve()
    generated_code_root.mkdir(parents=True, exist_ok=True)
    repo_root = (
        Path(args.output).resolve().parent
        if args.output
        else (workspace / "repos" / args.repo).resolve()
    )

    log_level = parse_log_level(args.log_level)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logging.info(f"Current Log Level: {logging._levelToName[log_level]}")

    stage_timing_report_path = output_dir / "stage_timing_report.json"
    existing_stage_timing_report = load_json_if_exists(stage_timing_report_path, force_regenerate=False)
    stage_timer = StageTimer(existing_report=existing_stage_timing_report)

    def _persist_stage_timing_report() -> None:
        try:
            save_json(stage_timer.to_dict(), stage_timing_report_path)
        except Exception as exc:
            logging.debug("Failed to persist stage timing report: %s", exc)

    stage_timer.set_persist_callback(_persist_stage_timing_report)

    def _final_persist_stage_timing_report() -> None:
        try:
            stage_timer.finish()
            save_json(stage_timer.to_dict(), stage_timing_report_path)
        except Exception as exc:
            logging.debug("Failed to finalize stage timing report: %s", exc)

    atexit.register(_final_persist_stage_timing_report)

    def _stage_timer_excepthook(exc_type, exc, tb):
        stage_timer.fail(exc)
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _stage_timer_excepthook

    logging.debug(repo_root)

    logging.debug(f"Workspace: {workspace}")

    if not repo_root.exists():
        raise FileNotFoundError(f"Repository '{args.repo}' not found at {repo_root}")

    force_regen = args.force_regenerate
    if force_regen:
        logging.info("Force regeneration enabled - will recreate all artifacts")

    # Prepare API configuration
    layout_policy = _build_layout_policy(args)
    logging.info(
        "Layout schema: enabled=%s, root=%s, whitelist=%s, alias_map_size=%d",
        layout_policy.get("enabled"),
        layout_policy.get("layout_root"),
        ",".join(layout_policy.get("top_whitelist", [])),
        len(layout_policy.get("alias_map", {})),
    )
    graph_module_disabled = bool(args.disable_graph_module) or str(args.parent_codegen_dag_source or "").strip().lower() == "none"
    dependency_graph_disabled = bool(args.disable_dependency_graph)
    raw_codegen_dag_source = str(args.parent_codegen_dag_source or "dependency").strip().lower()
    if graph_module_disabled:
        effective_codegen_dag_source = "none"
    elif dependency_graph_disabled and raw_codegen_dag_source == "dependency":
        effective_codegen_dag_source = "requirement"
    else:
        effective_codegen_dag_source = raw_codegen_dag_source
    structure_refinement_enabled = not bool(args.disable_structure_refinement)
    component_merge_enabled = not bool(args.disable_component_merge) and structure_refinement_enabled
    component_merge_embedding_analysis = bool(args.enable_component_merge_embedding_analysis) and structure_refinement_enabled
    strategist_enabled = not bool(args.disable_strategist)
    cross_requirement_component_merge_enabled = (
        bool(args.enable_cross_requirement_component_merge)
        and component_merge_enabled
    )
    logging.info(
        "Graph module: enabled=%s, mode=%s, dependency_graph_enabled=%s, no_graph_seed=%d",
        not graph_module_disabled,
        effective_codegen_dag_source,
        not dependency_graph_disabled,
        int(args.no_graph_seed),
    )
    logging.info(
        "Structure refinement: enabled=%s, component_merge=%s, cross_requirement_merge=%s, embedding_analysis=%s, strategist=%s",
        structure_refinement_enabled,
        component_merge_enabled,
        cross_requirement_component_merge_enabled,
        component_merge_embedding_analysis,
        strategist_enabled,
    )

    api_config = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "enable_output_token_routing": bool(args.enable_output_token_routing),
        "short_output_model": args.short_output_model,
        "short_output_max_tokens": max(1, int(args.short_output_max_tokens)),
        "long_output_model": args.long_output_model or args.model,
        "long_output_max_tokens": max(0, int(args.long_output_max_tokens)),
        "output_token_rerun_margin": max(0, int(args.output_token_rerun_margin)),
        "repo": args.repo,
        "path_allowed_roots": layout_policy.get("top_whitelist", []),
        "enable_two_stage_file_plan": bool(layout_policy.get("enabled", True)),
        "enable_skeleton_fill_tdd": True,
        "tdd_max_fix_retries": max(0, int(args.codegen_tdd_max_fix_retries)),
        "tdd_docker_image": "",
        "tdd_disable_docker": bool(args.codegen_tdd_disable_docker),
        "tdd_docker_network_host": bool(args.codegen_tdd_docker_network_host),
        "tdd_pip_project_root": str(generated_code_root),
        "tdd_pip_timeout": max(30, int(args.codegen_tdd_pip_timeout)),
        "tdd_missing_module_pip_retries": max(0, int(args.codegen_tdd_missing_module_pip_retries)),
        "import_postcheck_max_fix_attempts": max(0, int(args.import_postcheck_max_fix_attempts)),
        "package_postcheck_max_fix_attempts": max(0, int(args.package_postcheck_max_fix_attempts)),
    }

    repo0_root = Path(__file__).resolve().parent
    _ensure_codegen_tdd_docker_image(
        disable_docker=bool(args.codegen_tdd_disable_docker),
        repo0_root=repo0_root,
        network_host=bool(args.codegen_tdd_docker_network_host),
    )

    memory_agent = MemoryAgent(workspace, repos_dir=str(repo_root.parent))
    memory_path = output_dir / "memory.json"
    if memory_path.exists() and not force_regen:
        memory_agent.load_snapshot(memory_path)
        logging.info(f"Loaded existing memory snapshot from {memory_path}")
    else:
        memory_agent.build_memory(args.repo)

    generated_files_probe = output_dir / "generated_files.json"
    existing_generated_probe = load_json_if_exists(generated_files_probe, force_regen)
    if isinstance(existing_generated_probe, list):
        from agents import CodeGeneratorAgent
        probe_generator = CodeGeneratorAgent(api_config=api_config, output_dir=str(output_dir))
        existing_generated_probe, repaired_probe_count = _repair_existing_resume_postcheck_failures(
            generated_entries=existing_generated_probe,
            code_generator=probe_generator,
            code_output_dir=output_dir / "generated_code",
            implemented_components_context=memory_agent.format_implementations_for_prompt(
                status_filter="implemented"
            ),
        )
        if repaired_probe_count:
            logging.info(
                "Repaired %d existing generated component files via resume postchecks before resume",
                repaired_probe_count,
            )
            save_json(existing_generated_probe, generated_files_probe)
        realization_probe_report = _persist_component_realization_report(
            output_dir,
            existing_generated_probe,
        )
        _persist_component_import_postcheck_report(
            output_dir,
            existing_generated_probe,
        )
        _log_existing_component_realization_summary(realization_probe_report)
    else:
        logging.info(
            "Existing generated component status: passed=0 failed=0 total=0 (no reusable generated entries found)"
        )

    # Allow overriding the default README.req path via CLI
    if args.req_path:
        readme_req_path = args.req_path
    else:
        readme_req_path = repo_root / "README.req"

    requirements_path = args.requirements_file
    readme_output_dir = requirements_path.parent
    
    with stage_timer.stage("requirements_generation"):
        if not requirements_path.exists() or force_regen:
            if readme_req_path.exists():
                logging.info(f"Generating requirements.json from {readme_req_path}")
                readme_content = readme_req_path.read_text(encoding="utf-8")
                
                # Initialize LLM client for requirements parsing
                from agents.llm_client import LLMClient
                req_llm_client = LLMClient(api_config, str(output_dir), agent_name="rqmts_parser")
                
                # Requirements parsing can use parallel extraction if implemented
                generate_and_save_one_requirements(
                    readme_content, 
                    str(readme_output_dir), 
                    req_llm_client
                )
                if not requirements_path.exists():
                    raise FileNotFoundError(
                        f"Requirements generation finished without creating file: {requirements_path}"
                    )
                logging.info(f"Requirements generated and saved to {requirements_path}")
            else:
                logging.warning(f"README.req not found at {readme_req_path}, skipping requirements generation")
        else:
            logging.info(f"Using existing requirements.json at {requirements_path}")

    # Generate standalone merged-requirements artifact
    merge_output_path = output_dir / "requirements_merge_result.json"
    merge_result: Optional[dict] = None
    with stage_timer.stage("requirement_merge"):
        if args.skip_merge:
            logging.info("Skipping requirement merge step (--skip-merge enabled).")
        else:
            existing_merge_data = load_json_if_exists(merge_output_path, force_regen)
            if existing_merge_data:
                logging.info(f"Using existing requirement merge result at {merge_output_path}")
                merge_result = existing_merge_data
            else:
                if requirements_path.exists():
                    try:
                        req_payload = json.loads(requirements_path.read_text(encoding="utf-8"))
                        merge_agent = RequirementMergeAgent(api_config=api_config, output_dir=str(output_dir))
                        merge_result = merge_agent.merge_requirements(req_payload)
                        save_json(merge_result, merge_output_path)
                        logging.info(
                            "Requirement merge result saved to %s",
                            merge_output_path,
                        )
                    except Exception as exc:
                        logging.warning("Requirement merge step failed, continuing pipeline unchanged: %s", exc)
                else:
                    logging.warning("requirements.json not found at %s, skipping requirement merge step", requirements_path)

    # Prepare DAG requirements from merge result (fallback to original requirements)
    dag_requirements_path = output_dir / "requirements_for_dag.json"
    with stage_timer.stage("dag_requirements_preparation"):
        existing_dag_requirements = load_json_if_exists(
            dag_requirements_path,
            force_regen or args.skip_merge,
        )
        if existing_dag_requirements:
            logging.info(f"Using existing DAG requirements at {dag_requirements_path}")
        else:
            dag_requirements_items: List[dict] = []
            dag_requirements_source = "original"
            if merge_result is not None:
                dag_requirements_items = extract_requirements_items(merge_result)
                if dag_requirements_items:
                    dag_requirements_source = "merged"
            if not dag_requirements_items and requirements_path.exists():
                try:
                    dag_requirements_items = extract_requirements_items(
                        json.loads(requirements_path.read_text(encoding="utf-8"))
                    )
                except Exception as exc:
                    logging.warning("Failed to load base requirements for DAG: %s", exc)
            save_json({"requirements": dag_requirements_items}, dag_requirements_path)
            logging.info(
                "DAG requirements saved to %s (source=%s, count=%d)",
                dag_requirements_path,
                dag_requirements_source,
                len(dag_requirements_items),
            )

    # Generate edges for DAG requirements (merge-aware)
    edges_path = output_dir / "edges_for_dag.json"
    
    with stage_timer.stage("edge_generation"):
        if not edges_path.exists() or force_regen or args.skip_merge:
            if dag_requirements_path.exists():
                logging.info(f"Generating DAG edges from {dag_requirements_path}")
                requirements_content = dag_requirements_path.read_text(encoding="utf-8")
                
                # Initialize LLM client for graph parsing
                from agents.llm_client import LLMClient
                graph_llm_client = LLMClient(api_config, str(output_dir), agent_name="graph_parser")
                
                # Edge generation typically single call, but could batch analyze requirements
                generate_and_save_edges(
                    requirements_content,
                    str(output_dir),
                    graph_llm_client,
                    output_filename=edges_path.name,
                )
                logging.info(f"Edges generated and saved to {edges_path}")
            else:
                logging.warning(f"DAG requirements not found at {dag_requirements_path}, skipping edges generation")
        else:
            logging.info(f"Using existing edges.json at {edges_path}")

    # Load or generate requirement DAG
    dag_path = output_dir / "requirement_dag.json"
    dag_force_regen = force_regen or outputs_stale(dag_path, [dag_requirements_path, edges_path])
    with stage_timer.stage("dag_build"):
        existing_dag = load_json_if_exists(dag_path, dag_force_regen)
        if existing_dag:
            logging.info("Loading existing requirement DAG from %s for incremental evolution", dag_path)
            dag = RequirementDAG.from_dict(existing_dag)
            dag_inputs_changed = False
        else:
            dag_inputs_changed = True
            logging.info(
                "Building initial requirement DAG from merged-aware artifacts: requirements=%s, edges=%s",
                dag_requirements_path,
                edges_path,
            )
            dag = RequirementDAG.from_files(dag_requirements_path, edges_path)
            save_json(dag.to_dict(), dag_path)
            logging.info("Requirement DAG built and saved to %s", dag_path)

    # Load or generate decomposed DAG
    decomposed_dag_path = output_dir / "decomposed_dag.json"
    decomposed_force_regen = force_regen or dag_inputs_changed or outputs_stale(
        decomposed_dag_path,
        [dag_path],
    )
    with stage_timer.stage("dag_decomposition"):
        existing_decomposed = load_json_if_exists(decomposed_dag_path, decomposed_force_regen)

        if args.disable_decomposition:
            logging.info("Requirement DAG decomposition disabled; using original requirement DAG as planning DAG")
            decomposed_dag = clone_requirement_dag(dag)
            save_json(decomposed_dag.to_dict(), decomposed_dag_path)
        elif existing_decomposed:
            logging.info(f"Loading existing decomposed DAG from {decomposed_dag_path}")
            decomposed_dag = RequirementDAG.from_dict(existing_decomposed)
        else:
            logging.info("Decomposing requirement DAG")
            architect = ArchitectAgent(
                api_config=api_config, 
                output_dir=str(output_dir),
                max_workers=args.max_workers  # Pass max_workers for parallel decomposition
            )
            decomposed_dag = architect.decompose_dag(dag)
            save_json(decomposed_dag.to_dict(), decomposed_dag_path)

    # Apply DAG evolution if new requirements are provided
    evolution_applied = False
    evolution_parents: set[str] = set()
    evolution_source_parents_by_target: dict[str, set[str]] = {}
    evolution_operation_records: List[dict[str, Any]] = []
    with stage_timer.stage("dag_evolution"):
        if args.evolve_requirements_file:
            logging.debug(f"Evolving requirements from {args.evolve_requirements_file}")

            if not args.evolve_requirements_file.exists():
                logging.error(f"New requirements file {args.evolve_requirements_file} does not exist.")

            new_requirements = load_new_requirements(args.evolve_requirements_file)

            if not new_requirements:
                logging.warning(f"No new requirements found in {args.evolve_requirements_file}")
            else:
                logging.info(f"Applying DAG evolution for {len(new_requirements)} new requirements")
                evolution_agent = DAGEvolutionAgent(
                    dag=dag,
                    memory_agent=memory_agent,
                    api_config=api_config,
                    output_dir=str(output_dir)
                )
                strategist = StrategistAgent(
                    api_config=api_config,
                    output_dir=str(output_dir)
                )
                architect = ArchitectAgent(
                    api_config=api_config,
                    output_dir=str(output_dir),
                    max_workers=args.max_workers
                )

                for requirement in new_requirements:
                    requirement_payload = {
                        "name": requirement.name,
                        "description": requirement.description,
                        **requirement.metadata,
                    }
                    logging.debug(requirement)
                    decision = strategist.choose_dag_operation(
                        requirement_payload,
                        dag
                    )
                    action_override = build_evolution_action_override(decision)
                    if action_override is None:
                        logging.info(f"Skipping evolution for existing requirement: {requirement.name}")
                        continue

                    result = evolution_agent.evolve_requirement_with_subnodes(
                        requirement,
                        context=decision.get("reason", "Evolution input from CLI"),
                        decomposed_dag=decomposed_dag,
                        architect=architect,
                        action_override=action_override
                    )

                    if result.get("operation_record"):
                        op_record = result["operation_record"]
                        memory_agent.record_dag_operation(op_record)
                        evolution_operation_records.append(op_record)
                    if result.get("success"):
                        evolution_applied = True
                    else:
                        logging.warning(f"Evolution failed for {requirement.name}: {result.get('error')}")

                if evolution_applied:
                    save_json(dag.to_dict(), dag_path)
                    save_json(decomposed_dag.to_dict(), decomposed_dag_path)
                    logging.info("DAG evolution applied; refreshed DAG artifacts saved")

    if evolution_applied:
        evolution_summary = summarize_evolution_operations(
            evolution_operation_records,
            active_parents=set(dag.nodes.keys()),
        )
        evolution_parents = set(evolution_summary["regen_parents"])
        evolution_source_parents_by_target = {
            key: set(value)
            for key, value in evolution_summary["source_parents_by_target"].items()
        }
        logging.info(
            "Evolution regeneration scope: %d parents, with %d mapped source parent sets",
            len(evolution_parents),
            len(evolution_source_parents_by_target),
        )
    
    # Load or generate plan
    plan_path = output_dir / "plan.json"
    incremental_mode = evolution_applied and not force_regen and bool(evolution_parents)
    active_parent_names = set(dag.nodes.keys())
    should_regen = force_regen or dag_inputs_changed
    existing_plan = load_json_if_exists(plan_path, should_regen)
    
    with stage_timer.stage("execution_plan"):
        if graph_module_disabled and not incremental_mode:
            dag_requirement_payload = load_json_if_exists(dag_requirements_path, False) or {}
            dag_requirement_items = extract_requirements_items(dag_requirement_payload)
            plan = build_no_graph_plan(dag_requirement_items)
            if not plan:
                raise RuntimeError("No-graph planning produced an empty plan. Check requirements_for_dag.json.")
            save_json({"plan": plan}, plan_path)
            logging.info(
                "Created graph-free execution plan directly from requirements: parents=%d seed=%d",
                len(plan),
                int(args.no_graph_seed),
            )
        elif existing_plan and "plan" in existing_plan and not incremental_mode:
            logging.info(f"Loading existing plan from {plan_path}")
            plan = existing_plan["plan"]
        else:
            if existing_plan and "plan" in existing_plan and incremental_mode:
                logging.info("Updating execution plan incrementally")
                incremental_dag = build_parent_filtered_dag(decomposed_dag, evolution_parents)
                if incremental_dag.nodes:
                    planner = PlannerAgent(
                        max_items=len(incremental_dag.nodes),
                        api_config=api_config,
                        output_dir=str(output_dir),
                    )
                    incremental_plan = planner.create_plan_from_dag(
                        incremental_dag,
                        load_requirements(args)
                    )
                else:
                    incremental_plan = []
                plan = merge_plans(
                    existing_plan["plan"],
                    incremental_plan,
                    evolution_parents,
                    decomposed_dag,
                )
                save_json({"plan": plan}, plan_path)
            else:
                logging.info("Creating execution plan")
                total_nodes = len(decomposed_dag.nodes) if hasattr(decomposed_dag, 'nodes') else 100
                planner = PlannerAgent(
                    max_items=total_nodes, 
                    api_config=api_config, 
                    output_dir=str(output_dir),
                )
                plan = planner.create_plan_from_dag(decomposed_dag, load_requirements(args))
                if not plan:
                    raise RuntimeError("Planner produced an empty plan. Check requirements and edges artifacts.")
                save_json({"plan": plan}, plan_path)

    # Load or generate architectures for all tasks (grouped by parent)
    architectures_path = output_dir / "architectures.json"
    component_merge_report_path = output_dir / "component_merge_report.json"
    component_merge_embedding_report_path = output_dir / "component_merge_embedding_report.json"
    component_merge_agent: Optional[ComponentMergeAgent] = None
    component_split_agent: Optional[ComponentSplitAgent] = None
    module_assignment_agent = ModuleAssignmentAgent(
        api_config=api_config,
        output_dir=str(output_dir),
    )
    module_planning_agent = ModulePlanningAgent(
        api_config=api_config,
        output_dir=str(output_dir),
    )
    if component_merge_enabled or component_merge_embedding_analysis:
        component_merge_agent = ComponentMergeAgent(
            api_config=api_config,
            output_dir=str(output_dir),
            enable_embedding_analysis=component_merge_embedding_analysis,
            emb_threshold=float(args.component_merge_embedding_threshold),
            emb_dominance_gap=float(args.component_merge_dominance_gap),
            emb_name_weight=float(args.component_merge_name_weight),
            emb_resp_weight=float(args.component_merge_resp_weight),
            emb_subreq_weight=float(args.component_merge_subreq_weight),
            merge_admission_mode=str(args.component_merge_admission_mode),
            merge_relaxed_best=float(args.component_merge_relaxed_best),
            merge_relaxed_avg=float(args.component_merge_relaxed_avg),
            merge_relaxed_min_pair=float(args.component_merge_relaxed_min_pair),
            merge_relaxed_dominance_gap=float(args.component_merge_relaxed_dominance_gap),
        )
    if structure_refinement_enabled:
        component_split_agent = ComponentSplitAgent(
            api_config=api_config,
            output_dir=str(output_dir),
            enable_llm_split=True,
            split_min_confidence=float(args.component_split_min_confidence),
        )

    stage_timer.begin("architecture_generation")
    existing_architectures_raw = load_json_if_exists(architectures_path, should_regen)
    existing_component_merge_report = load_json_if_exists(component_merge_report_path, should_regen)
    existing_component_merge_embedding_report = load_json_if_exists(
        component_merge_embedding_report_path,
        should_regen,
    )
    fixed_component_merge_input: Optional[Path] = args.component_merge_input
    if fixed_component_merge_input is not None:
        if not fixed_component_merge_input.exists():
            raise FileNotFoundError(f"Component merge input snapshot not found: {fixed_component_merge_input}")
        with open(fixed_component_merge_input, "r", encoding="utf-8") as f:
            existing_architectures_raw = json.load(f)
        existing_component_merge_report = None
        existing_component_merge_embedding_report = None
        logging.info(
            "Using fixed component merge input snapshot and forcing component merge recomputation: %s",
            fixed_component_merge_input,
        )
    legacy_components_by_parent = build_parent_component_index(existing_architectures_raw)
    existing_architectures = existing_architectures_raw
    if existing_architectures:
        existing_count = len(existing_architectures) if isinstance(existing_architectures, list) else 0
        existing_architectures = filter_architectures_for_active_parents(
            existing_architectures,
            active_parent_names,
        )
        dropped = existing_count - len(existing_architectures)
        if dropped > 0:
            logging.info(
                "Dropped %d stale architecture entries that no longer map to DAG parents",
                dropped,
            )
    
    completed_architecture_parents: set[str] = set()
    if existing_architectures and not incremental_mode:
        completed_architecture_parents = detect_completed_architecture_parents(
            existing_architectures,
            active_parent_names,
        )
        completed_component_merge_parents = detect_completed_component_merge_parents(
            existing_component_merge_report,
            active_parent_names,
        )
        if completed_architecture_parents:
            logging.info(
                "Architecture resume detected %d/%d completed parents",
                len(completed_architecture_parents),
                len(active_parent_names),
            )
        if completed_component_merge_parents:
            logging.info(
                "Component merge resume detected %d/%d completed parents",
                len(completed_component_merge_parents),
                len(active_parent_names),
            )
        if completed_architecture_parents == active_parent_names:
            logging.info(f"Loading existing architectures from {architectures_path}")
            architectures = filter_architectures_for_active_parents(
                existing_architectures,
                active_parent_names,
            )
            if completed_component_merge_parents == active_parent_names and existing_component_merge_report:
                component_merge_report = existing_component_merge_report
                component_merge_embedding_report = existing_component_merge_embedding_report
                logging.info(
                    "Loading existing component merge report from %s and skipping merge recomputation",
                    component_merge_report_path,
                )
            else:
                save_component_merge_input_snapshot(
                    architectures=architectures,
                    output_dir=output_dir,
                    repo=args.repo,
                    source_path=fixed_component_merge_input,
                    requirements_path=args.requirements_file,
                    active_parent_names=active_parent_names,
                )
                architectures, component_merge_report, component_merge_embedding_report = apply_component_merge_to_architectures(
                    architectures,
                    component_merge_agent,
                    component_split_agent,
                    apply_merge=component_merge_enabled,
                    enable_embedding_analysis=component_merge_embedding_analysis,
                    enable_cross_requirement_merge=cross_requirement_component_merge_enabled,
                )
                save_json(component_merge_report, component_merge_report_path)
                if component_merge_embedding_report is not None:
                    save_json(component_merge_embedding_report, component_merge_embedding_report_path)
                save_json(architectures, architectures_path)
            logging.info(
                "Component normalize summary: before=%d after=%d merged=%d split=%d parents_with_merge=%d parents_with_split=%d",
                component_merge_report.get("stats", {}).get("components_before", 0),
                component_merge_report.get("stats", {}).get("components_after", 0),
                component_merge_report.get("stats", {}).get("merged_components", 0),
                component_merge_report.get("stats", {}).get("split_components", 0),
                component_merge_report.get("stats", {}).get("parents_with_merge", 0),
                component_merge_report.get("stats", {}).get("parents_with_split", 0),
            )
            register_architecture_components(memory_agent, architectures)
        else:
            logging.info(
                "Architecture stage will continue unfinished parents only: remaining=%d",
                len(active_parent_names - completed_architecture_parents),
            )
    if not (existing_architectures and not incremental_mode and completed_architecture_parents == active_parent_names):
        logging.info(f"Generating parent-level architectures (sequential to track memory)")
        dag_summary_data = {} if graph_module_disabled else decomposed_dag.summary()
        architecture_via_parent_groups = not bool(args.disable_decomposition)
        
        if architecture_via_parent_groups:
            # Group tasks by parent to reduce component explosion
            # Use original DAG to group decomposed tasks by their original parent requirements
            if graph_module_disabled:
                parent_groups = build_no_graph_parent_groups(plan)
                logging.info("Using graph-free parent grouping: %d parent groups with no sub-task DAG structure", len(parent_groups))
            else:
                parent_groups = group_tasks_by_parent(dag, decomposed_dag, plan)
            if incremental_mode and existing_architectures:
                parent_groups = [
                    (parent, subs)
                    for parent, subs in parent_groups
                    if parent.get("name") in evolution_parents
                ]
            logging.info(f"Processing {len(parent_groups)} parent groups instead of {len(plan)} individual tasks")
        else:
            parent_groups = []
            logging.info(
                "Decomposition disabled: bypassing parent-level architecture synthesis and generating direct task architectures for %d plan items",
                len(plan),
            )
        
        # Track existing modules across parent groups to avoid duplication
        existing_modules = []
        architectures = []
        existing_arch_by_parent: dict[str, dict] = {}
        if existing_architectures and not incremental_mode:
            for arch in existing_architectures:
                parent = _parent_from_architecture_entry(arch)
                if parent:
                    existing_arch_by_parent[parent] = arch
            architectures = [
                existing_arch_by_parent[parent]
                for parent in [item.get("name") for item, _ in parent_groups]
                if isinstance(parent, str) and parent in completed_architecture_parents and parent in existing_arch_by_parent
            ]
            register_architecture_components(memory_agent, architectures)

        if incremental_mode and existing_architectures:
            for arch in existing_architectures:
                existing_modules.extend(arch.get("architecture", {}).get("components", []))
        elif architectures:
            for arch in architectures:
                existing_modules.extend(arch.get("architecture", {}).get("components", []))
        
        # Process architecture generation sequentially so later items can see memory from earlier ones.
        prev_parent_task = None
        if architecture_via_parent_groups:
            for i, (parent, subs) in enumerate(parent_groups):
                parent_name = parent.get('name', 'Unknown')
                if not incremental_mode and parent_name in completed_architecture_parents:
                    prev_parent_task = parent
                    continue
                parent_context_modules = list(existing_modules)
                if incremental_mode:
                    source_parents = sorted(evolution_source_parents_by_target.get(parent_name, set()))
                    if source_parents:
                        source_components: List[dict] = []
                        for source_parent in source_parents:
                            source_components.extend(legacy_components_by_parent.get(source_parent, []))
                        if source_components:
                            parent_context_modules.extend(source_components)
                            parent_context_modules = dedupe_components_by_name(parent_context_modules)
                            logging.info(
                                "  Injected %d source components for parent '%s' from %s",
                                len(source_components),
                                parent_name,
                                ", ".join(source_parents),
                            )
                logging.info(f"\n{'='*60}")
                logging.info(f"Processing parent group {i+1}/{len(parent_groups)}: {parent_name}")
                logging.info(f"  Sub-requirements: {len(subs)}")
                logging.info(f"  Existing modules from previous parents: {len(parent_context_modules)}")
                logging.info(f"{'='*60}\n")

                try:
                    _, arch_result = process_parent_architecture_task(
                        (i, parent, subs),
                        api_config,
                        str(output_dir),
                        memory_agent,
                        parent_context_modules,
                        dag_summary_data
                    )

                    arch_result["parent_prev_node"] = prev_parent_task
                    architectures.append(arch_result)
                    save_json(architectures, architectures_path)
                    memory_agent.persist(output_dir)

                    components = arch_result.get("architecture", {}).get("components", [])
                    existing_modules.extend(components)
                    for comp in components:
                        comp_name = comp.get("name", "UnknownComponent")
                        serves_subreqs = comp.get("serves_subrequirements", [])
                        memory_agent.register_component_implementation(
                            component_name=comp_name,
                            requirement_node=parent_name,
                            file_path=f"planned/{comp_name}.py",
                            class_names=[comp_name],
                            function_signatures=[],
                            dependencies=[],
                            exports=comp.get("responsibilities", []),
                            status="planned",
                            responsibilities=comp.get("responsibilities", []),
                            serves_subrequirements=serves_subreqs,
                            parent_requirement=parent_name
                        )

                    logging.info(f"Registered {len(components)} components from parent '{parent_name}' in memory")
                    prev_parent_task = parent

                except Exception as e:
                    logging.error(f"Failed to generate architecture for parent {i}: {parent_name} - {e}")
                    raise
        else:
            direct_plan = list(plan)
            if incremental_mode:
                direct_plan = [
                    task for task in direct_plan
                    if get_task_parent_name(task.get("name", ""), decomposed_dag) in evolution_parents
                    or task.get("name") in evolution_parents
                ]
            for i, task in enumerate(direct_plan):
                task_name = task.get("name", "Unknown")
                if not incremental_mode and task_name in completed_architecture_parents:
                    prev_parent_task = task
                    continue
                logging.info(f"\n{'='*60}")
                logging.info(f"Processing direct architecture task {i+1}/{len(direct_plan)}: {task_name}")
                logging.info(f"  Existing modules from previous tasks: {len(existing_modules)}")
                logging.info(f"{'='*60}\n")
                implemented_components_desc = memory_agent.format_implementations_for_prompt()
                try:
                    _, arch_result = process_architecture_task(
                        (i, task),
                        api_config,
                        str(output_dir),
                        implemented_components_desc,
                        dag_summary_data,
                    )
                    arch_result["parent_prev_node"] = prev_parent_task
                    architectures.append(arch_result)
                    save_json(architectures, architectures_path)
                    memory_agent.persist(output_dir)

                    components = arch_result.get("architecture", {}).get("components", [])
                    existing_modules.extend(components)
                    for comp in components:
                        comp_name = comp.get("name", "UnknownComponent")
                        memory_agent.register_component_implementation(
                            component_name=comp_name,
                            requirement_node=task_name,
                            file_path=f"planned/{comp_name}.py",
                            class_names=[comp_name],
                            function_signatures=[],
                            dependencies=[],
                            exports=comp.get("responsibilities", []),
                            status="planned",
                            responsibilities=comp.get("responsibilities", []),
                            serves_subrequirements=comp.get("serves_subrequirements", []),
                            parent_requirement=task_name,
                        )
                    logging.info(f"Registered {len(components)} components from task '{task_name}' in memory")
                    prev_parent_task = task
                except Exception as e:
                    logging.error(f"Failed to generate direct architecture for task {i}: {task_name} - {e}")
                    raise
        
        if incremental_mode and existing_architectures:
            updated_by_parent = {
                _parent_from_architecture_entry(arch): arch
                for arch in architectures
                if _parent_from_architecture_entry(arch)
            }
            merged_architectures = []
            for arch in existing_architectures:
                parent_task = _parent_from_architecture_entry(arch)
                if not parent_task:
                    continue
                merged_architectures.append(updated_by_parent.pop(parent_task, arch))
            merged_architectures.extend(updated_by_parent.values())
            architectures = merged_architectures

        architectures = filter_architectures_for_active_parents(
            architectures,
            active_parent_names,
        )

        save_component_merge_input_snapshot(
            architectures=architectures,
            output_dir=output_dir,
            repo=args.repo,
            source_path=fixed_component_merge_input,
            requirements_path=args.requirements_file,
            active_parent_names=active_parent_names,
        )
        architectures, component_merge_report, component_merge_embedding_report = apply_component_merge_to_architectures(
            architectures,
            component_merge_agent,
            component_split_agent,
            apply_merge=component_merge_enabled,
            enable_embedding_analysis=component_merge_embedding_analysis,
            enable_cross_requirement_merge=cross_requirement_component_merge_enabled,
        )
        save_json(component_merge_report, component_merge_report_path)
        if component_merge_embedding_report is not None:
            save_json(component_merge_embedding_report, component_merge_embedding_report_path)
        logging.info(
            "Component normalize summary: before=%d after=%d merged=%d split=%d parents_with_merge=%d parents_with_split=%d",
            component_merge_report.get("stats", {}).get("components_before", 0),
            component_merge_report.get("stats", {}).get("components_after", 0),
            component_merge_report.get("stats", {}).get("merged_components", 0),
            component_merge_report.get("stats", {}).get("split_components", 0),
            component_merge_report.get("stats", {}).get("parents_with_merge", 0),
            component_merge_report.get("stats", {}).get("parents_with_split", 0),
        )

        save_json(architectures, architectures_path)
        
        _rebuild_flattened_architectures(output_dir, architectures)
    logging.info(
        "Architecture/component stage completed: parents=%d architectures=%d path=%s",
        len(active_parent_names),
        len(architectures),
        architectures_path,
    )
    stage_timer.end("architecture_generation")

    generated_files_path = output_dir / "generated_files.json"
    gap_addition_report: Dict[str, Any] | None = None
    if bool(getattr(args, "enable_gap_add_actions", False)):
        with stage_timer.stage("gap_addition"):
            gap_inputs = build_gap_add_stage_inputs(
                requirements_file=args.requirements_file,
                req_path=args.req_path,
                generated_files_path=generated_files_path,
                realization_report_path=output_dir / "component_realization_report.json",
            )
            architectures, gap_addition_report = run_gap_addition_stage(
                architectures=architectures,
                args=args,
                output_dir=output_dir,
                input_text=gap_inputs["input_text"],
                requirements_payload=gap_inputs["requirements_payload"],
                generated_entries=gap_inputs["generated_entries"],
                realization_report=gap_inputs["realization_report"],
                component_merge_agent=component_merge_agent,
                component_split_agent=component_split_agent,
            )
        if gap_addition_report is not None:
            save_json(gap_addition_report, output_dir / "gap_addition_report.json")
            save_json(architectures, architectures_path)
            _rebuild_flattened_architectures(output_dir, architectures)

    with stage_timer.stage("layout_plan_initial"):
        layout_grouping_report, package_api_plan = _refresh_layout_plan_artifacts(
            output_dir=output_dir,
            architectures=architectures,
            layout_policy=layout_policy,
        )
        logging.info(
            "Canonical grouping ready: candidates=%d, assigned_components=%d, default=%s",
            len(layout_policy.get("canonical_packages", [])),
            layout_grouping_report.get("stats", {}).get("component_count", 0)
            if isinstance(layout_grouping_report, dict)
            else 0,
            layout_policy.get("default_subpackage"),
        )
        component_plan_index = package_api_plan.get("component_index", {}) if isinstance(package_api_plan, dict) else {}
        logging.info(
            "Package API plan generated: packages=%d, components=%d",
            package_api_plan.get("package_count", 0) if isinstance(package_api_plan, dict) else 0,
            package_api_plan.get("component_count", 0) if isinstance(package_api_plan, dict) else 0,
        )

    existing_generated_for_memory = load_json_if_exists(generated_files_path, should_regen)
    if existing_generated_for_memory and not force_regen:
        from agents import CodeGeneratorAgent
        rehydrate_generator = CodeGeneratorAgent(api_config=api_config, output_dir=str(output_dir))
        rehydrate_memory_from_generated_artifacts(
            memory_agent=memory_agent,
            code_generator=rehydrate_generator,
            code_output_dir=output_dir / "generated_code",
            generated_entries=existing_generated_for_memory,
            module_registry={},
            parent_scope=active_parent_names,
            rerun_retained_tdd_failures=bool(args.resume_rerun_retained_tdd_failures),
        )

    # Keep component memory aligned with current DAG before graph construction.
    pruned_components = prune_memory_components_for_active_parents(
        memory_agent,
        active_parent_names,
    )
    if pruned_components:
        logging.info(
            "Pruned %d stale component-memory entries not present in current DAG",
            pruned_components,
        )

    # Build or reuse dependency graph from implemented components and product requirement edges.
    dependency_graph_path = output_dir / "dependency_graph.json"
    existing_dependency_graph = load_json_if_exists(dependency_graph_path, should_regen)
    dependency_graph_payload: dict[str, Any] | None = None
    if graph_module_disabled:
        logging.info("Skipping dependency graph generation because the graph module is disabled")
        dependency_graph_payload = {
            "components": [],
            "requirement_edges": [],
            "dependency_graph": {"edges": [], "component_edges": [], "unresolved": []},
        }
    elif dependency_graph_disabled:
        logging.info("Skipping dependency graph generation because --disable-dependency-graph is enabled")
        dependency_graph_payload = {
            "components": [],
            "requirement_edges": build_requirement_edges(dag),
            "dependency_graph": {"edges": [], "component_edges": [], "unresolved": []},
        }
    elif existing_dependency_graph and not incremental_mode:
        logging.info(
            "Loading existing dependency graph from %s and skipping recomputation",
            dependency_graph_path,
        )
        dependency_graph_payload = existing_dependency_graph
    else:
        dependency_graph_payload = generate_dependency_graph_artifacts(
            api_config,
            output_dir,
            memory_agent,
            dag,
        )

    codegen_parent_dag = build_codegen_parent_dag(
        dag_source=effective_codegen_dag_source,
        requirement_dag=dag,
        dependency_graph_payload=dependency_graph_payload,
    )


    # Load or generate actions for all architectures
    actions_path = output_dir / "actions.json"
    stage_timer.begin("action_selection")
    if not strategist_enabled:
        logging.info("Strategist disabled; skipping action selection and writing empty action hints")
        all_actions = build_empty_actions_for_architectures(architectures)
        save_json(all_actions, actions_path)
    elif bool(getattr(args, "enable_component_metric_actions", False)):
        logging.info("Metric-guided action mode enabled; skipping strategist baseline and starting from empty action hints")
        existing_actions = load_json_if_exists(actions_path, should_regen)
        if existing_actions and not incremental_mode:
            logging.info(f"Loading existing actions from {actions_path}")
            all_actions = merge_actions_for_architectures(
                existing_actions,
                [],
                architectures,
            )
        else:
            all_actions = _build_default_empty_actions(architectures)
            metric_merge_judge = None
            if bool(getattr(args, "enable_component_metric_merge_judge", False)):
                metric_merge_judge = _build_metric_merge_judge(component_merge_agent)
            all_actions, component_metric_action_report = augment_actions_with_component_metrics(
                architectures=architectures,
                actions=all_actions,
                decomposed_dag=decomposed_dag,
                split_cohesion_threshold=float(args.component_metric_split_cohesion_threshold),
                split_min_subrequirements=int(args.component_metric_split_min_subrequirements),
                merge_judge=metric_merge_judge,
                merge_max_small_subrequirements=int(args.component_metric_merge_max_small_subrequirements),
            )
            save_json(
                component_metric_action_report,
                output_dir / "component_metric_action_report.json",
            )
            save_json(all_actions, actions_path)
    else:
        existing_actions = load_json_if_exists(actions_path, should_regen)
        completed_action_parents: set[str] = set()
        if existing_actions and not incremental_mode:
            completed_action_parents = detect_completed_action_parents(
                existing_actions,
                {
                    _parent_from_architecture_entry(arch)
                    for arch in architectures
                    if _parent_from_architecture_entry(arch)
                },
            )
            if completed_action_parents == {
                _parent_from_architecture_entry(arch)
                for arch in architectures
                if _parent_from_architecture_entry(arch)
            }:
                logging.info(f"Loading existing actions from {actions_path}")
                all_actions = merge_actions_for_architectures(
                    existing_actions,
                    [],
                    architectures,
                )
            else:
                logging.info(
                    "Action stage will continue unfinished parents only: remaining=%d",
                    len(architectures) - len(completed_action_parents),
                )
                existing_actions = existing_actions if isinstance(existing_actions, list) else []
                target_action_architectures = [
                    arch for arch in architectures
                    if _parent_from_architecture_entry(arch) not in completed_action_parents
                ]
                logging.info(f"Choosing actions for {len(target_action_architectures)} remaining architectures (parallel workers: {args.max_workers})")
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    future_to_idx = {
                        executor.submit(
                            process_action_task,
                            arch_info,
                            api_config,
                            str(output_dir)
                        ): i for i, arch_info in enumerate(target_action_architectures)
                    }
                    new_actions_partial = [None] * len(target_action_architectures)
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            action_result = future.result()
                            new_actions_partial[idx] = action_result
                            logging.info(
                                "  ✓ Completed actions for remaining architecture %d/%d",
                                idx + 1,
                                len(target_action_architectures),
                            )
                        except Exception as e:
                            logging.error(f"Failed to choose actions for architecture {idx}: {e}")
                            raise
                all_actions = merge_actions_for_architectures(
                    existing_actions,
                    [entry for entry in new_actions_partial if isinstance(entry, dict)],
                    architectures,
                )
                save_json(all_actions, actions_path)
        else:
            logging.info(f"Choosing actions for all architectures (parallel workers: {args.max_workers})")

            # Use thread pool for I/O-bound LLM API calls
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        process_action_task,
                        arch_info,
                        api_config,
                        str(output_dir)
                    ): i for i, arch_info in enumerate(architectures)
                }
                
                # Collect results in order
                all_actions = [None] * len(architectures)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        action_result = future.result()
                        all_actions[idx] = action_result
                        logging.info(f"  ✓ Completed actions for architecture {idx+1}/{len(architectures)}")
                    except Exception as e:
                        logging.error(f"Failed to choose actions for architecture {idx}: {e}")
                        raise

            all_actions = [entry for entry in all_actions if isinstance(entry, dict)]
            all_actions = merge_actions_for_architectures(
                existing_actions,
                all_actions,
                architectures,
            )

            save_json(all_actions, actions_path)
    stage_timer.end("action_selection")

    architectures_with_action_hints = apply_action_hints_to_architectures(architectures, all_actions)
    architecture_component_count_before = count_architecture_components(architectures_with_action_hints)
    action_refinement_report_path = output_dir / "action_refinement_report.json"
    action_refinement_rounds = max(1, int(getattr(args, "action_refinement_rounds", 1) or 1))
    action_refinement_input_fingerprint = _stable_fingerprint(
        {
            "architectures_with_action_hints": architectures_with_action_hints,
            "action_refinement_semantics": "pre_action_component_merge_v1",
            "component_merge_agent_enabled": component_merge_agent is not None,
            "cross_requirement_component_merge_enabled": cross_requirement_component_merge_enabled,
            "component_split_agent_enabled": component_split_agent is not None,
            "action_refinement_rounds": action_refinement_rounds,
            "action_refinement_stop_on_stable": bool(args.action_refinement_stop_on_stable),
            "action_refinement_save_stops_component": bool(args.action_refinement_save_stops_component),
            "component_merge_admission_mode": str(args.component_merge_admission_mode),
            "component_merge_relaxed_best": float(args.component_merge_relaxed_best),
            "component_merge_relaxed_avg": float(args.component_merge_relaxed_avg),
            "component_merge_relaxed_min_pair": float(args.component_merge_relaxed_min_pair),
            "component_merge_relaxed_dominance_gap": float(args.component_merge_relaxed_dominance_gap),
        }
    )
    existing_action_refinement_report = load_json_if_exists(action_refinement_report_path, force_regen)
    merge_groups_added = 0
    split_groups_added = 0
    if component_split_agent is not None:
        if action_refinement_rounds > 1:
            if _artifact_matches_fingerprint(
                existing_action_refinement_report,
                action_refinement_input_fingerprint,
            ):
                logging.info(
                    "Reusing existing multi-round action feedback report: rounds=%d",
                    action_refinement_rounds,
                )
                action_refinement_report = existing_action_refinement_report or {}
            else:
                logging.info(
                    "Starting multi-round action feedback: rounds=%d stop_on_stable=%s save_stops_component=%s",
                    action_refinement_rounds,
                    bool(args.action_refinement_stop_on_stable),
                    bool(args.action_refinement_save_stops_component),
                )
                architectures, all_actions, action_refinement_report = run_action_feedback_rounds(
                    architectures=architectures,
                    initial_actions=all_actions,
                    component_merge_agent=component_merge_agent,
                    component_split_agent=component_split_agent,
                    rounds=action_refinement_rounds,
                    api_config=api_config,
                    output_dir=output_dir,
                    max_workers=args.max_workers,
                    stop_on_stable=bool(args.action_refinement_stop_on_stable),
                    save_stops_component=bool(args.action_refinement_save_stops_component),
                    enable_cross_requirement_merge=cross_requirement_component_merge_enabled,
                    save_round_artifacts=True,
                    existing_generated_entries=existing_generated_for_memory,
                    tdd_revise_failure_threshold=int(getattr(args, "tdd_revise_failure_threshold", 3) or 3),
                    decomposed_dag=decomposed_dag,
                    enable_metric_actions=bool(getattr(args, "enable_component_metric_actions", False)),
                    metric_split_cohesion_threshold=float(args.component_metric_split_cohesion_threshold),
                    metric_split_min_subrequirements=int(args.component_metric_split_min_subrequirements),
                    metric_merge_max_small_subrequirements=int(args.component_metric_merge_max_small_subrequirements),
                    metric_merge_judge=metric_merge_judge,
                )
                action_refinement_report = _attach_input_fingerprint(
                    action_refinement_report,
                    action_refinement_input_fingerprint,
                )
                save_json(all_actions, actions_path)
                save_json(action_refinement_report, action_refinement_report_path)
            architecture_component_count_after = int(
                action_refinement_report.get("stats", {}).get("components_after", 0) or 0
            )
            merge_groups_added = int(
                action_refinement_report.get("stats", {}).get("merge_group_count", 0) or 0
            )
            split_groups_added = int(
                action_refinement_report.get("stats", {}).get("split_group_count", 0) or 0
            )
        elif _artifact_matches_fingerprint(existing_action_refinement_report, action_refinement_input_fingerprint):
            logging.info(
                "Reusing existing action-guided structure refinement (fingerprint=%s)",
                action_refinement_input_fingerprint[:12],
            )
            architecture_component_count_after = count_architecture_components(architectures)
            merge_groups_added = int(
                (existing_action_refinement_report or {}).get("stats", {}).get("merge_group_count", 0) or 0
            )
            split_groups_added = int(
                (existing_action_refinement_report or {}).get("stats", {}).get("split_group_count", 0) or 0
            )
        elif _is_action_refinement_report_reusable(existing_action_refinement_report):
            logging.info(
                "Reusing legacy action-guided structure refinement and backfilling fingerprint (%s)",
                action_refinement_input_fingerprint[:12],
            )
            existing_action_refinement_report = _attach_input_fingerprint(
                existing_action_refinement_report,
                action_refinement_input_fingerprint,
            )
            save_json(existing_action_refinement_report, action_refinement_report_path)
            architecture_component_count_after = count_architecture_components(architectures)
            merge_groups_added = int(
                (existing_action_refinement_report or {}).get("stats", {}).get("merge_group_count", 0) or 0
            )
            split_groups_added = int(
                (existing_action_refinement_report or {}).get("stats", {}).get("split_group_count", 0) or 0
            )
        else:
            logging.info(
                "Starting action-guided structure refinement for %d architecture parents",
                len(architectures_with_action_hints),
            )
            architectures, refinement_report = apply_action_guided_structure_refinement(
                architectures=architectures_with_action_hints,
                component_merge_agent=component_merge_agent,
                component_split_agent=component_split_agent,
            )
            architecture_component_count_after = int(
                refinement_report.get("stats", {}).get("components_after", 0) or 0
            )
            merge_groups_added = int(
                refinement_report.get("stats", {}).get("merge_group_count", 0) or 0
            )
            split_groups_added = int(
                refinement_report.get("stats", {}).get("split_group_count", 0) or 0
            )
            action_refinement_report = _attach_input_fingerprint(
                {
                    "stats": {
                        "components_before": architecture_component_count_before,
                        "components_after": architecture_component_count_after,
                        "merge_group_count": merge_groups_added,
                        "split_group_count": split_groups_added,
                    }
                },
                action_refinement_input_fingerprint,
            )
            save_json(action_refinement_report, action_refinement_report_path)
        logging.info(
            "Action-guided structure refinement completed: components_before=%d components_after=%d merge_groups=%d split_groups=%d",
            architecture_component_count_before,
            architecture_component_count_after,
            merge_groups_added,
            split_groups_added,
        )
    else:
        architectures = architectures_with_action_hints

    save_json(architectures, architectures_path)
    _rebuild_flattened_architectures(output_dir, architectures)
    if bool(getattr(args, "stop_after_architecture_refinement", False)):
        logging.info(
            "Stopping after architecture refinement as requested; skipping module planning, assignment, and code generation."
        )
        stage_timer.finish()
        save_json(stage_timer.to_dict(), stage_timing_report_path)
        stage_timer.log_summary()
        logging.info(f"Architecture artifacts saved to {output_dir}")
        return
    logging.info(
        "Starting module planning for %d architecture parents after action-guided refinement",
        len(architectures),
    )
    with stage_timer.stage("module_planning"):
        module_plan = _apply_module_plan_to_layout_policy(
            output_dir=output_dir,
            architectures=architectures,
            actions=all_actions,
            layout_policy=layout_policy,
            module_planning_agent=module_planning_agent,
        )
    logging.info(
        "Module planning completed: module_families=%d",
        len(module_plan.get("module_families", []) if isinstance(module_plan, dict) else []),
    )
    logging.info("Starting module assignment using planned module families")
    with stage_timer.stage("module_assignment"):
        module_assignment = _apply_module_assignment_to_layout_policy(
            output_dir=output_dir,
            architectures=architectures,
            actions=all_actions,
            layout_policy=layout_policy,
            module_plan=module_plan,
            module_assignment_agent=module_assignment_agent,
        )
    logging.info(
        "Module assignment completed: assigned_paths=%d",
        len(module_assignment.get("component_package_path_index", {}) if isinstance(module_assignment, dict) else {}),
    )
    logging.info("Refreshing refined layout plan after module assignment")
    with stage_timer.stage("layout_plan_refined"):
        layout_grouping_report, package_api_plan = _refresh_layout_plan_artifacts(
            output_dir=output_dir,
            architectures=architectures,
            layout_policy=layout_policy,
        )
        component_plan_index = package_api_plan.get("component_index", {}) if isinstance(package_api_plan, dict) else {}
        logging.info(
            "Refreshed package planning after action-guided refinement: packages=%d components=%d module_families=%d assigned_paths=%d",
            package_api_plan.get("package_count", 0) if isinstance(package_api_plan, dict) else 0,
            package_api_plan.get("component_count", 0) if isinstance(package_api_plan, dict) else 0,
            len(module_plan.get("module_families", []) if isinstance(module_plan, dict) else []),
            len(module_assignment.get("component_package_path_index", {}) if isinstance(module_assignment, dict) else {}),
        )

    # Generate code for components
    from agents import CodeGeneratorAgent
    code_generator = CodeGeneratorAgent(api_config=api_config, output_dir=str(output_dir))
    
    # Check if code was already generated
    generated_files_path = output_dir / "generated_files.json"
    existing_generated_raw = load_json_if_exists(generated_files_path, should_regen)
    existing_generated = existing_generated_raw
    if existing_generated:
        existing_count = len(existing_generated) if isinstance(existing_generated, list) else 0
        existing_generated = filter_generated_files_for_active_parents(
            existing_generated,
            active_parent_names,
        )
        dropped = existing_count - len(existing_generated)
        if dropped > 0:
            logging.info(
                "Dropped %d stale generated-file records that no longer map to DAG parents",
                dropped,
            )
    existing_generated_index: Dict[str, Dict[str, Any]] = {}
    if isinstance(existing_generated, list):
        for entry in existing_generated:
            if not isinstance(entry, dict):
                continue
            parent = _parent_from_generated_entry(entry)
            component = str(entry.get("component") or "").strip()
            if not parent or not component:
                continue
            existing_generated_index[f"{parent}::{component}"] = entry
        _persist_all_component_reports(output_dir, existing_generated)
    code_output_dir = output_dir / "generated_code"
    code_output_dir.mkdir(exist_ok=True)
    module_registry: Dict[str, str] = {}
    retry_empty_component_keys: set[str] = set()
    retry_empty_preserved_entries: List[dict] = []
    if args.retry_empty_generated_components:
        if not isinstance(existing_generated, list) or not existing_generated:
            logging.warning(
                "--retry-empty-generated-components requested, but no existing generated_files.json entries were found."
            )
        else:
            for entry in existing_generated:
                if not isinstance(entry, dict):
                    continue
                key = _generated_entry_component_key(entry)
                if not key:
                    continue
                if _generated_entry_has_empty_files(entry):
                    retry_empty_component_keys.add(key)
                else:
                    retry_empty_preserved_entries.append(entry)
            if retry_empty_component_keys:
                filtered_architectures = filter_architectures_for_component_keys(
                    architectures,
                    retry_empty_component_keys,
                )
                logging.info(
                    "Retry-empty-components mode: retrying %d component(s) across %d parent(s); preserving %d existing entry/entries.",
                    len(retry_empty_component_keys),
                    len(filtered_architectures),
                    len(retry_empty_preserved_entries),
                )
                architectures = filtered_architectures
                existing_generated = []
            else:
                logging.info(
                    "Retry-empty-components mode: no entries with empty files found; nothing to regenerate."
                )
    logging.info(
        "Starting code generation stage: existing_generated=%d force_regen=%s incremental_mode=%s",
        len(existing_generated) if isinstance(existing_generated, list) else 0,
        force_regen,
        incremental_mode,
    )
    stage_timer.begin("code_generation")
    codegen_stage_started_at = time.perf_counter()
    resume_existing_generated: List[dict] = []
    resume_completed_parents: set[str] = set()
    if retry_empty_component_keys:
        resume_existing_generated = list(retry_empty_preserved_entries)
    all_architecture_parents = {
        _parent_from_architecture_entry(arch)
        for arch in architectures
        if _parent_from_architecture_entry(arch)
    }
    if existing_generated and code_output_dir.exists() and not force_regen and not incremental_mode:
        resume_completed_parents = detect_completed_generated_parents(
            existing_generated,
            architectures,
            rerun_retained_tdd_failures=bool(args.resume_rerun_retained_tdd_failures),
        )
        if resume_completed_parents and len(resume_completed_parents) < len(all_architecture_parents):
            resume_existing_generated = select_generated_entries_for_parents(
                existing_generated,
                resume_completed_parents,
            )
            logging.info(
                "Codegen resume detected %d/%d completed parents; reusing %d generated components and continuing remaining parents.",
                len(resume_completed_parents),
                len(all_architecture_parents),
                len(resume_existing_generated),
            )
            existing_generated = []

    if existing_generated and code_output_dir.exists() and not force_regen and not incremental_mode:
        # Verify that files actually exist
        all_files_exist = all(
            _generated_entry_has_usable_code(
                component_info,
                rerun_retained_tdd_failures=bool(args.resume_rerun_retained_tdd_failures),
            )
            for component_info in existing_generated
        )
        
        if all_files_exist:
            logging.info(f"Loading existing generated code from {code_output_dir}")
            created_files = existing_generated
            
            # Load existing component implementations into memory
            logging.info("Loading component implementations from existing generated code")
            for component_info in existing_generated:
                task_name = _parent_from_generated_entry(component_info) or "Unknown"
                component_name = component_info.get("component", "Unknown")
                files = component_info.get("files", {})
                
                if "code" in files:
                    code_file = Path(files["code"])
                    if code_file.exists():
                        try:
                            code_content = code_file.read_text(encoding='utf-8')
                            mock_code_result = {
                                "component_name": component_name,
                                "file_path": str(code_file.relative_to(code_output_dir)),
                                "code": code_content
                            }
                            metadata = code_generator.extract_component_metadata(
                                mock_code_result,
                                requirement_node=task_name
                            )
                            memory_agent.register_component_implementation(**metadata)
                            module_path = _module_from_relative_py_path(
                                str(code_file.relative_to(code_output_dir))
                            )
                            if module_path:
                                module_registry[component_name] = module_path
                            logging.info(f"  Registered: {component_name} for task {task_name}")
                        except Exception as e:
                            logging.warning(f"  Failed to register {component_name}: {e}")
        else:
            logging.info("Some generated files are missing, regenerating code")
            created_files = []
            for i, task in enumerate(plan):
                logging.info(f"Generating code for task {i+1}/{len(plan)}: {task.get('name', 'Unknown')}")
                
                # Get context of already implemented prerequisite components
                dependency_nodes = collect_dependency_nodes(decomposed_dag, task.get("name", ""))
                implemented_context = memory_agent.format_implementations_for_prompt(
                    filter_nodes=dependency_nodes
                )
                
                arch_info = architectures[i]
                retry_feedback_by_component: Dict[str, str] = {}
                arch_components = arch_info.get("architecture", {}).get("components", [])
                if isinstance(arch_components, list):
                    for cand in arch_components:
                        if not isinstance(cand, dict):
                            continue
                        cand_name = str(cand.get("name", "")).strip()
                        if not cand_name:
                            continue
                        prev_key = f"{task.get('name', 'Unknown')}::{cand_name}"
                        prev_entry = existing_generated_index.get(prev_key, {})
                        if isinstance(prev_entry, dict) and not _generated_entry_has_usable_code(
                            prev_entry,
                            rerun_retained_tdd_failures=bool(args.resume_rerun_retained_tdd_failures),
                        ):
                            feedback = str(
                                _evaluate_generated_entry_status(prev_entry).get("compressed_feedback") or ""
                            ).strip()
                            if feedback:
                                retry_feedback_by_component[cand_name] = feedback
                code_results = code_generator.generate_batch(
                    arch_info["architecture"],
                    task,
                    language="python",
                    implemented_components_context=implemented_context,
                    max_components=999,
                    retry_feedback_by_component=retry_feedback_by_component,
                )
                
                for code_result in code_results:
                    component_name = str(code_result.get("component_name", "")).strip()
                    parent_name = _parent_from_architecture_entry(arch_info) or str(task.get("name", "")).strip()
                    plan_key = f"{parent_name}::{component_name}"
                    planned_entry = component_plan_index.get(plan_key, {}) if isinstance(component_plan_index, dict) else {}
                    planned_rel_path = str(planned_entry.get("planned_file_path", "")).strip() if isinstance(planned_entry, dict) else ""
                    component_spec: Dict[str, Any] = {
                        "name": component_name,
                        "responsibilities": code_result.get("responsibilities", []),
                        "serves_subrequirements": code_result.get("serves_subrequirements", []),
                    }
                    arch_components = arch_info.get("architecture", {}).get("components", [])
                    if isinstance(arch_components, list):
                        for cand in arch_components:
                            if not isinstance(cand, dict):
                                continue
                            if str(cand.get("name", "")).strip() == component_name:
                                component_spec = cand
                                break
                    code_result, layout_meta = _enforce_layout_with_oov_retry(
                        code_generator=code_generator,
                        code_result=code_result,
                        component=component_spec,
                        unified_task=task if isinstance(task, dict) else {"name": parent_name},
                        architecture=arch_info.get("architecture", {}) if isinstance(arch_info, dict) else {},
                        implemented_context=implemented_context,
                        layout_policy=layout_policy,
                        planned_rel_path=str(planned_rel_path or ""),
                    )
                    planned_export_symbols = planned_entry.get("export_symbols", []) if isinstance(planned_entry, dict) else []
                    if not isinstance(planned_export_symbols, list) or not planned_export_symbols:
                        planned_export_symbols = _derive_component_export_symbols(
                            component_name=component_name,
                            responsibilities=code_result.get("responsibilities", []),
                            planned_file_path=str(code_result.get("file_path", "")),
                        )
                    files = code_generator.save_generated_code(code_result, str(code_output_dir))
                    import_postcheck = code_generator.postcheck_saved_component(
                        code_result=code_result,
                        repo_root=code_output_dir,
                        created_files=files,
                        implemented_components_context=implemented_context,
                    )
                    init_files: List[str] = []
                    code_file_path = files.get("code")
                    if code_file_path:
                        init_files = _ensure_package_inits(
                            Path(code_file_path),
                            Path(code_output_dir),
                            str(layout_policy.get("layout_root") or ""),
                        )
                        module_path = _module_from_relative_py_path(code_result.get("file_path", ""))
                        if module_path:
                            module_registry[code_result["component_name"]] = module_path
                    created_files.append({
                        "component": component_name,
                        "task": task.get("name", "Unknown"),
                        "component_responsibilities": code_result.get("responsibilities", []),
                        "component_export_symbols": planned_export_symbols,
                        "files": files,
                        "planned_file_path": code_result.get("file_path", ""),
                        "module_path": module_registry.get(code_result["component_name"], ""),
                        "init_files": init_files,
                        "layout_enforcement": layout_meta,
                        "import_postcheck": import_postcheck,
                        "syntax_postcheck": code_result.get("syntax_postcheck", {}),
                        "compile_postcheck": code_result.get("compile_postcheck", {}),
                        "generation_status": code_result.get("generation_status", ""),
                        "tdd_passed": code_result.get("tdd_passed"),
                        "tdd_final_pytest_rc": code_result.get("tdd_final_pytest_rc"),
                    })
                    
                    # Register component implementation in memory
                    try:
                        metadata = code_generator.extract_component_metadata(
                            code_result,
                            requirement_node=task.get("name", f"task_{i}")
                        )
                        memory_agent.register_component_implementation(**metadata)
                        logging.info(f"  Registered component: {code_result['component_name']}")
                    except Exception as e:
                        logging.warning(f"  Failed to register component {code_result['component_name']}: {e}")
            
            save_json(created_files, generated_files_path)
            _persist_all_component_reports(output_dir, created_files)
    else:
        logging.info(f"Generating code for parent-level architectures (sequential to maintain memory consistency)...")
        
        # Process code generation SEQUENTIALLY
        # Each parent's generated code is registered in memory before the next parent starts
        # This ensures no conflicts and proper dependency awareness
        created_files: list[dict] = []
        target_architectures = architectures
        if resume_existing_generated:
            logging.info("Loading completed parent implementations into memory for resume")
            rehydrate_memory_from_generated_artifacts(
                memory_agent=memory_agent,
                code_generator=code_generator,
                code_output_dir=Path(code_output_dir),
                generated_entries=resume_existing_generated,
                module_registry=module_registry,
                rerun_retained_tdd_failures=bool(args.resume_rerun_retained_tdd_failures),
            )

            created_files = list(resume_existing_generated)
            target_architectures = [
                arch for arch in architectures
                if _parent_from_architecture_entry(arch) not in resume_completed_parents
            ]
            logging.info(
                "Resume mode: skipped %d completed parents, remaining parents to generate: %d",
                len(resume_completed_parents),
                len(target_architectures),
            )
        if incremental_mode and existing_generated:
            # Load existing component implementations into memory for context
            logging.info("Loading existing component implementations for incremental generation")
            mapped_source_parents = {
                parent
                for parents in evolution_source_parents_by_target.values()
                for parent in parents
            }
            context_parent_scope = set(active_parent_names) | mapped_source_parents
            context_generated_entries = select_generated_entries_for_parents(
                existing_generated_raw,
                context_parent_scope,
            )
            rehydrate_memory_from_generated_artifacts(
                memory_agent=memory_agent,
                code_generator=code_generator,
                code_output_dir=Path(code_output_dir),
                generated_entries=context_generated_entries,
                module_registry=module_registry,
                rerun_retained_tdd_failures=bool(args.resume_rerun_retained_tdd_failures),
            )
            logging.info(
                "Loaded %d generated components as context (active=%d, mapped_sources=%d)",
                len(context_generated_entries),
                len(active_parent_names),
                len(mapped_source_parents),
            )

            created_files = [
                entry for entry in existing_generated
                if _parent_from_generated_entry(entry) not in evolution_parents
            ]
            target_architectures = [
                arch for arch in architectures
                if _parent_from_architecture_entry(arch) in evolution_parents
            ]

        architecture_index_by_parent = {
            _parent_from_architecture_entry(arch): idx
            for idx, arch in enumerate(architectures)
            if _parent_from_architecture_entry(arch)
        }
        prepared_architectures: List[Tuple[int, dict]] = []
        for arch_result in target_architectures:
            parent_task = arch_result.get("parent_task", "Unknown")
            original_index = architecture_index_by_parent.get(parent_task, len(prepared_architectures))
            parent_node = arch_result.get("parent_node")
            if not parent_node:
                parent_node = arch_result.get("architecture", {}).get("requirement")
                if not parent_node:
                    parent_node = {"name": parent_task}
            parent_prev_node = arch_result.get("parent_prev_node")
            if parent_prev_node is None and original_index > 0:
                logging.warning(
                    "No previous parent node found for parent %s, using previous architecture's parent node",
                    parent_task,
                )
                prev_arch = architectures[original_index - 1]
                parent_prev_node = prev_arch.get("parent_node") or prev_arch.get("architecture", {}).get("requirement")
            arch_result_with_context = dict(arch_result)
            arch_result_with_context["parent_node"] = parent_node
            arch_result_with_context["parent_prev_node"] = parent_prev_node
            prepared_architectures.append((original_index, arch_result_with_context))

        ordered_layer_parents = [
            arch_result.get("parent_task", "Unknown")
            for _, arch_result in sorted(prepared_architectures, key=lambda item: item[0])
        ]
        ordered_layer_parents = build_codegen_parent_order(
            dag_source=effective_codegen_dag_source,
            ordered_parents=ordered_layer_parents,
            no_graph_seed=int(args.no_graph_seed),
        )
        codegen_layers = build_parent_codegen_layers(codegen_parent_dag, ordered_layer_parents)
        logging.info(
            "Parent-level codegen will run in %d DAG layer(s) for %d remaining parents (max_workers=%d, source=%s).",
            len(codegen_layers),
            len(ordered_layer_parents),
            args.max_workers,
            effective_codegen_dag_source,
        )

        prepared_by_parent = {
            arch_result.get("parent_task", "Unknown"): (idx, arch_result)
            for idx, arch_result in prepared_architectures
        }

        for layer_idx, layer_parents in enumerate(codegen_layers, start=1):
            if not layer_parents:
                continue
            logging.info(
                "Codegen DAG layer %d/%d: %d parent(s) with no internal dependencies: %s",
                layer_idx,
                len(codegen_layers),
                len(layer_parents),
                ", ".join(layer_parents),
            )

            layer_contexts: Dict[str, str] = {}
            module_registry_snapshot = dict(module_registry)
            for parent_task in layer_parents:
                dependency_nodes = collect_dependency_nodes(codegen_parent_dag, parent_task)
                mapped_sources = sorted(evolution_source_parents_by_target.get(parent_task, set()))
                context_nodes = list(dict.fromkeys(dependency_nodes + mapped_sources))
                layer_contexts[parent_task] = memory_agent.format_implementations_for_prompt(
                    filter_nodes=context_nodes,
                    status_filter="implemented",
                )

            if len(layer_parents) == 1 or args.max_workers <= 1:
                for parent_task in layer_parents:
                    arch_index, arch_result_with_context = prepared_by_parent[parent_task]
                    try:
                        _, files = process_code_generation_task(
                            (arch_index, arch_result_with_context),
                            api_config,
                            str(output_dir),
                            str(code_output_dir),
                            memory_agent,
                            codegen_parent_dag,
                            layout_policy,
                            module_registry,
                            component_plan_index,
                            existing_generated_index,
                            source_context_parents=sorted(
                                evolution_source_parents_by_target.get(parent_task, set())
                            ),
                            implemented_context_override=layer_contexts[parent_task],
                            register_in_memory=True,
                            postcheck_max_workers=args.postcheck_max_workers,
                            rerun_retained_tdd_failures=bool(args.resume_rerun_retained_tdd_failures),
                        )
                        if files:
                            created_files.extend(files)
                            save_json(created_files, generated_files_path)
                            _persist_all_component_reports(output_dir, created_files)
                        memory_agent.persist(output_dir)
                    except Exception as e:
                        logging.error(
                            "Failed to generate code for parent %s (layer %d): %s",
                            parent_task,
                            layer_idx,
                            e,
                        )
                        continue
                continue

            with ThreadPoolExecutor(max_workers=min(args.max_workers, len(layer_parents))) as executor:
                future_to_parent = {
                    executor.submit(
                        process_code_generation_task,
                        prepared_by_parent[parent_task],
                        api_config,
                        str(output_dir),
                        str(code_output_dir),
                        None,
                        codegen_parent_dag,
                        layout_policy,
                        module_registry_snapshot,
                        component_plan_index,
                        existing_generated_index,
                        sorted(evolution_source_parents_by_target.get(parent_task, set())),
                        layer_contexts[parent_task],
                        False,
                        args.postcheck_max_workers,
                        bool(args.resume_rerun_retained_tdd_failures),
                    ): parent_task
                    for parent_task in layer_parents
                }

                for future in as_completed(future_to_parent):
                    parent_task = future_to_parent[future]
                    try:
                        _, files = future.result()
                        if files:
                            created_files.extend(files)
                            save_json(created_files, generated_files_path)
                            _persist_all_component_reports(output_dir, created_files)
                            rehydrate_memory_from_generated_artifacts(
                                memory_agent=memory_agent,
                                code_generator=code_generator,
                                code_output_dir=Path(code_output_dir),
                                generated_entries=files,
                                module_registry=module_registry,
                                parent_scope={parent_task},
                                rerun_retained_tdd_failures=bool(args.resume_rerun_retained_tdd_failures),
                            )
                        memory_agent.persist(output_dir)
                        logging.info(
                            "Codegen layer %d/%d completed parent '%s' with %d generated components.",
                            layer_idx,
                            len(codegen_layers),
                            parent_task,
                            len(files or []),
                        )
                    except Exception as e:
                        logging.error(
                            "Failed to generate code for parent %s (layer %d): %s",
                            parent_task,
                            layer_idx,
                            e,
                        )
                        continue
        
        save_json(created_files, generated_files_path)
        _persist_all_component_reports(output_dir, created_files)
    
    created_files = filter_generated_files_for_active_parents(
        created_files,
        active_parent_names,
    )
    save_json(created_files, generated_files_path)
    _persist_component_realization_report(output_dir, created_files)
    tdd_revise_report = build_tdd_revise_action_report(
        created_files,
        failure_threshold=int(getattr(args, "tdd_revise_failure_threshold", 2) or 2),
    )
    save_json(tdd_revise_report, output_dir / "tdd_revise_action_report.json")
    _persist_component_import_postcheck_report(output_dir, created_files)
    logging.info(
        "Code generation stage finished in %.2fs with %d generated component records.",
        time.perf_counter() - codegen_stage_started_at,
        len(created_files),
    )
    stage_timer.end("code_generation", components=len(created_files))

    # Post-generation lint/fix pass on generated Python files.
    lint_report_path = output_dir / "lint_fix_report.json"
    try:
        with stage_timer.stage("lint_fix"):
            from agents import LintFixAgent

            lint_fix_agent = LintFixAgent(api_config=api_config, output_dir=str(output_dir))
            lint_report = lint_fix_agent.run_after_codegen(
                generated_root=code_output_dir,
                generated_entries=created_files,
            )
            save_json(lint_report, lint_report_path)
            _persist_component_lint_report(output_dir, created_files, lint_report)
            logging.info(
                "Lint-fix stage completed: checked=%d, issues=%d, static_fixed=%d, llm_fixed=%d, unresolved=%d",
                lint_report.get("checked_files", 0),
                lint_report.get("files_with_issues", 0),
                lint_report.get("fixed_by_static", 0),
                lint_report.get("fixed_by_llm", 0),
                lint_report.get("unresolved", 0),
            )
    except Exception as exc:
        logging.warning("Lint-fix stage failed and was skipped: %s", exc)

    # Post-generation package API init export stage (scope: package + subpackages).
    init_export_report_path = output_dir / "init_export_report.json"
    try:
        with stage_timer.stage("init_export"):
            init_export_report = _build_package_init_exports(
                generated_root=code_output_dir,
                generated_entries=created_files if isinstance(created_files, list) else [],
                layout_root=str(layout_policy.get("layout_root") or ""),
                api_config=api_config,
                package_api_plan=package_api_plan if isinstance(package_api_plan, dict) else {},
                lazy_imports=bool(args.init_export_lazy_imports),
            )
            save_json(init_export_report, init_export_report_path)
            validation = init_export_report.get("validation", {}) if isinstance(init_export_report, dict) else {}
            logging.info(
                "Init-export stage completed: packages=%d, updated=%d, compile_failures=%d, import_failures=%d",
                init_export_report.get("packages_total", 0) if isinstance(init_export_report, dict) else 0,
                init_export_report.get("packages_updated", 0) if isinstance(init_export_report, dict) else 0,
                validation.get("compile_failure_count", 0) if isinstance(validation, dict) else 0,
                validation.get("import_failure_count", 0) if isinstance(validation, dict) else 0,
            )
    except Exception as exc:
        logging.warning("Init-export stage failed and was skipped: %s", exc)

    # Post-generation setup.py dependency inference stage
    try:
        with stage_timer.stage("setup_py"):
            setup_py_path = args.setup_py_path or (code_output_dir / "setup.py")
            derived_pkg_name = args.setup_py_package_name or str(layout_policy.get("layout_root") or args.repo)
            derived_pkg_name = _to_snake_case(derived_pkg_name)

            setup_agent = SetupPyAgent(
                api_config=api_config if args.setup_py_use_llm else {},
                output_dir=str(output_dir),
            )
            setup_report = setup_agent.run(
                project_root=code_output_dir,
                setup_py_path=setup_py_path,
                package_name=derived_pkg_name,
                skip_llm=not args.setup_py_use_llm,
                enable_postcheck=bool(args.setup_py_postcheck),
            )
            setup_report_path = output_dir / "setup_py_report.json"
            save_json(setup_report, setup_report_path)
            logging.info(
                "SetupPy stage completed: setup_py=%s postcheck=%s llm=%s",
                setup_py_path,
                bool(args.setup_py_postcheck),
                bool(args.setup_py_use_llm),
            )
    except Exception as exc:
        logging.warning("SetupPy stage failed and was skipped: %s", exc)

    # Log statistics
    total_components = len(created_files)
    unique_parents = len(
        {
            parent
            for parent in (_parent_from_generated_entry(entry) for entry in created_files)
            if parent
        }
    )
    logging.info(f"Code generation completed. {total_components} components generated for {unique_parents} parent requirements.")

    actions_by_parent = {
        _parent_from_action_entry(action): action
        for action in all_actions
        if isinstance(action, dict) and _parent_from_action_entry(action)
    }
    for arch_info in architectures:
        parent = _parent_from_architecture_entry(arch_info)
        action_info = actions_by_parent.get(parent, {"actions": []})
        architecture = arch_info.get("architecture", {})
        memory_agent.register_actions(action_info.get("actions", []), architecture)
    pruned_components = prune_memory_components_for_active_parents(
        memory_agent,
        active_parent_names,
    )
    if pruned_components:
        logging.info(
            "Pruned %d stale component-memory entries after code generation",
            pruned_components,
        )
    memory_agent.persist(output_dir)
    stage_timer.finish()
    save_json(stage_timer.to_dict(), stage_timing_report_path)
    stage_timer.log_summary()
    logging.info(f"Agents completed. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
