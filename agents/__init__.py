"""Minimal multi-agent scaffolding for Repo0."""

from __future__ import annotations

from ._compat_submodules import register_legacy_submodule_names

register_legacy_submodule_names()

from .cognitive.architect import ArchitectAgent
from .cognitive.generator import GenerationAgent
from .cognitive.memory import ComponentImplementation, MemoryAgent, MemorySnapshot
from .cognitive.module_assignment import ModuleAssignmentAgent
from .cognitive.module_planner import ModulePlanningAgent
from .cognitive.planner import PlannerAgent
from .cognitive.strategist import StrategistAgent
from .coding.code_generator import CodeGeneratorAgent
from .coding.fix_agent import FixAgent
from .coding.import_postcheck_fix_agent import ImportPostcheckFixAgent
from .coding.lint_fix_agent import LintFixAgent
from .coding.patch_agent import PatchAgent
from .coding.setup_py_agent import SetupPyAgent
from .coding.skeleton_review_agent import SkeletonReviewAgent
from .coding.static_preflight import run_static_preflight
from .coding.test_review_agent import TestReviewAgent
from .coding.test_rewrite_agent import TestRewriteAgent
from .merge.component_merge_agent import ComponentMergeAgent
from .merge.component_split_agent import ComponentSplitAgent
from .merge.requirement_merge_agent import RequirementMergeAgent
from .analysis.dependency_graph_agent import DependencyGraphAgent
from .infra.llm_client import LLMClient
from .rqmts.dag import RequirementDAG, RequirementNode
from .rqmts.dag_evolution import DAGEvolutionAgent, DAGOperation
from .rqmts.hierarchical_dag import HierarchicalDAG, HighLevelRequirement, TaskNode
from .rqmts.hierarchical_evolution import HierarchicalDAGEvolutionAgent

__all__ = [
    "ArchitectAgent",
    "MemoryAgent",
    "MemorySnapshot",
    "ComponentImplementation",
    "RequirementDAG",
    "RequirementNode",
    "DAGEvolutionAgent",
    "DAGOperation",
    "HierarchicalDAG",
    "HighLevelRequirement",
    "TaskNode",
    "HierarchicalDAGEvolutionAgent",
    "PlannerAgent",
    "GenerationAgent",
    "ModuleAssignmentAgent",
    "ModulePlanningAgent",
    "StrategistAgent",
    "LLMClient",
    "CodeGeneratorAgent",
    "FixAgent",
    "ImportPostcheckFixAgent",
    "LintFixAgent",
    "run_static_preflight",
    "DependencyGraphAgent",
    "PatchAgent",
    "SkeletonReviewAgent",
    "TestReviewAgent",
    "TestRewriteAgent",
    "RequirementMergeAgent",
    "ComponentMergeAgent",
    "ComponentSplitAgent",
    "SetupPyAgent",
]
