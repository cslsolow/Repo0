"""Dependency graph, localization, and depth-action analysis."""

from .dependency_graph_agent import DependencyGraphAgent
from .depth_action_builder import CapabilitySignature, DepthActionBuilder
from .gap_addition import GapAdditionCandidate, GapAdditionDecision, apply_gap_addition_decision, propose_gap_candidate_for_parent, run_local_gap_cleanup

__all__ = [
    "CapabilitySignature",
    "DependencyGraphAgent",
    "DepthActionBuilder",
    "GapAdditionCandidate",
    "GapAdditionDecision",
    "apply_gap_addition_decision",
    "propose_gap_candidate_for_parent",
    "run_local_gap_cleanup",
]
