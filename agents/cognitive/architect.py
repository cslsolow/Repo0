"""Architect agent that decomposes high-level requirements into actionable sub-requirements."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, Iterable, List, Set

from agents.infra.llm_client import LLMClient
from agents.rqmts.dag import RequirementDAG, RequirementNode
from .generator import GenerationAgent


@dataclass
class SubRequirement:
    """A decomposed, finer-grained requirement derived from a parent requirement."""

    name: str
    description: str
    parent: str
    order: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_node(self) -> RequirementNode:
        """Convert to a RequirementNode for DAG construction."""
        combined_metadata = dict(self.metadata)
        combined_metadata["parent"] = self.parent
        combined_metadata["order"] = self.order
        return RequirementNode(name=self.name, description=self.description, metadata=combined_metadata)


class ArchitectAgent:
    """Decompose high-level requirements into granular sub-requirements and rebuild the DAG."""

    def __init__(self, max_sub_requirements: int = 12, min_description_length: int = 100, api_config: Dict[str, Any] | None = None, output_dir: str = ".", max_workers: int = 4) -> None:
        self.max_sub_requirements = max_sub_requirements
        self.min_description_length = min_description_length
        self.api_config = api_config or {}
        self.output_dir = output_dir
        self.llm_client = LLMClient(self.api_config, output_dir, agent_name="architect") if self.api_config.get("api_key") else None
        self.max_workers = max_workers

    def decompose_dag(self, original_dag: RequirementDAG) -> RequirementDAG:
        """Expand each node in the DAG into sub-requirements and construct a refined DAG."""
        if original_dag.is_empty():
            return original_dag
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_sub_nodes: dict[str, RequirementNode] = {}
        new_adjacency: dict[str, set[str]] = {}
        decomposed_count = 0
        
        # Collect all nodes to process
        nodes_to_decompose = list(original_dag.topological_order())
        
        # Decompose nodes in parallel
        if nodes_to_decompose:
            logging.info(f"Decomposing {len(nodes_to_decompose)} nodes in parallel (workers: {self.max_workers})...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all decomposition tasks
                future_to_node = {
                    executor.submit(self.decompose_requirement, node): node
                    for node in nodes_to_decompose
                }
                
                # Collect results as they complete
                node_to_subreqs = {}
                for future in as_completed(future_to_node):
                    parent_node = future_to_node[future]
                    try:
                        sub_requirements = future.result()
                        node_to_subreqs[parent_node.name] = sub_requirements
                        decomposed_count += 1
                    except Exception as e:
                        logging.error(f"Failed to decompose '{parent_node.name}': {e}")
                        # Fallback: keep original node
                        all_sub_nodes[parent_node.name] = parent_node
                        new_adjacency.setdefault(parent_node.name, set())
            
            # Now process results in topological order to maintain proper connections
            for parent_node in nodes_to_decompose:
                if parent_node.name not in node_to_subreqs:
                    continue
                
                sub_requirements = node_to_subreqs[parent_node.name]
                sub_names: list[str] = []
                
                # Create sub-requirement nodes
                for sub_req in sub_requirements:
                    sub_node = sub_req.to_node()
                    all_sub_nodes[sub_node.name] = sub_node
                    new_adjacency.setdefault(sub_node.name, set())
                    sub_names.append(sub_node.name)
                
                # Connect sub-requirements using labeled dependencies when available.
                if sub_names:
                    explicit_edges = self._add_labeled_sub_requirement_edges(
                        sub_requirements,
                        new_adjacency,
                    )
                    if explicit_edges == 0:
                        for i, current_sub_name in enumerate(sub_names):
                            if i > 0:
                                prev_sub_name = sub_names[i - 1]
                                new_adjacency[prev_sub_name].add(current_sub_name)
                else:
                    logging.warning(f"Failed to decompose '{parent_node.name}': {sub_names}")
                
                # Connect to children from original DAG
                parent_name = parent_node.name
                original_children = original_dag.adjacency.get(parent_name, set())
                if original_children:
                    # The last sub-requirement connects to children (or parent itself if no subs)
                    last_node_name = sub_names[-1] if sub_names else parent_name
                    
                    for original_child_name in original_children:
                        # Check if the child has been decomposed into sub-requirements
                        # Sub-requirements have metadata["parent"] == original_child_name
                        child_first_sub = None
                        for node_name, node in all_sub_nodes.items():
                            if node.metadata.get("parent") == original_child_name:
                                # Found a sub-requirement of the child
                                # Use the first one (order 0)
                                if node.metadata.get("order", 0) == 0:
                                    child_first_sub = node_name
                                    break
                        
                        # Connect to child's first sub-requirement, or to child itself if not decomposed
                        target_node = child_first_sub if child_first_sub else original_child_name
                        new_adjacency[last_node_name].add(target_node)
            else:
                logging.warning(f"No sub-requirements found for '{parent_node.name}'")
        
        logging.info(f"DAG decomposition summary: {decomposed_count} nodes decomposed")
        return RequirementDAG(all_sub_nodes, new_adjacency)

    def generate_parent_architecture(
        self,
        parent_requirement: Dict[str, Any],
        sub_requirements: List[Dict[str, Any]],
        environment_feedback: str,
        existing_modules: List[Dict[str, Any]] | None = None,
        dag_summary: Dict[str, Any] | None = None,
    ) -> Dict[str, object]:
        """Compatibility wrapper: architect module owns parent-level architecture synthesis."""
        generator = GenerationAgent(api_config=self.api_config, output_dir=self.output_dir)
        return generator.generate_parent_architecture(
            parent_requirement=parent_requirement,
            sub_requirements=sub_requirements,
            environment_feedback=environment_feedback,
            existing_modules=existing_modules,
            dag_summary=dag_summary,
        )

    def generate_architecture(
        self,
        requirement: Dict[str, Any] | str,
        environment_feedback: str,
        dag_summary: Dict[str, Any] | None = None,
    ) -> Dict[str, object]:
        """Compatibility wrapper: architect module also exposes direct requirement-level synthesis."""
        generator = GenerationAgent(api_config=self.api_config, output_dir=self.output_dir)
        return generator.generate_architecture(
            requirement=requirement,
            environment_feedback=environment_feedback,
            dag_summary=dag_summary,
        )

    def decompose_requirement(self, requirement: RequirementNode) -> List[SubRequirement]:
        """Break down a single requirement through reasoning-first DAG labeling."""
        if not self.llm_client:
            return self._fallback_decompose_requirement(requirement)

        try:
            analysis = self._generate_requirement_analysis(requirement)
            response = self.llm_client.call_json([
                {"role": "system", "content": "You are an expert software architect. Always return one valid JSON object."},
                {"role": "user", "content": self._build_sub_requirement_label_prompt(requirement, analysis)},
            ])
            sub_requirements = self._sub_requirements_from_label_response(
                requirement,
                analysis,
                response,
            )
            logging.info(f"Successfully decomposed requirement '{requirement.name}' into {len(sub_requirements)} sub-requirements")
            return sub_requirements if sub_requirements else self._fallback_decompose_requirement(requirement)
        
        except Exception as e:
            logging.warning(f"LLM decomposition failed for '{requirement.name}' ({e}), using fallback")
            return self._fallback_decompose_requirement(requirement)

    def _generate_requirement_analysis(self, requirement: RequirementNode) -> str:
        """Generate a self-contained analysis before labeling sub-requirements."""
        prompt = f"""You are a software architect expert. Analyze the high-level requirement below in as much detail as needed before decomposing it.

Requirement Name: {requirement.name}
Description: {requirement.description}

Your analysis should make the requirement self-contained for later decomposition. Cover:
- intended behavior and user-visible functionality
- important inputs, outputs, data contracts, and interface expectations
- constraints, invariants, and error cases
- internal responsibilities that may need separate implementation treatment
- ambiguous or underspecified scope that should be preserved for planning

Do not produce sub-requirements yet. Return only a JSON object:
{{
  "analysis": "detailed self-contained requirement reasoning"
}}
"""
        response = self.llm_client.call_json([
            {"role": "system", "content": "You are an expert software architect. Always return one valid JSON object."},
            {"role": "user", "content": prompt},
        ])
        if isinstance(response, dict):
            analysis = str(response.get("analysis") or response.get("thought") or response.get("reasoning") or "").strip()
            if analysis:
                return analysis
        return requirement.description.strip()

    def _build_sub_requirement_label_prompt(self, requirement: RequirementNode, analysis: str) -> str:
        return f"""You are tasked with labeling a self-contained requirement analysis into a sub-requirement DAG.

Parent Requirement: {requirement.name}
Original Description: {requirement.description}

Complete Requirement Analysis:
{analysis}

Instructions:
1. Break the analysis into concrete sub-requirements only where separable requirement-side functionality may need an independent behavioral contract during implementation.
2. Do not target a fixed number of sub-requirements. Use as many or as few as the requirement analysis justifies.
3. Each sub-requirement must describe requirement-side functionality, not packages, files, classes, or implementation components.
4. Avoid documentation-only and testing-only sub-requirements.
5. For each sub-requirement, list dependency indices for earlier sub-requirements that the current sub-requirement logically follows. A dependency from u to v means that v relies on the behavior, output, data contract, or interface assumption established by u.
6. Dependencies must point only to previous sub-requirements using zero-based indices. Use an empty list when the sub-requirement can be understood directly from the enriched requirement description.

Return only a JSON object:
{{
  "analysis": "brief explanation of how the analysis was segmented",
  "sub_requirements": [
    {{
      "name": "short-name-without-parent-prefix",
      "description": "clear requirement-side functionality",
      "depend": [0],
      "rationale": "why this sub-requirement is separated"
    }}
  ]
}}
"""

    def _sub_requirements_from_label_response(
        self,
        requirement: RequirementNode,
        analysis: str,
        response: Any,
    ) -> List[SubRequirement]:
        if not isinstance(response, dict):
            return []
        raw_subs = response.get("sub_requirements")
        if not isinstance(raw_subs, list):
            raw_subs = response.get("sub-requirements")
        if not isinstance(raw_subs, list):
            raw_subs = response.get("sub_questions")
        if not isinstance(raw_subs, list):
            raw_subs = response.get("sub-questions")
        if not isinstance(raw_subs, list):
            return []

        sub_requirements: list[SubRequirement] = []
        used_names: set[str] = set()
        for i, sub_data in enumerate(raw_subs[: self.max_sub_requirements]):
            if not isinstance(sub_data, dict):
                continue
            sub_name_part = self._normalize_sub_requirement_name(
                sub_data.get("name") or sub_data.get("description"),
                fallback=f"sub-{i}",
            )
            if sub_name_part in used_names:
                sub_name_part = f"{sub_name_part}-{i}"
            used_names.add(sub_name_part)
            description = str(sub_data.get("description") or "").strip()
            if not description:
                description = f"Implement sub-requirement {sub_name_part} for {requirement.name}"
            metadata = {
                "analysis": analysis,
                "label_analysis": response.get("analysis") or response.get("thought") or "",
                "depend": self._normalize_dependency_indices(sub_data.get("depend"), upper_bound=i),
                "rationale": str(sub_data.get("rationale") or sub_data.get("reason") or "").strip(),
            }
            sub_requirements.append(
                SubRequirement(
                    name=f"{requirement.name}::{sub_name_part}",
                    description=description,
                    parent=requirement.name,
                    order=i,
                    metadata=metadata,
                )
            )
        return sub_requirements

    def _normalize_dependency_indices(self, value: Any, upper_bound: int) -> List[int]:
        if not isinstance(value, list):
            return []
        deps: list[int] = []
        for item in value:
            try:
                dep = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= dep < upper_bound and dep not in deps:
                deps.append(dep)
        return deps

    def _add_labeled_sub_requirement_edges(
        self,
        sub_requirements: List[SubRequirement],
        adjacency: dict[str, set[str]],
    ) -> int:
        added = 0
        names = [sub.name for sub in sub_requirements]
        for target_index, sub_req in enumerate(sub_requirements):
            deps = sub_req.metadata.get("depend", [])
            if not isinstance(deps, list):
                continue
            for dep in deps:
                if not isinstance(dep, int) or dep < 0 or dep >= target_index:
                    continue
                source_name = names[dep]
                target_name = sub_req.name
                adjacency.setdefault(source_name, set()).add(target_name)
                added += 1
        return added

    def _normalize_sub_requirement_name(self, value: Any, fallback: str) -> str:
        """Return the local sub-requirement name without parent prefixes."""
        raw_name = str(value or fallback).strip() or fallback
        sub_name = raw_name.split("::")[-1].strip()
        sub_name = re.sub(r"\s+", "-", sub_name.lower())
        sub_name = re.sub(r"[^a-z0-9_.-]+", "", sub_name)
        return sub_name or fallback
    
    def _fallback_decompose_requirement(self, requirement: RequirementNode) -> List[SubRequirement]:
        """Fallback to heuristic-based decomposition when LLM is unavailable."""
        logging.warning("LLM decomposition unavailable, using minimal heuristic-based decomposition")
        description = requirement.description.strip()
        keywords = self._extract_keywords(description)
        sub_requirements: list[SubRequirement] = []
        
        # Simplified: Only create 1-2 sub-requirements focused on core implementation
        if "measure" in keywords or "calculate" in keywords or "compute" in keywords:
            sub_requirements.append(
                SubRequirement(
                    name=f"{requirement.name}::core-logic",
                    description=f"Implement core data structures and computation logic for {requirement.name}",
                    parent=requirement.name,
                    order=0,
                )
            )
        elif "support" in keywords or "handle" in keywords or "process" in keywords:
            sub_requirements.append(
                SubRequirement(
                    name=f"{requirement.name}::processing",
                    description=f"Implement input validation and transformation logic for {requirement.name}",
                    parent=requirement.name,
                    order=0,
                )
            )
        elif "provide" in keywords or "offer" in keywords or "enable" in keywords:
            sub_requirements.append(
                SubRequirement(
                    name=f"{requirement.name}::interface",
                    description=f"Define and implement public API for {requirement.name}",
                    parent=requirement.name,
                    order=0,
                )
            )
        else:
            # Default: single comprehensive implementation sub-requirement
            sub_requirements.append(
                SubRequirement(
                    name=f"{requirement.name}::implementation",
                    description=f"Implement core functionality for {requirement.name}",
                    parent=requirement.name,
                    order=0,
                )
            )
        
        return sub_requirements[: self.max_sub_requirements]

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract normalized keywords from requirement text for decomposition heuristics."""
        normalized = re.sub(r"[^a-z\s]+", "", text.lower())
        tokens = normalized.split()
        keywords = {
            "measure",
            "calculate",
            "compute",
            "support",
            "handle",
            "provide",
            "offer",
            "enable",
            "verify",
            "check",
            "compare",
            "maintain",
            "update",
            "aggregate",
            "process",
            "align",
            "include",
            "allow",
            "configure",
        }
        return keywords & set(tokens)
