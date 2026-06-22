"""Architect agent that decomposes high-level requirements into actionable sub-requirements."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

    def __init__(self, max_sub_requirements: int = 3, min_description_length: int = 100, api_config: Dict[str, Any] | None = None, output_dir: str = ".", max_workers: int = 4) -> None:
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
                
                # Connect sub-requirements sequentially within the same parent
                if sub_names:
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
        """Break down a single requirement into multiple sub-requirements using LLM."""
        if not self.llm_client:
            return self._fallback_decompose_requirement(requirement)
        
        prompt = f"""You are a software architect expert. Decompose the following high-level requirement into **at most {self.max_sub_requirements}** concrete, actionable sub-requirements.

Requirement Name: {requirement.name}
Description: {requirement.description}

CRITICAL RULES:
1. Aim for **2-3 sub-requirements maximum** - only split if truly necessary
2. Each sub-requirement should be **substantial and independently implementable**
3. Avoid over-decomposition - merge related tasks into single sub-requirements
4. Skip trivial sub-tasks (documentation, testing) - focus on core implementation only

For each sub-requirement, provide:
- name: A SHORT unique identifier (just the sub-name part, e.g., "core-logic", "api-interface")
- description: A clear, actionable description of what needs to be implemented
- order: Execution order (0-based, sequential)

Return ONLY a JSON array of sub-requirement objects. Each should represent a meaningful chunk of work.

Note: Keep names concise - they will be prefixed with the parent requirement name automatically.
"""
        
        try:
            response = self.llm_client.call_json([
                {"role": "system", "content": "You are an expert software architect. Always return valid JSON arrays."},
                {"role": "user", "content": prompt}
            ])
            
            if isinstance(response, list):
                sub_reqs_data = response
            elif isinstance(response, dict) and "sub_requirements" in response:
                sub_reqs_data = response["sub_requirements"]
            elif isinstance(response, dict) and "requirements" in response:
                sub_reqs_data = response["requirements"]
            else:
                sub_reqs_data = [response] if isinstance(response, dict) else []
            
            sub_requirements: list[SubRequirement] = []
            for i, sub_data in enumerate(sub_reqs_data[: self.max_sub_requirements]):
                # Handle both dict and non-dict responses
                if isinstance(sub_data, dict):
                    raw_name = str(sub_data.get("name", f"sub-{i}"))
                    # Clean up the name - remove parent prefix if LLM included it
                    if "::" in raw_name:
                        # If LLM included parent prefix, extract just the sub-name part
                        sub_name_part = raw_name.split("::")[-1]
                    else:
                        sub_name_part = raw_name
                    
                    # Construct full name with parent prefix
                    name = f"{requirement.name}::{sub_name_part}"
                    description = str(sub_data.get("description", "Implement sub-requirement"))
                    order = int(sub_data.get("order", i))
                    metadata = sub_data
                else:
                    # If sub_data is not a dict, create a minimal representation
                    name = f"{requirement.name}::sub-{i}"
                    description = str(sub_data) if sub_data else "Implement sub-requirement"
                    order = i
                    metadata = {"raw_data": sub_data}
                
                sub_requirements.append(
                    SubRequirement(
                        name=name,
                        description=description,
                        parent=requirement.name,
                        order=order,
                        metadata=metadata,
                    )
                )
            
            logging.info(f"Successfully decomposed requirement '{requirement.name}' into {len(sub_requirements)} sub-requirements")
            return sub_requirements if sub_requirements else self._fallback_decompose_requirement(requirement)
        
        except Exception as e:
            logging.warning(f"LLM decomposition failed for '{requirement.name}' ({e}), using fallback")
            return self._fallback_decompose_requirement(requirement)
    
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
