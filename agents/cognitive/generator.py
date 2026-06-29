"""Generation agent that drafts an architecture for the top requirement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

from agents.infra.llm_client import LLMClient


@dataclass
class Component:
    name: str
    responsibilities: List[str]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "responsibilities": self.responsibilities}


class GenerationAgent:
    """Synthesize a coarse architecture that addresses the highest priority requirement."""

    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.llm_client = LLMClient(self.api_config, output_dir, agent_name="generator") if self.api_config.get("api_key") else None

    def generate_parent_architecture(
        self,
        parent_requirement: Dict[str, Any],
        sub_requirements: List[Dict[str, Any]],
        environment_feedback: str,
        existing_modules: List[Dict[str, Any]] = None,
        dag_summary: Dict[str, Any] | None = None,
    ) -> Dict[str, object]:
        """
        Generate unified architecture for a parent requirement and all its sub-requirements.
        This approach prevents component explosion and conflicts between subnodes.
        
        Args:
            parent_requirement: The high-level parent requirement
            sub_requirements: List of sub-requirements decomposed from parent
            environment_feedback: Memory context about existing implementations
            existing_modules: List of already implemented modules from sibling parents
            dag_summary: Summary of the dependency DAG
        
        Returns:
            Architecture dict with components covering all sub-requirements
        """
        if not self.llm_client:
            return self._fallback_parent_architecture(parent_requirement, sub_requirements, environment_feedback)
        
        parent_name = parent_requirement.get('name', 'Unknown Parent')
        parent_desc = parent_requirement.get('description', '')
        
        # Format sub-requirements
        sub_reqs_text = "\n".join([
            f"  - {sub.get('name', 'Unnamed')}: {sub.get('description', '')}"
            for sub in sub_requirements
        ])
        
        dag_context = ""
        if dag_summary:
            dag_context = f"\n\nDAG Context: {dag_summary.get('node_count', 0)} nodes, {dag_summary.get('edge_count', 0)} edges"
        
        prompt = f"""You are a software architecture expert. Design a unified, cohesive architecture for the following parent requirement and ALL its sub-requirements.

PARENT REQUIREMENT: {parent_name}
Description: {parent_desc}

SUB-REQUIREMENTS TO IMPLEMENT:
{sub_reqs_text}

Environment Context:
{environment_feedback}{dag_context}

CRITICAL INSTRUCTIONS:
1. Prefer a compact set of components. Introduce additional components only when there is a clear architectural boundary.
2. Each component should serve multiple related sub-requirements. Avoid 1:1 mapping between sub-requirements and components.
3. Optimize for the smallest set of high-cohesion, clearly ownable modules, not the longest feature checklist.
4. Merge adjacent responsibilities when they share the same data model, lifecycle, owner, and change together in practice.
5. Do NOT create separate components for validation, orchestration, export, adapters, helpers, or metadata unless they have clear standalone reuse value.
6. Split components only when there is strong evidence of different interfaces, dependencies, runtime constraints, or independent evolution.
7. Consider reusable capabilities from the environment context, but do not let existing module structure override the current parent's domain boundaries.
8. If uncertain, choose fewer, broader, more cohesive components.
9. Preserve domain semantics from the parent requirement and sub-requirements. Component names and responsibilities must retain the concrete domain anchors implied by the requirement taxonomy.
10. Avoid overly generic component names such as ModelLibrary, CoreService, RuntimeManager, UtilityModule, or GenericAPI unless they include a concrete domain qualifier from the requirement space.
11. If a parent/sub-requirement exposes named feature families, keep those families visible through component names, responsibilities, or served sub-requirements so later planning can still recover the domain structure.
12. Do NOT create one component per feature family by default. Multiple nearby feature families should be grouped into the same component when they share data structures, runtime assumptions, or a stable owner.
13. Prefer a small number of domain-cohesive components over a checklist decomposition of every named feature.

Return ONLY a JSON object:
{{
  "components": [{{
    "name": "ComponentName",
    "responsibilities": ["responsibility 1", "responsibility 2"],
    "serves_subrequirements": ["sub-req-name-1", "sub-req-name-2"]
  }}],
  "component_count": <number>,
  "rationale": "Brief explanation of design decisions and how components work together"
}}"""
        
        try:
            logging.info(
                "Parent architecture LLM request starting for '%s': sub_requirements=%d",
                parent_name,
                len(sub_requirements),
            )
            response = self.llm_client.call_json([
                {"role": "system", "content": "You are an expert software architect specializing in cohesive, modular design. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ], temperature=0.0, operation_name="cognitive_generator.parent_architecture")
            
            if isinstance(response, list):
                logging.warning("LLM returned list for parent architecture, using fallback")
                return self._fallback_parent_architecture(parent_requirement, sub_requirements, environment_feedback)
            
            components_data = response.get("components", [])
            components = []
            for comp_data in components_data:
                components.append({
                    "name": str(comp_data.get("name", "UnnamedComponent")),
                    "responsibilities": comp_data.get("responsibilities", []),
                    "serves_subrequirements": comp_data.get("serves_subrequirements", [])
                })
            
            architecture = {
                "requirement": parent_requirement,
                "sub_requirements": sub_requirements,
                "environment": environment_feedback,
                "components": components,
                "component_count": len(components),
                "dag_summary": dag_summary or {},
                "rationale": response.get("rationale", "Unified architecture for parent and all sub-requirements"),
                "notes": f"Unified architecture covering {len(sub_requirements)} sub-requirements with {len(components)} components",
            }
            
            logging.info(f"Generated parent architecture with {len(components)} components for '{parent_name}' ({len(sub_requirements)} sub-reqs)")
            return architecture
        
        except Exception as e:
            logging.warning(f"Parent architecture generation failed ({e}), using fallback")
            return self._fallback_parent_architecture(parent_requirement, sub_requirements, environment_feedback)

    def generate_architecture(
        self,
        requirement: Union[str, Dict[str, Any]],
        environment_feedback: str,
        dag_summary: Dict[str, Any] | None = None,
    ) -> Dict[str, object]:
        """Generate architecture using LLM based on the DAG-aware plan entry."""
        requirement_text, requirement_payload = self._normalize_requirement(requirement)
        
        if not self.llm_client:
            return self._fallback_generate_architecture(requirement_text, requirement_payload, environment_feedback, dag_summary)
        
        dag_context = ""
        if dag_summary:
            dag_context = f"""\nDAG Context:
- Total nodes: {dag_summary.get('node_count', 0)}
- Total edges: {dag_summary.get('edge_count', 0)}
- Root nodes: {', '.join(dag_summary.get('roots', [])[:3])}
- Leaf nodes: {', '.join(dag_summary.get('leaves', [])[:3])}"""
        
        prompt = f"""You are a software architecture expert. Design a modular architecture for the following requirement.

Requirement: {requirement_payload.get('name', requirement_text)}
Description: {requirement_payload.get('description', requirement_text)}

Environment Context:
{environment_feedback}{dag_context}

Design a compact architecture. Introduce additional components only when there is a clear architectural boundary that justifies it.

General component granularity rules:
- Prefer fewer, broader, high-cohesion components over many thin wrappers.
- Do not create a separate component for validation, orchestration, export, adapter, helper, or metadata logic unless it has independent reuse value.
- Merge responsibilities that share the same data model, lifecycle, dependencies, and owner.
- Split only when responsibilities have clearly different interfaces, runtime constraints, or evolve independently.
- If uncertain, choose fewer components.
- Preserve the domain vocabulary of the requirement. Do not collapse all components into generic names like Core, Library, Service, or API without a concrete domain qualifier.
- Component names should keep enough semantic signal that downstream package planning can still recover the original domain families.

For each component, specify:
- name: Component name (PascalCase)
- responsibilities: Array of specific responsibilities; use as many as needed to describe the component clearly without turning the list into a feature checklist.

Return ONLY a JSON object with:
{{
  "components": [{{
    "name": "ComponentName",
    "responsibilities": ["responsibility 1", "responsibility 2"]
  }}],
  "rationale": "Brief explanation of the architecture design"
}}"""
        
        try:
            logging.info(
                "Architecture LLM request starting for '%s'",
                requirement_payload.get('name', 'Unknown'),
            )
            response = self.llm_client.call_json([
                {"role": "system", "content": "You are an expert software architect. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ], operation_name="cognitive_generator.architecture")
            
            # Handle case where LLM returns a list instead of dict
            if isinstance(response, list):
                logging.warning("LLM returned list instead of dict for architecture, using fallback")
                return self._fallback_generate_architecture(requirement_payload, environment_feedback, dag_summary)
            
            components_data = response.get("components", [])
            components = []
            for comp_data in components_data:
                components.append(Component(
                    name=str(comp_data.get("name", "UnnamedComponent")),
                    responsibilities=comp_data.get("responsibilities", ["Provide functionality"])
                ))
            
            architecture = {
                "requirement": requirement_payload,
                "environment": environment_feedback,
                "components": [component.to_dict() for component in components],
                "dag_summary": dag_summary or {},
                "rationale": response.get("rationale", "LLM-generated architecture"),
                "notes": "Architecture generated by LLM.",
            }
            
            logging.info(f"Successfully generated architecture with {len(components)} components for requirement '{requirement_payload.get('name', 'Unknown')}'")
            return architecture
        
        except Exception as e:
            logging.warning(f"LLM architecture generation failed ({e}), using fallback")
            return self._fallback_generate_architecture(requirement_text, requirement_payload, environment_feedback, dag_summary)
    
    def _fallback_generate_architecture(
        self,
        requirement_text: str,
        requirement_payload: Dict[str, Any],
        environment_feedback: str,
        dag_summary: Dict[str, Any] | None,
    ) -> Dict[str, object]:
        """Fallback to heuristic-based architecture when LLM is unavailable."""
        components = self._seed_components(requirement_text)
        architecture = {
            "requirement": requirement_payload,
            "environment": environment_feedback,
            "components": [component.to_dict() for component in components],
            "dag_summary": dag_summary or {},
            "notes": "Architecture is heuristic-based and meant as a starting point.",
        }
        return architecture

    def _normalize_requirement(self, requirement: Union[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        if isinstance(requirement, str):
            text = requirement.strip() or "unspecified requirement"
            payload: dict[str, Any] = {"name": text, "description": text}
            return text, payload
        payload = dict(requirement)
        name = str(payload.get("name", "")).strip()
        description = str(payload.get("description", "")).strip()
        text_parts = [part for part in [name, description] if part]
        text = " - ".join(text_parts) if text_parts else "unspecified requirement"
        payload.setdefault("name", name or "requirement")
        payload.setdefault("description", description or text)
        return text, payload

    def _fallback_parent_architecture(
        self,
        parent_requirement: Dict[str, Any],
        sub_requirements: List[Dict[str, Any]],
        environment_feedback: str,
    ) -> Dict[str, object]:
        """Fallback architecture generation for parent + sub-requirements."""
        # Generate 3-5 components based on sub-requirement count
        num_components = min(max(3, len(sub_requirements) // 2), 5)
        components = []
        
        for i in range(num_components):
            # Distribute sub-requirements across components
            served_subs = sub_requirements[i::num_components]
            component = {
                "name": f"{parent_requirement.get('name', 'Module').replace(' ', '')}Component{i+1}",
                "responsibilities": [
                    f"Implement {sub.get('name', 'functionality')}" 
                    for sub in served_subs[:3]  # Max 3 responsibilities per component
                ],
                "serves_subrequirements": [sub.get('name', f'sub-{i}') for sub in served_subs]
            }
            components.append(component)
        
        return {
            "requirement": parent_requirement,
            "sub_requirements": sub_requirements,
            "environment": environment_feedback,
            "components": components,
            "component_count": len(components),
            "notes": f"Fallback architecture: {len(components)} components for {len(sub_requirements)} sub-requirements",
        }
    
    def _seed_components(self, requirement: str) -> List[Component]:
        """Create a predictable set of components driven by requirement keywords."""
        lowered = requirement.lower()
        components: list[Component] = []
        if "memory" in lowered:
            components.append(
                Component(
                    name="MemorySynchronizer",
                    responsibilities=[
                        "Capture repository facts",
                        "Persist deltas after strategist actions",
                    ],
                )
            )
        if "plan" in lowered or "task" in lowered:
            components.append(
                Component(
                    name="TaskPlanner",
                    responsibilities=[
                        "Rank requirements",
                        "Emit structured prompts for the generator",
                    ],
                )
            )
        if "generate" in lowered or "architecture" in lowered:
            components.append(
                Component(
                    name="ArchitectureBuilder",
                    responsibilities=[
                        "Expand the first requirement into modules",
                        "Describe interactions for strategist analysis",
                    ],
                )
            )
        if not components:
            components.append(
                Component(
                    name="GenericModule",
                    responsibilities=["Provide baseline capability for unspecified requirement"],
                )
            )
        return components
