"""DAG Evolution Agent for managing incremental DAG construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal
import json
import logging

from agents.cognitive.memory import MemoryAgent
from agents.infra.llm_client import LLMClient

from .dag import RequirementDAG, RequirementNode


@dataclass
class DAGOperation:
    """Record of a DAG modification operation."""
    
    operation_type: Literal["add", "create", "split", "merge", "delete", "revise"]
    timestamp: str
    affected_nodes: List[str]
    details: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_type": self.operation_type,
            "timestamp": self.timestamp,
            "affected_nodes": self.affected_nodes,
            "details": self.details,
            "reason": self.reason,
        }


class DAGEvolutionAgent:
    """Manages incremental construction and evolution of requirement DAGs."""
    
    def __init__(
        self, 
        dag: RequirementDAG,
        memory_agent: MemoryAgent | None = None,
        api_config: Dict[str, Any] | None = None,
        output_dir: str = "."
    ) -> None:
        self.dag = dag
        self.memory_agent = memory_agent
        self.api_config = api_config or {}
        self.llm_client = LLMClient(self.api_config, output_dir, agent_name="dag_evolution") if self.api_config.get("api_key") else None
        self.operation_history: List[DAGOperation] = []
    
    def add_sub_requirement(
        self, 
        new_requirement: RequirementNode,
        parent_names: List[str] | None = None,
        context: str = "",
        action_override: Dict[str, Any] | None = None,
        decomposed_dag: RequirementDAG | None = None,
        architect: Any | None = None
    ) -> Dict[str, Any]:
        """
        Add a new requirement node to the DAG and determine necessary operations.
        
        This method analyzes the new requirement against the existing DAG and
        decides whether to:
        - Simply add it
        - Split an existing requirement
        - Merge with existing requirements
        - Delete obsolete requirements
        - Revise existing requirements
        """
        import datetime
        
        # Analyze impact of the new requirement
        action = action_override or self._analyze_requirement_impact(new_requirement, context)
        if not isinstance(action, dict):
            raise ValueError("Invalid action")
        
        try:
            operation_type = action.get("operation", "add")
            if operation_type == "create":
                operation_type = "add"
            relation_type = str(action.get("relation_type") or "").upper()
            if not parent_names:
                suggested = action.get("suggested_parents") or []
                if suggested:
                    parent_names = [p for p in suggested if p in self.dag.nodes]
            if operation_type == "add" and relation_type != "CHILD":
                parent_names = None
            
            if operation_type == "add":
                return self.add_requirement(
                    new_requirement,
                    parent_names,
                    action,
                    decomposed_dag=decomposed_dag,
                    architect=architect
                )
            
            elif operation_type == "split":
                timestamp = datetime.datetime.now().isoformat()
                result = {
                    "action": operation_type,
                    "affected_nodes": [],
                    "new_nodes": [],
                    "details": action,
                }
                # Split existing requirement
                target = action.get("target")
                if not target:
                    raise ValueError("Split operation requires 'target'")
                sub_reqs = self._create_split_requirements(target, new_requirement, action)
                created = self.dag.split_requirement(target, sub_reqs)
                result["new_nodes"] = created
                result["affected_nodes"] = [target]
                
                operation = DAGOperation(
                    operation_type="split",
                    timestamp=timestamp,
                    affected_nodes=[target] + created,
                    details={"original": target, "created": created},
                    reason=action.get("reason", "Requirement split for better granularity"),
                )
                if decomposed_dag and architect:
                    result["subnode_updates"] = self._update_decomposed_for_split(
                        decomposed_dag,
                        architect,
                        result["affected_nodes"] + result["new_nodes"]
                    )
            
            elif operation_type == "merge":
                timestamp = datetime.datetime.now().isoformat()
                result = {
                    "action": operation_type,
                    "affected_nodes": [],
                    "new_nodes": [],
                    "details": action,
                }
                # Merge with existing requirements
                targets = action.get("targets") or []
                if not targets:
                    raise ValueError("Merge operation requires 'targets'")
                merged_node = self._create_merged_requirement(targets, new_requirement, action)
                merged_name = self.dag.merge_requirements(targets, merged_node)
                result["new_nodes"] = [merged_name]
                result["affected_nodes"] = targets
                
                operation = DAGOperation(
                    operation_type="merge",
                    timestamp=timestamp,
                    affected_nodes=targets + [merged_name],
                    details={"merged_from": targets, "merged_to": merged_name},
                    reason=action.get("reason", "Requirements merged to reduce redundancy"),
                )
                if decomposed_dag and architect:
                    result["subnode_updates"] = self._update_decomposed_for_merge(
                        decomposed_dag,
                        architect,
                        result["affected_nodes"] + result["new_nodes"]
                    )
            
            elif operation_type == "delete":
                timestamp = datetime.datetime.now().isoformat()
                result = {
                    "action": operation_type,
                    "affected_nodes": [],
                    "new_nodes": [],
                    "details": action,
                }
                # Delete obsolete requirements
                targets = action.get("targets") or []
                if not targets:
                    raise ValueError("Delete operation requires 'targets'")
                for target in targets:
                    self.dag.delete_requirement(target, reconnect=action.get("reconnect", True))
                # Then add the new requirement
                self.dag.add_requirement(new_requirement, parent_names)
                result["new_nodes"] = [new_requirement.name]
                result["affected_nodes"] = targets
                
                operation = DAGOperation(
                    operation_type="delete",
                    timestamp=timestamp,
                    affected_nodes=targets + [new_requirement.name],
                    details={"deleted": targets, "added": new_requirement.name},
                    reason=action.get("reason", "Obsolete requirements removed"),
                )
                if decomposed_dag and architect:
                    result["subnode_updates"] = self._update_decomposed_for_delete(
                        decomposed_dag,
                        architect,
                        result["affected_nodes"] + result["new_nodes"]
                    )
            
            elif operation_type == "revise":
                timestamp = datetime.datetime.now().isoformat()
                result = {
                    "action": operation_type,
                    "affected_nodes": [],
                    "new_nodes": [],
                    "details": action,
                }
                # Revise existing requirement
                target = action.get("target")
                if not target:
                    raise ValueError("Revise operation requires 'target'")
                revised = self._create_revised_requirement(target, new_requirement, action)
                new_name = self.dag.revise_requirement(target, revised)
                result["new_nodes"] = [new_name]
                result["affected_nodes"] = [target]
                
                operation = DAGOperation(
                    operation_type="revise",
                    timestamp=timestamp,
                    affected_nodes=[target, new_name],
                    details={"original": target, "revised_to": new_name},
                    reason=action.get("reason", "Requirement revised for clarity"),
                )
                if decomposed_dag and architect:
                    result["subnode_updates"] = self._update_decomposed_for_revise(
                        decomposed_dag,
                        architect,
                        result["affected_nodes"] + result["new_nodes"]
                    )
            
            else:
                raise ValueError(f"Unknown operation: {action['operation']}")
            
            self.operation_history.append(operation)
            result["success"] = True
            result["operation_record"] = operation.to_dict()
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        return result

    def add_requirement(
        self,
        new_requirement: RequirementNode,
        parent_names: List[str] | None = None,
        action: Dict[str, Any] | None = None,
        decomposed_dag: RequirementDAG | None = None,
        architect: Any | None = None
    ) -> Dict[str, Any]:
        """Add a high-level requirement node to the DAG."""
        import datetime

        action = action or {"operation": "add"}
        operation_type = action.get("operation", "add")
        if operation_type == "create":
            operation_type = "add"
        timestamp = datetime.datetime.now().isoformat()

        result = {
            "action": operation_type,
            "affected_nodes": [],
            "new_nodes": [],
            "details": action,
        }

        try:
            inferred_edges = self._infer_edges_for_new_requirement(new_requirement)
            self.dag.add_requirement(new_requirement, parent_names)
            for parent, child in inferred_edges:
                if parent in self.dag.nodes and child in self.dag.nodes:
                    self.dag.add_dependency(parent, child)
            result["new_nodes"] = [new_requirement.name]
            result["affected_nodes"] = parent_names or []

            operation = DAGOperation(
                operation_type=action.get("operation", "add"),
                timestamp=timestamp,
                affected_nodes=[new_requirement.name],
                details={"parents": parent_names or []},
                reason=action.get("reason", "New requirement added"),
            )
            self.operation_history.append(operation)
            if decomposed_dag and architect:
                result["subnode_updates"] = self._update_decomposed_for_add(
                    decomposed_dag,
                    architect,
                    
                    result["affected_nodes"] + result["new_nodes"]
                )
            result["success"] = True
            result["operation_record"] = operation.to_dict()
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logging.error(f"Error processing task: {e}")

        return result

    def _infer_edges_for_new_requirement(
        self,
        new_requirement: RequirementNode
    ) -> List[tuple[str, str]]:
        """Infer edges for a new requirement using incremental edge generation."""
        if not self.llm_client:
            return []

        from agents.ingest import graph_parser

        try:
            requirements_payload = {
                "requirements": [
                    {"name": node.name, "description": node.description}
                    for node in self.dag.nodes.values()
                ]
            }
        except Exception as e:
            logging.error(f"Failed to recover requirements: {e}")
            return []

        new_requirement_content = {
            "name": new_requirement.name,
            "description": new_requirement.description
        }

        requirements_content = json.dumps(requirements_payload, ensure_ascii=False)
        existing_edges = {}
        for parent, children in self.dag.adjacency.items():
            existing_edges[parent] = sorted(children)

        edges_data = graph_parser.generate_incremental_edges(
            requirements_content=requirements_content,
            existing_edges=existing_edges,
            new_requirement=new_requirement_content,
            llm_client=self.llm_client,
        )

        if not edges_data:
            return []
        logging.debug("Generated edges: %s", edges_data)
        pairs: List[tuple[str, str]] = []
        for parent, children in edges_data.items():
            if isinstance(children, list):
                for child in children:
                    pairs.append((parent, child))
            else:
                pairs.append((parent, children))
        return pairs

    def evolve_requirement_with_subnodes(
        self,
        new_requirement: RequirementNode,
        parent_names: List[str] | None = None,
        context: str = "",
        decomposed_dag: RequirementDAG | None = None,
        architect: Any | None = None,
        action_override: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Evolve a requirement at the parent level, then update affected sub-nodes.
        
        This applies the evolution to the main DAG first, then refreshes the
        decomposed DAG for any affected parents.
        """
        operation = (action_override or {}).get("operation", "add")
        if operation == "create":
            operation = "add"

        logging.debug(f"Evolving requirement {new_requirement.name} with operation {operation}")
        if operation == "add":
            result = self.add_requirement(
                new_requirement,
                parent_names,
                action_override,
                decomposed_dag=decomposed_dag,
                architect=architect
            )
        else:
            result = self.add_sub_requirement(
                new_requirement,
                parent_names,
                context,
                action_override=action_override,
                decomposed_dag=decomposed_dag,
                architect=architect
            )
        if not result.get("success"):
            return result
        return result
    
    def _analyze_requirement_impact(
        self, 
        new_requirement: RequirementNode,
        context: str
    ) -> Dict[str, Any]:
        """Analyze how a new requirement should be integrated into the DAG."""
        
        if not self.llm_client:
            logging.warning("LLM client not available for requirement impact analysis, using fallback")
            return self._fallback_analyze_impact(new_requirement)
        
        # Build context
        dag_summary = self.dag.summary()
        existing_requirements = [
            {"name": node.name, "description": node.description}
            for node in self.dag.nodes.values()
        ]
        
        prompt = f"""You are analyzing how to integrate a new requirement into an existing requirement DAG.

New Requirement:
- Name: {new_requirement.name}
- Description: {new_requirement.description}

Existing DAG Summary:
- Total nodes: {dag_summary['node_count']}
- Roots: {', '.join(dag_summary['roots'][:5])}
- Leaves: {', '.join(dag_summary['leaves'][:5])}

Existing Requirements (sample):
{json.dumps(existing_requirements[:10], indent=2)}

Context: {context}

Analyze the impact and decide the best operation:

1. **create/add**: Simply add as a new node (use if it's genuinely new and doesn't overlap)
2. **split**: Split an existing requirement (use if the new req reveals that an existing req should be broken down)
3. **merge**: Merge with existing requirements (use if the new req overlaps significantly with existing ones)
4. **delete**: Delete obsolete requirements (use if the new req makes existing ones redundant)
5. **revise**: Revise an existing requirement (use if the new req is a better version of an existing one)

Return ONLY a JSON object:
{{
  "operation": "create|add|split|merge|delete|revise",
  "reason": "Brief explanation of why this operation is chosen",
  "confidence": 0.0-1.0,
  "target": "name of existing requirement (for split/revise)",
  "targets": ["list of requirement names (for merge/delete)"],
  "suggested_parents": ["parent requirement names for the new req"]
}}"""

        try:
            response = self.llm_client.call_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            # Validate response
            if not isinstance(response, dict) or "operation" not in response:
                return self._fallback_analyze_impact(new_requirement)
            
            return response
            
        except Exception:
            logging.warning(f"LLM requirement impact analysis failed, using fallback for '{new_requirement.name}'")
            return self._fallback_analyze_impact(new_requirement)
    
    def _fallback_analyze_impact(self, new_requirement: RequirementNode) -> Dict[str, Any]:
        """Fallback heuristic when LLM is unavailable."""
        # Simple heuristic: check for name similarity
        existing_names = set(self.dag.nodes.keys())
        
        # Check for exact match (revise case)
        if new_requirement.name in existing_names:
            return {
                "operation": "revise",
                "target": new_requirement.name,
                "reason": "Exact name match found",
                "confidence": 0.8,
            }
        
        # Check for partial match (merge case)
        for existing_name in existing_names:
            if (
                new_requirement.name.lower() in existing_name.lower() or
                existing_name.lower() in new_requirement.name.lower()
            ):
                return {
                    "operation": "merge",
                    "targets": [existing_name],
                    "reason": "Partial name match detected",
                    "confidence": 0.6,
                }
        
        # Default: simple add
        return {
            "operation": "add",
            "reason": "No conflicts detected",
            "confidence": 0.5,
            "suggested_parents": [],
        }
    
    def _create_split_requirements(
        self,
        target: str,
        new_requirement: RequirementNode,
        action: Dict[str, Any]
    ) -> List[RequirementNode]:
        """Create sub-requirements for a split operation."""
        original = self.dag.nodes[target]
        
        # Include both the original (refined) and the new requirement
        return [
            RequirementNode(
                name=f"{target}::core",
                description=original.description,
                metadata={"split_type": "core", **original.metadata}
            ),
            new_requirement,
        ]
    
    def _create_merged_requirement(
        self,
        targets: List[str],
        new_requirement: RequirementNode,
        action: Dict[str, Any]
    ) -> RequirementNode:
        """Create a merged requirement from multiple existing ones."""
        descriptions = [self.dag.nodes[t].description for t in targets if t in self.dag.nodes]
        descriptions.append(new_requirement.description)
        
        combined_description = " | ".join(descriptions)
        
        return RequirementNode(
            name=new_requirement.name,
            description=combined_description,
            metadata={"merge_source": targets}
        )
    
    def _create_revised_requirement(
        self,
        target: str,
        new_requirement: RequirementNode,
        action: Dict[str, Any]
    ) -> RequirementNode:
        """Create a revised version of an existing requirement."""
        # Use the new requirement but preserve some metadata
        original = self.dag.nodes[target]
        new_requirement.metadata.update({
            "original_metadata": original.metadata,
            "revise_reason": action.get("reason", "Updated specification"),
        })
        return new_requirement
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get the history of all DAG operations."""
        return [op.to_dict() for op in self.operation_history]
    
    def export_state(self) -> Dict[str, Any]:
        """Export current DAG state and operation history."""
        return {
            "dag": self.dag.to_dict(),
            "operations": self.get_operation_history(),
            "summary": self.dag.summary(),
        }

    def _update_decomposed_for_add(
        self,
        decomposed_dag: RequirementDAG,
        architect: Any,
        affected_parents: List[str]
    ) -> Dict[str, Any]:
        affected_parents = self._normalize_parent_names(affected_parents)
        removed_subnodes: List[str] = []
        added_subnodes: List[str] = []
        edge_updates: List[Dict[str, str]] = []

        for parent in affected_parents:
            if parent not in self.dag.nodes:
                continue
            existing_subnodes = [
                name for name in self._get_subnode_names(decomposed_dag, parent)
                if name in decomposed_dag.nodes
            ]
            if existing_subnodes:
                continue
            parent_node = self.dag.nodes[parent]
            sub_requirements = architect.decompose_requirement(parent_node)
            sub_nodes = [sub_req.to_node() for sub_req in sub_requirements]
            if not sub_nodes:
                sub_nodes = [
                    RequirementNode(
                        name=parent_node.name,
                        description=parent_node.description,
                        metadata=dict(parent_node.metadata)
                    )
                ]
            for node in sub_nodes:
                decomposed_dag.add_requirement(node)
                added_subnodes.append(node.name)
            for i in range(len(sub_nodes) - 1):
                decomposed_dag.add_dependency(sub_nodes[i].name, sub_nodes[i + 1].name)

        edge_updates.extend(self._reconnect_decomposed_edges(decomposed_dag, affected_parents))
        return {
            "removed_subnodes": removed_subnodes,
            "added_subnodes": added_subnodes,
            "edge_updates": edge_updates,
        }

    def _update_decomposed_for_revise(
        self,
        decomposed_dag: RequirementDAG,
        architect: Any,
        affected_parents: List[str]
    ) -> Dict[str, Any]:
        return self._refresh_decomposed_subnodes(decomposed_dag, architect, affected_parents)

    def _update_decomposed_for_child(
        self,
        decomposed_dag: RequirementDAG,
        architect: Any,
        affected_parents: List[str]
    ) -> Dict[str, Any]:
        return self._update_decomposed_for_add(decomposed_dag, architect, affected_parents)

    def _update_decomposed_for_merge(
        self,
        decomposed_dag: RequirementDAG,
        architect: Any,
        affected_parents: List[str]
    ) -> Dict[str, Any]:
        return self._refresh_decomposed_subnodes(decomposed_dag, architect, affected_parents)

    def _update_decomposed_for_split(
        self,
        decomposed_dag: RequirementDAG,
        architect: Any,
        affected_parents: List[str]
    ) -> Dict[str, Any]:
        return self._refresh_decomposed_subnodes(decomposed_dag, architect, affected_parents)

    def _update_decomposed_for_delete(
        self,
        decomposed_dag: RequirementDAG,
        architect: Any,
        affected_parents: List[str]
    ) -> Dict[str, Any]:
        return self._refresh_decomposed_subnodes(decomposed_dag, architect, affected_parents)

    def _refresh_decomposed_subnodes(
        self,
        decomposed_dag: RequirementDAG,
        architect: Any,
        affected_parents: List[str]
    ) -> Dict[str, Any]:
        affected_parents = self._normalize_parent_names(affected_parents)
        removed_subnodes: List[str] = []
        added_subnodes: List[str] = []

        for parent in affected_parents:
            for sub_name in self._get_subnode_names(decomposed_dag, parent):
                decomposed_dag.delete_requirement(sub_name, reconnect=False)
                removed_subnodes.append(sub_name)

        for parent in affected_parents:
            if parent not in self.dag.nodes:
                continue
            parent_node = self.dag.nodes[parent]
            sub_requirements = architect.decompose_requirement(parent_node)
            sub_nodes = [sub_req.to_node() for sub_req in sub_requirements]
            if not sub_nodes:
                sub_nodes = [
                    RequirementNode(
                        name=parent_node.name,
                        description=parent_node.description,
                        metadata=dict(parent_node.metadata)
                    )
                ]
            for node in sub_nodes:
                decomposed_dag.add_requirement(node)
                added_subnodes.append(node.name)
            for i in range(len(sub_nodes) - 1):
                decomposed_dag.add_dependency(sub_nodes[i].name, sub_nodes[i + 1].name)

        edge_updates = self._reconnect_decomposed_edges(decomposed_dag, affected_parents)
        return {
            "removed_subnodes": removed_subnodes,
            "added_subnodes": added_subnodes,
            "edge_updates": edge_updates,
        }

    def _reconnect_decomposed_edges(
        self,
        decomposed_dag: RequirementDAG,
        affected_parents: List[str]
    ) -> List[Dict[str, str]]:
        edge_updates: List[Dict[str, str]] = []
        affected_set = set(self._normalize_parent_names(affected_parents))
        for parent, children in self.dag.adjacency.items():
            if parent not in affected_set and not affected_set.intersection(children):
                continue
            source_last = self._get_last_subnode(decomposed_dag, parent)
            if not source_last:
                continue
            for child in children:
                target_first = self._get_first_subnode(decomposed_dag, child)
                if not target_first:
                    continue
                decomposed_dag.add_dependency(source_last, target_first)
                edge_updates.append({"source": source_last, "target": target_first})
        return edge_updates

    def _get_subnode_names(self, dag: RequirementDAG, parent_name: str) -> List[str]:
        """Return ordered subnode names for a parent requirement."""
        subnodes = [
            node for node in dag.nodes.values()
            if node.metadata.get("parent") == parent_name
        ]
        if not subnodes and parent_name in dag.nodes:
            return [parent_name]
        
        subnodes_sorted = sorted(
            subnodes,
            key=lambda n: (n.metadata.get("order", 0), n.name)
        )
        names = [node.name for node in subnodes_sorted]
        if parent_name in dag.nodes and parent_name not in names:
            names.append(parent_name)
        return names

    def _get_first_subnode(self, dag: RequirementDAG, parent_name: str) -> str | None:
        names = self._get_subnode_names(dag, parent_name)
        return names[0] if names else None

    def _get_last_subnode(self, dag: RequirementDAG, parent_name: str) -> str | None:
        names = self._get_subnode_names(dag, parent_name)
        return names[-1] if names else None

    def _normalize_parent_names(self, names: List[Any]) -> List[str]:
        normalized: List[str] = []
        for entry in names:
            if isinstance(entry, list):
                normalized.extend(self._normalize_parent_names(entry))
                continue
            if isinstance(entry, str) and entry:
                normalized.append(entry)
        return normalized
