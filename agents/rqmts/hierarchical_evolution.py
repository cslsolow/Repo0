"""Hierarchical DAG Evolution Agent for two-level DAG management."""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import datetime

from agents.cognitive.memory import MemoryAgent
from agents.infra.llm_client import LLMClient

from .dag import RequirementNode
from .dag_evolution import DAGOperation, DAGEvolutionAgent
from .hierarchical_dag import HierarchicalDAG, HighLevelRequirement, TaskNode


class HierarchicalDAGEvolutionAgent:
    """Manages evolution of hierarchical DAGs with high-level and task-level graphs."""
    
    def __init__(
        self,
        hierarchical_dag: HierarchicalDAG,
        memory_agent: MemoryAgent | None = None,
        api_config: Dict[str, Any] | None = None,
        output_dir: str = "."
    ) -> None:
        self.hierarchical_dag = hierarchical_dag
        self.memory_agent = memory_agent
        self.api_config = api_config or {}
        self.llm_client = LLMClient(self.api_config, output_dir, agent_name="hierarchical_evolution") if self.api_config.get("api_key") else None
        self.operation_history: List[DAGOperation] = []
        
        # Create evolution agent for high-level DAG if it exists
        self.hl_evolution_agent: DAGEvolutionAgent | None = None
        if hierarchical_dag.high_level_dag:
            self.hl_evolution_agent = DAGEvolutionAgent(
                hierarchical_dag.high_level_dag,
                memory_agent,
                api_config,
                output_dir
            )
    
    def add_high_level_requirement(
        self,
        requirement: HighLevelRequirement | RequirementNode,
        parent_names: List[str] | None = None,
        context: str = ""
    ) -> Dict[str, Any]:
        """Add a new high-level requirement to the top-level DAG."""
        if not self.hierarchical_dag.high_level_dag:
            raise RuntimeError("High-level DAG not initialized")
        
        if self.hl_evolution_agent:
            result = self.hl_evolution_agent.add_sub_requirement(
                requirement,
                parent_names,
                context
            )
            
            # Record in our history
            if result.get("operation_record"):
                op_record = result["operation_record"]
                operation = DAGOperation(
                    operation_type=op_record["operation_type"],
                    timestamp=op_record["timestamp"],
                    affected_nodes=op_record["affected_nodes"],
                    details={**op_record["details"], "level": "high_level"},
                    reason=op_record["reason"]
                )
                self.operation_history.append(operation)
            
            return result
        else:
            # Fallback: simple add
            self.hierarchical_dag.high_level_dag.add_requirement(requirement, parent_names)
            return {
                "success": True,
                "action": "add",
                "new_nodes": [requirement.name],
                "level": "high_level"
            }
    
    def decompose_requirement_to_tasks(
        self,
        requirement_name: str,
        tasks: List[TaskNode] | None = None,
        task_adjacency: Dict[str, List[str]] | None = None,
        auto_decompose: bool = True
    ) -> Dict[str, Any]:
        """
        Decompose a high-level requirement into a task graph.
        
        Args:
            requirement_name: Name of the high-level requirement
            tasks: List of task nodes (if None and auto_decompose=True, uses LLM)
            task_adjacency: Task dependencies
            auto_decompose: Use LLM to automatically decompose if tasks not provided
        """
        if requirement_name not in self.hierarchical_dag.high_level_dag.nodes:
            return {
                "success": False,
                "error": f"Requirement '{requirement_name}' not found in high-level DAG"
            }
        
        # Auto-decompose using LLM if needed
        if tasks is None and auto_decompose:
            decomposition = self._llm_decompose_requirement(requirement_name)
            tasks = decomposition.get("tasks", [])
            task_adjacency = decomposition.get("adjacency", {})
        
        if not tasks:
            return {
                "success": False,
                "error": "No tasks provided and auto-decomposition failed"
            }
        
        # Create task graph
        try:
            graph_id = self.hierarchical_dag.create_task_graph(
                requirement_name,
                tasks,
                task_adjacency or {}
            )
            
            # Record operation
            timestamp = datetime.datetime.now().isoformat()
            operation = DAGOperation(
                operation_type="add",
                timestamp=timestamp,
                affected_nodes=[requirement_name],
                details={
                    "level": "task_level",
                    "graph_id": graph_id,
                    "task_count": len(tasks)
                },
                reason=f"Decomposed '{requirement_name}' into {len(tasks)} tasks"
            )
            self.operation_history.append(operation)
            
            return {
                "success": True,
                "graph_id": graph_id,
                "task_count": len(tasks),
                "tasks": [t.name for t in tasks]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_task_to_requirement(
        self,
        requirement_name: str,
        task: TaskNode,
        parent_task_names: List[str] | None = None
    ) -> Dict[str, Any]:
        """Add a new task to an existing requirement's task graph."""
        try:
            added = self.hierarchical_dag.add_tasks_to_requirement(
                requirement_name,
                [task],
                parent_task_names
            )
            
            timestamp = datetime.datetime.now().isoformat()
            operation = DAGOperation(
                operation_type="add",
                timestamp=timestamp,
                affected_nodes=[task.name],
                details={
                    "level": "task_level",
                    "requirement": requirement_name,
                    "parents": parent_task_names or []
                },
                reason=f"Added task '{task.name}' to '{requirement_name}'"
            )
            self.operation_history.append(operation)
            
            return {
                "success": True,
                "added_tasks": added,
                "requirement": requirement_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def split_high_level_requirement(
        self,
        requirement_name: str,
        sub_requirements: List[HighLevelRequirement | RequirementNode],
        preserve_tasks: bool = True
    ) -> Dict[str, Any]:
        """
        Split a high-level requirement into sub-requirements.
        
        Args:
            requirement_name: Name of requirement to split
            sub_requirements: New sub-requirements
            preserve_tasks: If True, distribute tasks among sub-requirements
        """
        if not self.hierarchical_dag.high_level_dag:
            return {"success": False, "error": "High-level DAG not initialized"}
        
        # Get existing tasks if any
        existing_tasks = None
        if preserve_tasks:
            existing_tasks = self.hierarchical_dag.get_all_tasks_for_requirement(requirement_name)
        
        # Split the high-level requirement
        try:
            created = self.hierarchical_dag.high_level_dag.split_requirement(
                requirement_name,
                sub_requirements,
                preserve_edges=True
            )
            
            # Redistribute tasks if needed
            if existing_tasks and preserve_tasks:
                tasks_per_req = len(existing_tasks) // len(created)
                for i, sub_req_name in enumerate(created):
                    start_idx = i * tasks_per_req
                    end_idx = start_idx + tasks_per_req if i < len(created) - 1 else len(existing_tasks)
                    sub_tasks = existing_tasks[start_idx:end_idx]
                    
                    if sub_tasks:
                        # Update parent reference
                        for task in sub_tasks:
                            task.parent_requirement = sub_req_name
                        
                        # Create task graph for sub-requirement
                        self.hierarchical_dag.create_task_graph(
                            sub_req_name,
                            sub_tasks,
                            {}  # Simple sequential dependencies
                        )
            
            timestamp = datetime.datetime.now().isoformat()
            operation = DAGOperation(
                operation_type="split",
                timestamp=timestamp,
                affected_nodes=[requirement_name] + created,
                details={
                    "level": "high_level",
                    "original": requirement_name,
                    "created": created,
                    "tasks_redistributed": preserve_tasks
                },
                reason=f"Split '{requirement_name}' into {len(created)} sub-requirements"
            )
            self.operation_history.append(operation)
            
            return {
                "success": True,
                "created_requirements": created,
                "original": requirement_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _llm_decompose_requirement(self, requirement_name: str) -> Dict[str, Any]:
        """Use LLM to decompose a high-level requirement into tasks."""
        if not self.llm_client:
            logging.warning(f"LLM client not available for decomposing requirement '{requirement_name}', returning empty tasks")
            return {"tasks": [], "adjacency": {}}
        
        requirement = self.hierarchical_dag.high_level_dag.nodes[requirement_name]
        
        prompt = f"""Decompose the following high-level requirement into concrete, actionable tasks.

Requirement: {requirement.name}
Description: {requirement.description}

Create 3-7 specific tasks that would implement this requirement. Each task should be:
1. Concrete and actionable
2. Small enough to be completed independently
3. Clearly defined with measurable outcomes

Return ONLY a JSON object:
{{
  "tasks": [
    {{
      "name": "Task name (concise)",
      "description": "What needs to be done",
      "estimated_effort": 1-5 (story points or hours)
    }}
  ],
  "adjacency": {{
    "TaskName1": ["TaskName2", "TaskName3"],
    "TaskName2": []
  }}
}}"""

        try:
            response = self.llm_client.call_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            # Handle case where LLM returns a list instead of dict
            if isinstance(response, list):
                logging.warning("LLM returned list instead of dict for task decomposition, using empty result")
                return {"tasks": [], "adjacency": {}}
            
            tasks = []
            for task_data in response.get("tasks", []):
                task = TaskNode(
                    name=task_data.get("name", "Unnamed Task"),
                    description=task_data.get("description", ""),
                    parent_requirement=requirement_name,
                    estimated_effort=task_data.get("estimated_effort", 1),
                    status="todo"
                )
                tasks.append(task)
            
            adjacency = response.get("adjacency", {})
            
            return {
                "tasks": tasks,
                "adjacency": adjacency
            }
        except Exception as e:
            logging.warning(f"LLM decomposition failed: {e}")
            return {"tasks": [], "adjacency": {}}
    
    def get_execution_plan(self) -> List[Dict[str, Any]]:
        """Get the complete execution plan for both levels."""
        return self.hierarchical_dag.get_full_execution_order()
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get all operations performed on the hierarchical DAG."""
        return [op.to_dict() for op in self.operation_history]
    
    def export_state(self) -> Dict[str, Any]:
        """Export complete hierarchical DAG state."""
        return {
            "hierarchical_dag": self.hierarchical_dag.to_dict(),
            "operations": self.get_operation_history(),
            "summary": self.hierarchical_dag.summary()
        }
