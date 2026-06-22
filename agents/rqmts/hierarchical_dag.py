"""Hierarchical DAG structure with high-level requirements and task-level subgraphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set
import json

from .dag import RequirementDAG, RequirementNode


@dataclass
class HighLevelRequirement(RequirementNode):
    """High-level requirement that can be decomposed into tasks."""
    
    task_graph_id: str | None = None  # Reference to associated task graph
    decomposition_status: str = "pending"  # pending, decomposed, implemented
    
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["task_graph_id"] = self.task_graph_id
        base["decomposition_status"] = self.decomposition_status
        return base
    
@dataclass
class TaskNode(RequirementNode):
    """Low-level task node in the task graph."""
    
    parent_requirement: str | None = None  # Reference to high-level requirement
    estimated_effort: int = 1  # Effort estimate (story points, hours, etc.)
    status: str = "todo"  # todo, in_progress, done
    
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["parent_requirement"] = self.parent_requirement
        base["estimated_effort"] = self.estimated_effort
        base["status"] = self.status
        return base


class HierarchicalDAG:
    """
    Two-level DAG structure:
    - High-level DAG: Strategic requirements and their dependencies
    - Task-level DAGs: Concrete tasks decomposed from each high-level requirement
    """
    
    def __init__(self):
        self.high_level_dag: RequirementDAG | None = None
        self.task_graphs: Dict[str, RequirementDAG] = {}  # requirement_name -> task DAG
        self.requirement_to_graph: Dict[str, str] = {}  # requirement_name -> graph_id
    
    def set_high_level_dag(self, dag: RequirementDAG) -> None:
        """Set the high-level requirements DAG."""
        self.high_level_dag = dag
    
    def create_task_graph(
        self, 
        requirement_name: str,
        tasks: List[TaskNode],
        task_adjacency: Dict[str, List[str]]
    ) -> str:
        """
        Create a task-level subgraph for a high-level requirement.
        
        Args:
            requirement_name: Name of the high-level requirement
            tasks: List of task nodes
            task_adjacency: Dependencies between tasks
            
        Returns:
            Graph ID for the created task graph
        """
        if not self.high_level_dag or requirement_name not in self.high_level_dag.nodes:
            raise ValueError(f"High-level requirement '{requirement_name}' not found")
        
        # Create graph ID
        graph_id = f"tasks_{requirement_name}"
        
        # Link tasks to parent requirement
        for task in tasks:
            task.parent_requirement = requirement_name
        
        # Create task DAG
        task_nodes = {task.name: task for task in tasks}
        task_dag = RequirementDAG(task_nodes, task_adjacency)
        
        # Store the task graph
        self.task_graphs[graph_id] = task_dag
        self.requirement_to_graph[requirement_name] = graph_id
        
        # Update high-level requirement
        hl_req = self.high_level_dag.nodes[requirement_name]
        if hasattr(hl_req, 'task_graph_id'):
            hl_req.task_graph_id = graph_id
            hl_req.decomposition_status = "decomposed"
        else:
            hl_req.metadata["task_graph_id"] = graph_id
            hl_req.metadata["decomposition_status"] = "decomposed"
        
        return graph_id
    
    def get_task_graph(self, requirement_name: str) -> RequirementDAG | None:
        """Get the task graph for a high-level requirement."""
        graph_id = self.requirement_to_graph.get(requirement_name)
        if graph_id:
            return self.task_graphs.get(graph_id)
        return None
    
    def add_tasks_to_requirement(
        self,
        requirement_name: str,
        new_tasks: List[TaskNode],
        parent_task_names: List[str] | None = None
    ) -> List[str]:
        """Add new tasks to an existing task graph."""
        task_graph = self.get_task_graph(requirement_name)
        if not task_graph:
            # Create new task graph if doesn't exist
            adjacency = {}
            if parent_task_names:
                for task in new_tasks:
                    adjacency[task.name] = []
            return self.create_task_graph(requirement_name, new_tasks, adjacency)
        
        # Add tasks to existing graph
        added_names = []
        for task in new_tasks:
            task.parent_requirement = requirement_name
            task_graph.add_requirement(task, parent_task_names)
            added_names.append(task.name)
        
        return added_names
    
    def get_all_tasks_for_requirement(self, requirement_name: str) -> List[TaskNode]:
        """Get all tasks associated with a high-level requirement."""
        task_graph = self.get_task_graph(requirement_name)
        if not task_graph:
            return []
        return list(task_graph.nodes.values())
    
    def get_decomposition_progress(self, requirement_name: str) -> Dict[str, Any]:
        """Get progress metrics for a high-level requirement's tasks."""
        tasks = self.get_all_tasks_for_requirement(requirement_name)
        if not tasks:
            return {
                "total_tasks": 0,
                "completed": 0,
                "in_progress": 0,
                "todo": 0,
                "progress_percentage": 0.0
            }
        
        status_counts = {"todo": 0, "in_progress": 0, "done": 0}
        total_effort = 0
        completed_effort = 0
        
        for task in tasks:
            status = getattr(task, 'status', 'todo')
            effort = getattr(task, 'estimated_effort', 1)
            
            status_counts[status] = status_counts.get(status, 0) + 1
            total_effort += effort
            
            if status == "done":
                completed_effort += effort
        
        return {
            "total_tasks": len(tasks),
            "completed": status_counts.get("done", 0),
            "in_progress": status_counts.get("in_progress", 0),
            "todo": status_counts.get("todo", 0),
            "total_effort": total_effort,
            "completed_effort": completed_effort,
            "progress_percentage": (completed_effort / total_effort * 100) if total_effort > 0 else 0.0
        }
    
    def get_full_execution_order(self) -> List[Dict[str, Any]]:
        """
        Get a complete execution order considering both levels:
        1. High-level requirements in topological order
        2. For each requirement, its tasks in topological order
        """
        if not self.high_level_dag:
            return []
        
        execution_plan = []
        
        for hl_req in self.high_level_dag.topological_order():
            req_entry = {
                "requirement": hl_req.name,
                "description": hl_req.description,
                "type": "high_level",
                "tasks": []
            }
            
            # Get tasks for this requirement
            task_graph = self.get_task_graph(hl_req.name)
            if task_graph:
                for task in task_graph.topological_order():
                    req_entry["tasks"].append({
                        "name": task.name,
                        "description": task.description,
                        "status": getattr(task, 'status', 'todo'),
                        "effort": getattr(task, 'estimated_effort', 1),
                        "dependencies": task_graph.dependencies(task.name)
                    })
            
            execution_plan.append(req_entry)
        
        return execution_plan
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics for the hierarchical DAG."""
        hl_summary = self.high_level_dag.summary() if self.high_level_dag else {}
        
        total_tasks = sum(len(graph.nodes) for graph in self.task_graphs.values())
        total_task_edges = sum(
            sum(len(targets) for targets in graph.adjacency.values())
            for graph in self.task_graphs.values()
        )
        
        # Calculate overall progress
        all_progress = []
        for req_name in self.requirement_to_graph.keys():
            progress = self.get_decomposition_progress(req_name)
            all_progress.append(progress)
        
        total_completed = sum(p["completed"] for p in all_progress)
        total_all_tasks = sum(p["total_tasks"] for p in all_progress)
        
        return {
            "high_level": {
                "requirements": hl_summary.get("node_count", 0),
                "dependencies": hl_summary.get("edge_count", 0),
                "roots": hl_summary.get("roots", []),
                "leaves": hl_summary.get("leaves", [])
            },
            "task_level": {
                "total_graphs": len(self.task_graphs),
                "total_tasks": total_tasks,
                "total_dependencies": total_task_edges,
                "decomposed_requirements": len(self.requirement_to_graph)
            },
            "progress": {
                "total_tasks": total_all_tasks,
                "completed_tasks": total_completed,
                "completion_percentage": (total_completed / total_all_tasks * 100) if total_all_tasks > 0 else 0.0
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export the entire hierarchical DAG structure."""
        return {
            "high_level_dag": self.high_level_dag.to_dict() if self.high_level_dag else None,
            "task_graphs": {
                graph_id: graph.to_dict()
                for graph_id, graph in self.task_graphs.items()
            },
            "requirement_to_graph": self.requirement_to_graph,
            "summary": self.summary()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchicalDAG":
        """Reconstruct a HierarchicalDAG from exported data."""
        hierarchical_dag = cls()
        
        # Reconstruct high-level DAG
        if data.get("high_level_dag"):
            hl_data = data["high_level_dag"]
            nodes = {}
            for node_data in hl_data.get("nodes", []):
                node = RequirementNode(
                    name=node_data["name"],
                    description=node_data["description"],
                    metadata=node_data.get("metadata", {})
                )
                nodes[node.name] = node
            
            adjacency = {}
            for edge in hl_data.get("edges", []):
                source = edge["source"]
                target = edge["target"]
                if source not in adjacency:
                    adjacency[source] = []
                adjacency[source].append(target)
            
            hierarchical_dag.high_level_dag = RequirementDAG(nodes, adjacency)
        
        # Reconstruct task graphs
        for graph_id, graph_data in data.get("task_graphs", {}).items():
            task_nodes = {}
            for node_data in graph_data.get("nodes", []):
                task = TaskNode(
                    name=node_data["name"],
                    description=node_data["description"],
                    metadata=node_data.get("metadata", {}),
                    parent_requirement=node_data.get("parent_requirement"),
                    estimated_effort=node_data.get("estimated_effort", 1),
                    status=node_data.get("status", "todo")
                )
                task_nodes[task.name] = task
            
            task_adjacency = {}
            for edge in graph_data.get("edges", []):
                source = edge["source"]
                target = edge["target"]
                if source not in task_adjacency:
                    task_adjacency[source] = []
                task_adjacency[source].append(target)
            
            hierarchical_dag.task_graphs[graph_id] = RequirementDAG(task_nodes, task_adjacency)
        
        hierarchical_dag.requirement_to_graph = data.get("requirement_to_graph", {})
        
        return hierarchical_dag
