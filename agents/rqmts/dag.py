"""Utilities for loading requirement DAGs from README-derived artifacts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable


@dataclass
class RequirementNode:
    """Single requirement extracted from readme_output artifacts."""

    name: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
        }


class RequirementDAG:
    """Directed acyclic graph describing requirement dependencies."""

    def __init__(self, nodes: Dict[str, RequirementNode], adjacency: Dict[str, Iterable[str]]) -> None:
        self.nodes = dict(nodes)
        self.adjacency: dict[str, set[str]] = {}
        for name in self.nodes:
            targets = set(adjacency.get(name, []))
            # Filter out targets that are not in nodes
            valid_targets = {target for target in targets if target in self.nodes}
            self.adjacency[name] = valid_targets
        self.reverse_adjacency: dict[str, set[str]] = {name: set() for name in self.nodes}
        for source, targets in self.adjacency.items():
            for target in targets:
                self.reverse_adjacency[target].add(source)

    @classmethod
    def from_repo(cls, repo_root: Path) -> "RequirementDAG":
        readme_dir = repo_root / "readme_output"
        requirements_path = readme_dir / "requirements.json"
        edges_path = readme_dir / "edges.json"
        return cls.from_files(requirements_path, edges_path)

    @classmethod
    def from_files(cls, requirements_path: Path, edges_path: Path) -> "RequirementDAG":
        """Load a DAG from explicit requirements/edges artifact paths."""
        nodes = cls._load_requirements(requirements_path)
        adjacency = cls._load_edges(edges_path, list(nodes.keys()))
        return cls(nodes, adjacency)

    @staticmethod
    def _load_requirements(path: Path) -> dict[str, RequirementNode]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict) and "requirements" in data:
            raw_requirements = data.get("requirements", [])
        elif isinstance(data, list):
            raw_requirements = data
        else:
            raw_requirements = []
        nodes: dict[str, RequirementNode] = {}
        for entry in raw_requirements:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            description = str(entry.get("description", "")).strip()
            if not name:
                continue
            metadata = {k: v for k, v in entry.items() if k not in {"name", "description"}}
            nodes[name] = RequirementNode(name=name, description=description, metadata=metadata)
        return nodes

    @staticmethod
    def _load_edges(path: Path, node_names: Iterable[str]) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = {name: set() for name in node_names}
        if not path.exists():
            return adjacency
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return adjacency
        edges: list[tuple[str, str]] = []
        if isinstance(data, dict):
            for source, targets in data.items():
                if isinstance(targets, list):
                    edges.extend((source, str(target)) for target in targets)
                else:
                    edges.append((source, str(targets)))
        elif isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    source = str(entry.get("source", ""))
                    target = str(entry.get("target", ""))
                    edges.append((source, target))
                elif (
                    isinstance(entry, (list, tuple))
                    and len(entry) == 2
                ):
                    edges.append((str(entry[0]), str(entry[1])))
        for source, target in edges:
            source = source.strip()
            target = target.strip()
            if source in adjacency and target:
                adjacency[source].add(target)
        return adjacency

    def is_empty(self) -> bool:
        return not self.nodes

    def dependencies(self, requirement_name: str) -> list[str]:
        return sorted(self.reverse_adjacency.get(requirement_name, set()))

    def topological_order(self) -> list[RequirementNode]:
        # Prefer zero-indegree nodes that have children (out-degree) when selecting
        indegree = {name: len(self.reverse_adjacency.get(name, set())) for name in self.nodes}
        available = {name for name, degree in indegree.items() if degree == 0}
        order: list[RequirementNode] = []
        seen: set[str] = set()

        def _priority_key(name: str) -> tuple[int, str]:
            # Higher out-degree (more children) should come first -> use negative
            out_degree = len(self.adjacency.get(name, set()))
            return (-out_degree, name)

        while available:
            # Choose the available node with highest priority (most children, then name)
            current = min(available, key=_priority_key)
            available.remove(current)
            seen.add(current)
            order.append(self.nodes[current])
            for neighbor in sorted(self.adjacency.get(current, set())):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    available.add(neighbor)

        # If there are nodes not seen due to cycles or missing edges, append them deterministically
        if len(order) != len(self.nodes):
            missing = sorted(set(self.nodes) - seen)
            order.extend(self.nodes[name] for name in missing)
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [
                {"source": source, "target": target}
                for source, targets in self.adjacency.items()
                for target in sorted(targets)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequirementDAG":
        """Reconstruct a RequirementDAG from dictionary representation."""
        nodes: dict[str, RequirementNode] = {}
        
        # Parse nodes
        for node_data in data.get("nodes", []):
            node = RequirementNode(
                name=node_data["name"],
                description=node_data["description"],
                metadata=node_data.get("metadata", {})
            )
            nodes[node.name] = node
        
        # Parse edges to build adjacency
        adjacency: dict[str, set[str]] = {name: set() for name in nodes}
        for edge in data.get("edges", []):
            source = edge["source"]
            target = edge["target"]
            if source in adjacency:
                adjacency[source].add(target)
        
        return cls(nodes, adjacency)

    def summary(self) -> dict[str, Any]:
        edge_count = sum(len(targets) for targets in self.adjacency.values())
        # no dependency
        roots = sorted(name for name, deps in self.reverse_adjacency.items() if not deps)
        # no Successor
        leaves = sorted(name for name, targets in self.adjacency.items() if not targets)
        return {
            "node_count": len(self.nodes),
            "edge_count": edge_count,
            "roots": roots,
            "leaves": leaves,
        }

    def add_requirement(self, node: RequirementNode, parent_names: list[str] | None = None) -> None:
        """Add a new requirement node to the DAG with optional parent dependencies."""
        if node.name in self.nodes:
            raise ValueError(f"Requirement '{node.name}' already exists in DAG")
        
        self.nodes[node.name] = node
        self.adjacency[node.name] = set()
        self.reverse_adjacency[node.name] = set()
        
        # Add edges from parents to this node
        if parent_names:
            for parent in parent_names:
                if parent in self.nodes:
                    self.adjacency[parent].add(node.name)
                    self.reverse_adjacency[node.name].add(parent)

    def add_dependency(self, source: str, target: str) -> None:
        """Add a dependency edge between existing nodes."""
        if source not in self.nodes:
            raise ValueError(f"Source requirement '{source}' not found in DAG")
        if target not in self.nodes:
            raise ValueError(f"Target requirement '{target}' not found in DAG")
        self.adjacency[source].add(target)
        self.reverse_adjacency[target].add(source)

    def split_requirement(
        self, 
        original_name: str, 
        sub_requirements: list[RequirementNode],
        preserve_edges: bool = True
    ) -> list[str]:
        """
        Split a requirement into multiple sub-requirements.
        
        Args:
            original_name: Name of the requirement to split
            sub_requirements: List of new sub-requirement nodes
            preserve_edges: If True, transfer original's dependencies to sub-requirements
            
        Returns:
            List of names of newly created sub-requirements
        """
        if original_name not in self.nodes:
            raise ValueError(f"Requirement '{original_name}' not found in DAG")
        
        original_node = self.nodes[original_name]
        created_names = []
        
        # Get original dependencies
        parents = list(self.reverse_adjacency.get(original_name, set()))
        children = list(self.adjacency.get(original_name, set()))
        
        # Add all sub-requirements
        for i, sub_req in enumerate(sub_requirements):
            # Ensure unique names
            if sub_req.name in self.nodes:
                sub_req.name = f"{sub_req.name}_{i}"
            
            # Add metadata tracking the split
            sub_req.metadata["split_from"] = original_name
            sub_req.metadata["split_index"] = i
            
            self.nodes[sub_req.name] = sub_req
            self.adjacency[sub_req.name] = set()
            self.reverse_adjacency[sub_req.name] = set()
            created_names.append(sub_req.name)
        
        if preserve_edges and created_names:
            # Connect parents to first sub-requirement
            for parent in parents:
                self.adjacency[parent].discard(original_name)
                self.adjacency[parent].add(created_names[0])
                self.reverse_adjacency[created_names[0]].add(parent)
            
            # Chain sub-requirements
            for i in range(len(created_names) - 1):
                self.adjacency[created_names[i]].add(created_names[i + 1])
                self.reverse_adjacency[created_names[i + 1]].add(created_names[i])
            
            # Connect last sub-requirement to children
            for child in children:
                self.reverse_adjacency[child].discard(original_name)
                self.adjacency[created_names[-1]].add(child)
                self.reverse_adjacency[child].add(created_names[-1])
        
        # Remove original node
        self._remove_node(original_name)
        
        return created_names

    def merge_requirements(
        self, 
        requirement_names: list[str], 
        merged_node: RequirementNode,
        merge_strategy: str = "union"
    ) -> str:
        """
        Merge multiple requirements into a single requirement.
        
        Args:
            requirement_names: Names of requirements to merge
            merged_node: The new merged requirement node
            merge_strategy: "union" or "intersection" for edge handling
            
        Returns:
            Name of the merged requirement
        """
        if not requirement_names:
            raise ValueError("Must provide at least one requirement to merge")
        
        for name in requirement_names:
            if name not in self.nodes:
                raise ValueError(f"Requirement '{name}' not found in DAG")
        
        # Collect all parents and children
        all_parents: set[str] = set()
        all_children: set[str] = set()
        
        for name in requirement_names:
            all_parents.update(self.reverse_adjacency.get(name, set()))
            all_children.update(self.adjacency.get(name, set()))
        
        # Remove self-references
        all_parents -= set(requirement_names)
        all_children -= set(requirement_names)
        
        # Add metadata tracking the merge
        merged_node.metadata["merged_from"] = requirement_names
        merged_node.metadata["merge_strategy"] = merge_strategy
        
        # Ensure unique name
        if merged_node.name in self.nodes and merged_node.name not in requirement_names:
            raise ValueError(f"Merged requirement name '{merged_node.name}' already exists")
        
        # Remove original nodes
        for name in requirement_names:
            self._remove_node(name)
        
        # Add merged node
        self.nodes[merged_node.name] = merged_node
        self.adjacency[merged_node.name] = set()
        self.reverse_adjacency[merged_node.name] = set()
        
        # Connect edges based on strategy
        if merge_strategy == "union":
            # Connect all parents and children
            for parent in all_parents:
                self.adjacency[parent].add(merged_node.name)
                self.reverse_adjacency[merged_node.name].add(parent)
            
            for child in all_children:
                self.adjacency[merged_node.name].add(child)
                self.reverse_adjacency[child].add(merged_node.name)
        
        return merged_node.name

    def delete_requirement(self, requirement_name: str, reconnect: bool = True) -> None:
        """
        Delete a requirement from the DAG.
        
        Args:
            requirement_name: Name of the requirement to delete
            reconnect: If True, connect parents directly to children
        """
        if requirement_name not in self.nodes:
            raise ValueError(f"Requirement '{requirement_name}' not found in DAG")
        
        if reconnect:
            parents = list(self.reverse_adjacency.get(requirement_name, set()))
            children = list(self.adjacency.get(requirement_name, set()))
            
            # Connect each parent to each child
            for parent in parents:
                for child in children:
                    self.adjacency[parent].add(child)
                    self.reverse_adjacency[child].add(parent)
        
        self._remove_node(requirement_name)

    def revise_requirement(
        self, 
        original_name: str, 
        new_node: RequirementNode,
        preserve_edges: bool = True
    ) -> str:
        """
        Revise (replace) a requirement with a new version.
        
        Args:
            original_name: Name of the requirement to revise
            new_node: The new requirement node
            preserve_edges: If True, preserve all edges
            
        Returns:
            Name of the new requirement
        """
        if original_name not in self.nodes:
            raise ValueError(f"Requirement '{original_name}' not found in DAG")
        
        # Add metadata tracking the revise
        new_node.metadata["revised_from"] = original_name
        
        if preserve_edges:
            parents = list(self.reverse_adjacency.get(original_name, set()))
            children = list(self.adjacency.get(original_name, set()))
            
            # Remove original
            self._remove_node(original_name)
            
            # Add new node
            self.nodes[new_node.name] = new_node
            self.adjacency[new_node.name] = set()
            self.reverse_adjacency[new_node.name] = set()
            
            # Restore edges
            for parent in parents:
                self.adjacency[parent].add(new_node.name)
                self.reverse_adjacency[new_node.name].add(parent)
            
            for child in children:
                self.adjacency[new_node.name].add(child)
                self.reverse_adjacency[child].add(new_node.name)
        else:
            self._remove_node(original_name)
            self.nodes[new_node.name] = new_node
            self.adjacency[new_node.name] = set()
            self.reverse_adjacency[new_node.name] = set()
        
        return new_node.name

    def _remove_node(self, node_name: str) -> None:
        """Internal method to remove a node and clean up all edges."""
        if node_name not in self.nodes:
            return
        
        # Remove from adjacency lists
        for parent in self.reverse_adjacency.get(node_name, set()):
            self.adjacency[parent].discard(node_name)
        
        for child in self.adjacency.get(node_name, set()):
            self.reverse_adjacency[child].discard(node_name)
        
        # Remove the node itself
        del self.nodes[node_name]
        del self.adjacency[node_name]
        del self.reverse_adjacency[node_name]
