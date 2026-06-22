"""Memory agent that captures a lightweight structural view of the repos folder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import json
from pathlib import Path
from typing import Any, Iterable, List, Dict, Optional

from agents.coding.structured_contracts import extract_structured_contract_facts


@dataclass
class ComponentImplementation:
    """Metadata for an implemented component."""
    
    component_name: str  # Name of the component
    requirement_node: str  # DAG node this component implements
    file_path: str  # Relative path from repo root
    class_names: List[str] = field(default_factory=list)  # Classes defined in this component
    function_signatures: List[Dict[str, Any]] = field(default_factory=list)  # Function signatures
    dependencies: List[str] = field(default_factory=list)  # Other components/modules this depends on
    exports: List[str] = field(default_factory=list)  # Public API: classes/functions exposed
    status: str = "implemented"  # implemented, tested, documented
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass
class MemorySnapshot:
    """Structured view of the repo tree plus the most recent planning context."""

    repo_name: str | None
    files: list[str]
    requirements: list[str]
    notes: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    architecture: dict[str, Any] | None = None
    dag_operations: list[dict[str, Any]] = field(default_factory=list)
    implemented_components: Dict[str, ComponentImplementation] = field(default_factory=dict)  # requirement_node -> implementation

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemorySnapshot":
        """Reconstruct a MemorySnapshot from stored JSON data."""
        implemented_components: Dict[str, ComponentImplementation] = {}
        for key, value in data.get("implemented_components", {}).items():
            if isinstance(value, dict):
                implemented_components[key] = ComponentImplementation(**value)
        return cls(
            repo_name=data.get("repo_name"),
            files=list(data.get("files", [])),
            requirements=list(data.get("requirements", [])),
            notes=data.get("notes", ""),
            actions=list(data.get("actions", [])),
            architecture=data.get("architecture"),
            dag_operations=list(data.get("dag_operations", [])),
            implemented_components=implemented_components,
        )


class MemoryAgent:
    """Builds and persists repository memory for the other agents."""

    def __init__(self, workspace_root: Path, repos_dir: str = "repos") -> None:
        self.workspace_root = Path(workspace_root)
        self.repos_dir = self.workspace_root / repos_dir
        self.snapshot: MemorySnapshot | None = None

    def build_memory(self, repo_name: str | None = None, limit: int = 60) -> MemorySnapshot:
        """Scan the repo (or a specific project) and cache a structured snapshot."""
        target_root = self._resolve_target(repo_name)
        files = self._collect_files(target_root, limit)
        requirements = self._load_requirements(target_root)
        notes = (
            f"Scanned {target_root.name} with {len(files)} tracked files and "
            f"{len(requirements)} inferred requirements."
        )
        self.snapshot = MemorySnapshot(
            repo_name=repo_name,
            files=files,
            requirements=requirements,
            notes=notes,
        )
        return self.snapshot

    def register_actions(
        self,
        actions: Iterable[dict[str, Any]],
        architecture: dict[str, Any] | None = None,
    ) -> MemorySnapshot:
        """Attach chosen actions and architectural context to the snapshot."""
        snapshot = self._require_snapshot()
        snapshot.actions.extend(list(actions))
        snapshot.architecture = architecture or snapshot.architecture
        return snapshot

    def persist(self, output_dir: Path) -> Path:
        """Write the in-memory snapshot to disk for downstream tools."""
        snapshot = self._require_snapshot()
        output_dir.mkdir(parents=True, exist_ok=True)
        memory_path = output_dir / "memory.json"
        with open(memory_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(snapshot.to_dict(), indent=2))
        return memory_path

    def load_snapshot(self, memory_path: Path) -> MemorySnapshot:
        """Load a previously persisted snapshot from disk."""
        if not memory_path.exists():
            raise FileNotFoundError(f"Memory snapshot not found: {memory_path}")
        data = json.loads(memory_path.read_text(encoding="utf-8"))
        snapshot = MemorySnapshot.from_dict(data)
        self.snapshot = snapshot
        return snapshot
    
    def record_dag_operation(self, operation: dict[str, Any]) -> None:
        """Record a DAG operation in the memory snapshot."""
        snapshot = self._require_snapshot()
        snapshot.dag_operations.append(operation)
    
    def get_dag_history(self) -> list[dict[str, Any]]:
        """Get the history of all DAG operations."""
        snapshot = self._require_snapshot()
        return snapshot.dag_operations
    
    def register_component_implementation(
        self,
        component_name: str,
        requirement_node: str,
        file_path: str,
        class_names: List[str] | None = None,
        function_signatures: List[Dict[str, Any]] | None = None,
        dependencies: List[str] | None = None,
        exports: List[str] | None = None,
        status: str = "implemented",
        **kwargs: Any
    ) -> ComponentImplementation:
        """Register a component implementation in memory.
        
        This method can be called multiple times for the same requirement_node:
        - First with status="planned" during architecture generation
        - Then with status="implemented" after code generation
        
        Args:
            component_name: Name of the component
            requirement_node: DAG node this component implements
            file_path: Relative path from repo root
            class_names: List of class names defined
            function_signatures: List of function signature dicts with 'name', 'params', 'return_type'
            dependencies: List of dependencies (modules/components)
            exports: Public API exposed by this component
            status: Implementation status (planned, implemented, tested)
            **kwargs: Additional metadata
            
        Returns:
            The created ComponentImplementation object
        """
        snapshot = self._require_snapshot()
        
        # Create unique key for each component under a parent
        # Format: parent_requirement::component_name
        component_key = f"{requirement_node}::{component_name}"
        existing = snapshot.implemented_components.get(component_key)

        # Never downgrade an already implemented component back to planned.
        if existing is not None:
            existing_rank = {"planned": 0, "implemented": 1, "tested": 2, "documented": 3}.get(existing.status, 0)
            new_rank = {"planned": 0, "implemented": 1, "tested": 2, "documented": 3}.get(status, 0)
            if new_rank < existing_rank:
                logging.info(
                    "  Preserving %s at status=%s; ignoring downgrade to %s",
                    component_key,
                    existing.status,
                    status,
                )
                return existing
        
        merged_metadata = dict(existing.metadata) if existing is not None else {}
        merged_metadata.update(kwargs)

        impl = ComponentImplementation(
            component_name=component_name,
            requirement_node=requirement_node,
            file_path=file_path,
            class_names=class_names or [],
            function_signatures=function_signatures or [],
            dependencies=dependencies or [],
            exports=exports or [],
            status=status,
            metadata=merged_metadata,
        )
        
        # Use component_key to allow multiple components per parent
        snapshot.implemented_components[component_key] = impl
        
        if existing and existing.status == "planned" and status == "implemented":
            logging.info(f"  Updated {component_name} from planned to implemented")
        
        return impl
    
    def get_implemented_components(self, requirement_node: Optional[str] = None, status_filter: Optional[str] = None) -> Dict[str, ComponentImplementation]:
        """Get implemented components, optionally filtered by requirement node and status.
        
        Args:
            requirement_node: Optional requirement node prefix to filter by
            status_filter: Optional status to filter by (e.g., "implemented", not "planned")
            
        Returns:
            Dictionary of component_key -> ComponentImplementation
        """
        snapshot = self._require_snapshot()
        
        result = {}
        for key, impl in snapshot.implemented_components.items():
            # Apply requirement_node filter
            if requirement_node and not key.startswith(f"{requirement_node}::"):
                continue
            
            # Apply status filter
            if status_filter and impl.status != status_filter:
                continue
                
            result[key] = impl
        
        return result
    
    def query_component(self, requirement_node: str) -> Optional[ComponentImplementation]:
        """Query implementation details for a specific requirement node.
        
        Args:
            requirement_node: The requirement node to query
            
        Returns:
            ComponentImplementation if found, None otherwise
        """
        snapshot = self._require_snapshot()
        return snapshot.implemented_components.get(requirement_node)
    
    def get_available_dependencies(self) -> Dict[str, List[str]]:
        """Get all available dependencies from implemented components.
        
        Returns:
            Dictionary mapping component names to their exported APIs
        """
        snapshot = self._require_snapshot()
        available_deps = {}
        
        for node, impl in snapshot.implemented_components.items():
            if impl.exports:
                available_deps[impl.component_name] = impl.exports
        
        return available_deps
    
    def format_implementations_for_prompt(self, filter_nodes: Optional[List[str]] = None, status_filter: str = "implemented") -> str:
        """Format implemented components as a string for LLM prompts.
        
        Args:
            filter_nodes: Optional list of requirement node prefixes to include
            status_filter: Only include components with this status (default: "implemented")
            
        Returns:
            Formatted string describing available implementations
        """
        snapshot = self._require_snapshot()
        
        # Get components matching filters
        impls = {}
        if filter_nodes is not None and not filter_nodes:
            return "No prerequisite components have been implemented yet."
        for key, impl in snapshot.implemented_components.items():
            # Apply status filter
            if status_filter and impl.status != status_filter:
                continue
                
            # Apply node filter (check if any filter_node is a prefix)
            if filter_nodes is not None:
                if not any(key.startswith(f"{node}::") for node in filter_nodes):
                    continue
            
            impls[key] = impl
        
        if not impls:
            return "No components have been implemented yet."
        
        lines = ["=== IMPLEMENTED COMPONENTS (Available for reuse) ==="]
        
        for key, impl in impls.items():
            # Extract parent node from key (format: parent_requirement::component_name)
            parent_node = key.split("::")[0] if "::" in key else key
            lines.append(f"\n[{impl.component_name}] (Parent: {parent_node})")
            lines.append(f"  File: {impl.file_path}")
            
            class_methods: Dict[str, List[Dict[str, Any]]] = {}
            module_functions: List[Dict[str, Any]] = []

            for sig in impl.function_signatures:
                func_name = sig.get("name", "")
                if (
                    func_name.startswith("_")
                    and not func_name.startswith("__init__")
                ):
                    continue
                class_name = sig.get("class_name")
                if class_name:
                    class_methods.setdefault(class_name, []).append(sig)
                else:
                    module_functions.append(sig)

            if impl.class_names or class_methods:
                lines.append("  Classes:")
                class_names = list(impl.class_names)
                for class_name in class_methods:
                    if class_name not in class_names:
                        class_names.append(class_name)
                for class_name in class_names:
                    lines.append(f"    - {class_name}")
                    methods = class_methods.get(class_name, [])
                    if methods:
                        lines.append("      Methods:")
                        for sig in methods[:10]:
                            func_name = sig.get('name', 'unknown')
                            params = sig.get('params', [])
                            return_type = sig.get('return_type', 'Any')
                            params_str = ', '.join(params) if isinstance(params, list) else str(params)
                            lines.append(f"        - {func_name}({params_str}) -> {return_type}")
                if module_functions:
                    lines.append("  Module Functions:")
                    for sig in module_functions[:10]:
                        func_name = sig.get('name', 'unknown')
                        params = sig.get('params', [])
                        return_type = sig.get('return_type', 'Any')
                        params_str = ', '.join(params) if isinstance(params, list) else str(params)
                        lines.append(f"    - {func_name}({params_str}) -> {return_type}")
            elif impl.function_signatures:
                lines.append("  Functions:")
                for sig in impl.function_signatures[:10]:  # Limit to 10 functions
                    func_name = sig.get('name', 'unknown')
                    params = sig.get('params', [])
                    return_type = sig.get('return_type', 'Any')
                    params_str = ', '.join(params) if isinstance(params, list) else str(params)
                    lines.append(f"    - {func_name}({params_str}) -> {return_type}")
            
            if impl.exports:
                lines.append(f"  Public API: {', '.join(impl.exports[:10])}")
            
            if impl.dependencies:
                lines.append(f"  Dependencies: {', '.join(impl.dependencies[:5])}")

            contract_facts = impl.metadata.get("structured_contract_facts", [])
            if isinstance(contract_facts, list) and contract_facts:
                lines.append("  State Contracts:")
                for fact in contract_facts[:6]:
                    lines.append(f"    - {fact}")

            contract_issues = impl.metadata.get("structured_contract_issues", [])
            if isinstance(contract_issues, list) and contract_issues:
                lines.append("  Contract Warnings:")
                for issue in contract_issues[:4]:
                    lines.append(f"    - {issue}")
            
            lines.append(f"  Status: {impl.status}")
        
        return "\n".join(lines)

    def _collect_files(self, target_root: Path, limit: int) -> list[str]:
        """Return a deterministic subset of files for quick situational awareness."""
        files: list[str] = []
        for path in sorted(target_root.rglob("*")):
            if path.is_file():
                try:
                    rel_path = path.relative_to(self.workspace_root)
                except ValueError:
                    # Allow callers to build memory for repos that live outside the
                    # original workspace root by falling back to the scanned repo root.
                    rel_path = path.relative_to(target_root)
                files.append(str(rel_path))
            if len(files) >= limit:
                break
        return files

    def _load_requirements(self, target_root: Path) -> list[str]:
        """Read requirements from README-derived JSON if present."""
        candidate = target_root / "readme_output" / "requirements.json"
        if not candidate.exists():
            return []
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
        if isinstance(data, dict):
            extracted: list[str] = []
            for value in data.values():
                if isinstance(value, str):
                    extracted.append(value.strip())
                elif isinstance(value, list):
                    extracted.extend(str(item).strip() for item in value if str(item).strip())
            return [item for item in extracted if item]
        return []

    def _resolve_target(self, repo_name: str | None) -> Path:
        if repo_name:
            target_root = self.repos_dir / repo_name
        else:
            target_root = self.repos_dir
        if not target_root.exists():
            raise FileNotFoundError(f"Could not locate target repository at {target_root}")
        return target_root

    def _require_snapshot(self) -> MemorySnapshot:
        if self.snapshot is None:
            raise RuntimeError("Memory snapshot has not been built yet.")
        return self.snapshot
