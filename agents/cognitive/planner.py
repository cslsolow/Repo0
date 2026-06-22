"""Planner agent that turns a requirement DAG into actionable tasks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agents.infra.llm_client import LLMClient
from agents.rqmts.dag import RequirementDAG


class PlannerAgent:
    """Generate a structured plan tailored to the observed environment."""

    def __init__(self, max_items: int = 5, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.max_items = max_items
        self.api_config = api_config or {}
        self.llm_client = LLMClient(self.api_config, output_dir, agent_name="planner") if self.api_config.get(
            "api_key") else None

    def create_requirement_list(self, global_requirements: str) -> List[str]:
        """Split a free-form requirement paragraph into an ordered list."""
        text = (global_requirements or "").strip()
        if not text:
            return ["Document repository structure and identify missing modules."]
        candidates: list[str] = []
        for line in text.replace(";", "\n").replace(".", "\n").splitlines():
            normalized = line.strip(" -\t")
            if normalized:
                candidates.append(normalized)
        if not candidates:
            candidates = [text]
        prioritized = candidates[: self.max_items]
        return prioritized

    def create_plan_from_dag(
            self,
            requirement_dag: RequirementDAG,
            fallback_text: str = "",
    ) -> List[Dict[str, object]]:
        """Produce an intelligent plan derived from the requirement DAG using LLM with dependency-aware planning."""
        if not self.llm_client:
            logging.warning("LLM client not available, using fallback plan generation")
            return self._fallback_plan_from_dag(requirement_dag, fallback_text)

        ordered_nodes = requirement_dag.topological_order() if not requirement_dag.is_empty() else []
        if not ordered_nodes:
            logging.warning("No ordered nodes found in DAG, using fallback plan generation")
            return self._fallback_plan_from_dag(requirement_dag, fallback_text)

        # Group nodes by dependency layers for structured planning
        dependency_layers = self._group_nodes_by_layers(ordered_nodes, requirement_dag)

        max_nodes_per_batch = 10  # Soft limit to avoid token overflow
        if len(ordered_nodes) <= max_nodes_per_batch:
            # Small DAG, single round is sufficient
            return self._create_plan_single_round(ordered_nodes, requirement_dag)
        else:
            # Large DAG, use layer-based multi-round planning
            logging.info(
                f"Using layer-based planning for {len(ordered_nodes)} nodes across {len(dependency_layers)} layers")
            return self._create_plan_layer_based(dependency_layers, requirement_dag, max_nodes_per_batch)

    def _group_nodes_by_layers(self, ordered_nodes: List, requirement_dag: RequirementDAG) -> List[List]:
        """Group nodes into dependency layers for structured planning."""
        layers = []
        processed = set()
        remaining = list(ordered_nodes)

        while remaining:
            current_layer = []
            for node in remaining[:]:
                deps = requirement_dag.dependencies(node.name)
                # Add to current layer if all dependencies are already processed
                if all(dep in processed for dep in deps):
                    current_layer.append(node)
                    processed.add(node.name)
                    remaining.remove(node)

            if not current_layer:
                # Circular dependency or error, add remaining nodes to avoid infinite loop
                logging.warning(f"Breaking dependency cycle, adding {len(remaining)} remaining nodes")
                layers.append(remaining)
                break

            layers.append(current_layer)

        return layers

    def _create_plan_layer_based(
            self,
            dependency_layers: List[List],
            requirement_dag: RequirementDAG,
            max_nodes_per_batch: int,
    ) -> List[Dict[str, object]]:
        """Create plan by processing dependency layers, ensuring prerequisites are planned first."""
        all_plans = []
        processed_nodes = set()
        dag_summary = requirement_dag.summary()

        for layer_idx, layer_nodes in enumerate(dependency_layers):
            if len(all_plans) >= self.max_items:
                logging.info(f"Reached max_items limit ({self.max_items}), stopping planning")
                break

            # Process layer in batches if needed
            for batch_start in range(0, len(layer_nodes), max_nodes_per_batch):
                batch_nodes = layer_nodes[batch_start:batch_start + max_nodes_per_batch]
                remaining_slots = self.max_items - len(all_plans)

                if remaining_slots <= 0:
                    break

                nodes_info = []
                for node in batch_nodes:
                    nodes_info.append({
                        "name": node.name,
                        "description": node.description,
                        "dependencies": requirement_dag.dependencies(node.name),
                    })

                # Context about already planned tasks
                previous_tasks_context = ""
                if all_plans:
                    previous_tasks_context = f"\n\nAlready planned tasks ({len(all_plans)}):\n"
                    previous_tasks_context += "\n".join(f"- {task['name']}" for task in all_plans[-10:])

                prompt = f"""You are a software planning expert. Create an execution plan from dependency layer {layer_idx + 1}.

DAG Summary:
- Total nodes: {dag_summary['node_count']}
- Current layer: {layer_idx + 1}/{len(dependency_layers)}
- Layer nodes: {len(layer_nodes)}
- Current batch: nodes {batch_start + 1}-{min(batch_start + max_nodes_per_batch, len(layer_nodes))}
- Remaining plan slots: {remaining_slots}{previous_tasks_context}

Layer Nodes (all dependencies in previous layers):
{chr(10).join(f"{i + 1}. {n['name']}: {n['description'][:100]}..." if len(n['description']) > 100 else f"{i + 1}. {n['name']}: {n['description']}" for i, n in enumerate(nodes_info))}

Task: Select up to {min(remaining_slots, len(nodes_info))} most important tasks from this layer. 
All dependencies for these tasks have been planned in earlier layers.

Return ONLY a JSON array of task objects with: name, description, dependencies, rationale."""

                try:
                    response = self.llm_client.call_json([
                        {"role": "system",
                         "content": "You are an expert software project planner. Always return valid JSON arrays."},
                        {"role": "user", "content": prompt}
                    ])

                    batch_plan = self._extract_plan_from_response(response)

                    # Add to overall plan with correct order
                    for task in batch_plan:
                        if task['name'] not in processed_nodes:
                            task['order'] = len(all_plans)
                            all_plans.append(task)
                            processed_nodes.add(task['name'])

                            if len(all_plans) >= self.max_items:
                                break

                    logging.info(
                        f"Layer {layer_idx + 1}, batch {batch_start // max_nodes_per_batch + 1}: added {len(batch_plan)} tasks, total: {len(all_plans)}")

                except Exception as e:
                    logging.warning(f"Layer {layer_idx + 1} batch planning failed: {e}, continuing")
                    continue

        self._enrich_plan(all_plans)
        logging.info(f"Layer-based planning completed: {len(all_plans)} tasks from {len(dependency_layers)} layers")
        return all_plans if all_plans else self._fallback_plan_from_dag(requirement_dag, "")

    def _create_plan_single_round(
            self,
            ordered_nodes: List,
            requirement_dag: RequirementDAG,
    ) -> List[Dict[str, object]]:
        """Create plan in a single LLM call for small DAGs."""
        dag_summary = requirement_dag.summary()
        nodes_info = []
        for node in ordered_nodes[:self.max_items]:
            nodes_info.append({
                "name": node.name,
                "description": node.description,
                "dependencies": requirement_dag.dependencies(node.name),
            })

        prompt = f"""You are a software planning expert. Given a requirement DAG, create an ordered execution plan.

Requirement DAG Summary:
- Total nodes: {dag_summary['node_count']}
- Total edges: {dag_summary['edge_count']}
- Root nodes: {', '.join(dag_summary['roots'][:5])}

Requirement Nodes (ordered by dependencies):
{chr(10).join(f"{i + 1}. {n['name']}: {n['description'][:100]}..." if len(n['description']) > 100 else f"{i + 1}. {n['name']}: {n['description']}" for i, n in enumerate(nodes_info))}

Task: Create a plan with at most {min(self.max_items, len(nodes_info))} prioritized tasks. For each task, specify:
- name: The requirement name
- description: What needs to be done
- order: Execution priority (0-based)
- dependencies: List of prerequisite requirement names
- status: "pending"
- rationale: Why this task was prioritized

Return ONLY a JSON array of task objects."""

        try:
            response = self.llm_client.call_json([
                {"role": "system",
                 "content": "You are an expert software project planner. Always return valid JSON arrays."},
                {"role": "user", "content": prompt}
            ])

            plan = self._extract_plan_from_response(response)
            self._enrich_plan(plan)

            logging.info(f"Single-round planning: created {len(plan)} tasks")
            return plan

        except Exception as e:
            logging.warning(f"Single-round planning failed ({e}), using fallback")
            return self._fallback_plan_from_dag(requirement_dag, "")

    def _create_plan_multi_round(
            self,
            ordered_nodes: List,
            requirement_dag: RequirementDAG,
            batch_size: int,
    ) -> List[Dict[str, object]]:
        """Create plan through multiple LLM calls for large DAGs."""
        all_plans = []
        processed_nodes = set()
        dag_summary = requirement_dag.summary()

        # Process in batches
        for batch_idx in range(0, len(ordered_nodes), batch_size):
            batch_nodes = ordered_nodes[batch_idx:batch_idx + batch_size]
            if len(all_plans) >= self.max_items:
                logging.info(f"Reached max_items limit ({self.max_items}), stopping planning")
                break

            nodes_info = []
            for node in batch_nodes:
                nodes_info.append({
                    "name": node.name,
                    "description": node.description,
                    "dependencies": requirement_dag.dependencies(node.name),
                })

            # Include context about already planned tasks
            previous_tasks_context = ""
            if all_plans:
                previous_tasks_context = f"\n\nAlready planned tasks ({len(all_plans)}):\n"
                previous_tasks_context += "\n".join(f"- {task['name']}" for task in all_plans[-10:])

            remaining_slots = self.max_items - len(all_plans)
            prompt = f"""You are a software planning expert. Continue creating an execution plan from the requirement DAG.

DAG Summary:
- Total nodes: {dag_summary['node_count']}
- Processing batch: {batch_idx // batch_size + 1} (nodes {batch_idx + 1}-{min(batch_idx + batch_size, len(ordered_nodes))})
- Remaining slots in plan: {remaining_slots}{previous_tasks_context}

Current Batch Nodes (ordered by dependencies):
{chr(10).join(f"{i + 1}. {n['name']}: {n['description'][:100]}..." if len(n['description']) > 100 else f"{i + 1}. {n['name']}: {n['description']}" for i, n in enumerate(nodes_info))}

Task: Select up to {min(remaining_slots, len(nodes_info))} most important tasks from this batch. Consider:
1. Dependencies on already planned tasks
2. Priority based on topological order
3. Impact on downstream tasks

Return ONLY a JSON array of task objects with: name, description, dependencies, rationale."""

            try:
                response = self.llm_client.call_json([
                    {"role": "system",
                     "content": "You are an expert software project planner. Always return valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ])

                batch_plan = self._extract_plan_from_response(response)

                # Add to overall plan with correct order
                for task in batch_plan:
                    if task['name'] not in processed_nodes:
                        task['order'] = len(all_plans)
                        all_plans.append(task)
                        processed_nodes.add(task['name'])

                        if len(all_plans) >= self.max_items:
                            break

                logging.info(
                    f"Batch {batch_idx // batch_size + 1}: added {len(batch_plan)} tasks, total: {len(all_plans)}")

            except Exception as e:
                logging.warning(f"Batch {batch_idx // batch_size + 1} planning failed: {e}, continuing with next batch")
                continue

        self._enrich_plan(all_plans)
        logging.info(f"Multi-round planning completed: {len(all_plans)} tasks from {len(ordered_nodes)} nodes")
        return all_plans if all_plans else self._fallback_plan_from_dag(requirement_dag, "")

    def _extract_plan_from_response(self, response) -> List[Dict[str, object]]:
        """Extract plan from LLM response."""
        if isinstance(response, list):
            return response
        elif isinstance(response, dict) and "plan" in response:
            return response["plan"]
        elif isinstance(response, dict) and "tasks" in response:
            return response["tasks"]
        elif isinstance(response, dict):
            return [response]
        return []

    def _enrich_plan(self, plan: List[Dict[str, object]]) -> None:
        """Add default values to plan tasks."""
        for i, task in enumerate(plan):
            task.setdefault("order", i)
            task.setdefault("status", "pending")
            task.setdefault("dependencies", [])

    def _fallback_plan_from_dag(
            self,
            requirement_dag: RequirementDAG,
            fallback_text: str = "",
    ) -> List[Dict[str, object]]:
        """Fallback to deterministic plan when LLM is unavailable."""
        ordered_nodes = requirement_dag.topological_order() if not requirement_dag.is_empty() else []
        plan: list[dict[str, object]] = []
        if ordered_nodes:
            for order, node in enumerate(ordered_nodes[: self.max_items]):
                plan.append(
                    {
                        "name": node.name,
                        "description": node.description,
                        "order": order,
                        "dependencies": requirement_dag.dependencies(node.name),
                        "status": "pending"
                    }
                )
            return plan
        fallback_items = self.create_requirement_list(fallback_text)
        for order, description in enumerate(fallback_items[: self.max_items]):
            plan.append(
                {
                    "name": f"generated-{order}",
                    "description": description,
                    "order": order,
                    "dependencies": [],
                    "status": "pending",
                }
            )
        return plan
