"""Strategist agent that picks file operations based on the proposed architecture."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from agents.infra.llm_client import LLMClient
from agents.rqmts.dag import RequirementDAG

# File/Component level operations.
# create/add/delete/refactor are requirement-iteration/DAG-evolution actions;
# component refinement intentionally filters them out here.
_ALLOWED_FILE_ACTIONS = [
    "save",
    "split",
    "merge",
    "revise",
]

# Requirement/DAG level operations  
_ALLOWED_DAG_OPERATIONS = ["add", "split", "merge", "delete", "revise"]


class StrategistAgent:
    """Map architecture metadata to a list of concrete repository actions."""

    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.llm_client = LLMClient(self.api_config, output_dir, agent_name="strategist") if self.api_config.get("api_key") else None

    def choose_actions(self, architecture: Dict[str, object]) -> List[dict[str, str]]:
        """Generate repository actions from architecture using LLM."""
        if not self.llm_client:
            return self._fallback_choose_actions(architecture)
        
        components = architecture.get("components", [])
        requirement_info = architecture.get("requirement", {})
        
        component_lines = []
        for comp in components:
            responsibilities = ", ".join(comp.get("responsibilities", []))
            previous_action = str(comp.get("recommended_action") or "").strip()
            previous_rationale = str(comp.get("recommended_action_rationale") or "").strip()
            previous_feedback = ""
            if previous_action or previous_rationale:
                previous_feedback = f" Previous feedback: action={previous_action or 'none'}"
                if previous_rationale:
                    previous_feedback += f", rationale={previous_rationale}"
                previous_feedback += "."
            component_lines.append(
                f"- {comp.get('name', 'Component')}: {responsibilities}{previous_feedback}"
            )
        components_desc = "\n".join(component_lines)
        
        prompt = f"""You are a software project strategist. Given an architecture design, determine the optimal repository actions.

Requirement: {requirement_info.get('name', 'Unknown')}

Architecture Components:
{components_desc}

Available Actions:
- save: Accept this component as stable for this feedback decision
- split: Break component into smaller modules
- revise: Refactor existing component
- merge: Combine related components
- add: Add new functionality to component
- delete: Remove obsolete component

Action semantics:
- Treat split as a structural refinement signal first: prefer it when one component still mixes multiple domain families that should become distinct files or subpackages.
- Do not recommend split merely because a component is large; use it when semantic boundaries are unclear or when feature families would otherwise collapse into generic packages.
- Do NOT use split to create one module per feature. Prefer shared module families for nearby features that naturally live together.
- Treat revise as a non-structural action: keep the component boundary, but revise its responsibilities or implementation strategy.
- Preserve requirement vocabulary. If the requirement exposes domain families, keep those families visible through action rationales so later planning can retain them.
- In multi-round feedback, consider any Previous feedback shown for each component. You may keep the same action if it is still valid, or revise it based on the refined architecture.

For each component, select the most appropriate action. Return ONLY a JSON array:
[
  {{
    "component": "ComponentName",
    "action": "action_name",
    "rationale": "Brief reason for this action"
  }}
]"""
        
        try:
            response = self.llm_client.call_json([
                {"role": "system", "content": "You are an expert software strategist. Always return valid JSON arrays."},
                {"role": "user", "content": prompt}
            ])
            
            if isinstance(response, list):
                actions_data = response
            elif isinstance(response, dict) and "actions" in response:
                actions_data = response["actions"]
            else:
                actions_data = [response] if isinstance(response, dict) else []
            
            actions: list[dict[str, str]] = []
            for action_data in actions_data:
                component = str(action_data.get("component", "repository"))
                action = str(action_data.get("action", "save"))
                if action not in _ALLOWED_FILE_ACTIONS:
                    action = "save"
                actions.append({
                    "component": component,
                    "action": action,
                    "rationale": action_data.get("rationale", "")
                })
            
            logging.info(f"Successfully chose {len(actions)} actions for architecture components")
            return actions if actions else self._fallback_choose_actions(architecture)
        
        except Exception as e:
            logging.warning(f"LLM action selection failed ({e}), using fallback")
            return self._fallback_choose_actions(architecture)
        
    def choose_dag_operation(
        self,
        new_requirement: Dict[str, Any],
        dag: RequirementDAG,
    ) -> Dict[str, Any]:
        """
        Determine what DAG operation should be performed for a new requirement.
        
        Returns a dict with:
        - tag: one of ["EXISTING", "ADD", "RELATION"]
        - relation_type: one of ["REVISE", "CHILD", "MERGE", "SPLIT", "DELETE"] when tag is RELATION
        - affected_requirements: list of impacted high-level requirements when tag is RELATION
        - reason: explanation
        """
        if not self.llm_client:
            return self._fallback_dag_operation(new_requirement, dag)
        
        existing_reqs = [
            {"name": node.name, "description": node.description}
            for node in list(dag.nodes.values())[:100]  # Limit to avoid context overflow
        ]
        existing_reqs_text = json.dumps(existing_reqs, ensure_ascii=False, indent=2)

        
        prompt = f"""
### Context
You are a strict Requirements Analyst. Your goal is to integrate a "New Requirement" into a DAG.

### Input Data
**New Requirement:**
- Name: {new_requirement.get('name', 'Unknown')}
- Description: {new_requirement.get('description', '')}

**Existing Requirements Candidates:**
{existing_reqs_text}

### Decision Logic (Follow Strictly)

**Step 1: Check for Duplicates**
- Is it exactly the same semantic meaning? -> **EXISTING**

**Step 2: Check for Independence (The "ADD" Test)**
- Ask yourself: *Can this new requirement exist or be implemented even if the existing requirement is deleted?*
- Is it a distinct functional module or feature?
- Is it explicitly a part/subset of any existing requirement? **If YES -> treat as NO for ADD.**
- **Only if both questions are clearly YES -> Tag: ADD**. If unsure, continue to Step 3.

**Step 3: Check for Strict Relationship (Only if Step 2 is NO)**
- **RELATION**: Select this ONLY if the new requirement is logically *dependent* on the existing one.
    - **CHILD**: The new requirement is a specific sub-step, implementation detail, or strict subset of the existing one. *It inherits the context of the parent.*
    - **REVISE**: The new requirement explicitly conflicts with or updates the logic of the existing one.
    - **MERGE**: The new requirement should merge multiple overlapping requirements into one.
    - **SPLIT**: The new requirement shows one existing requirement should be split into smaller ones.
    - **DELETE**: The new requirement makes one or more old requirements obsolete and removable.

### Critical Constraints
- **Do NOT create a relation just because they share keywords.** (e.g., "User Login" and "User Dashboard" are distinct features -> ADD, not CHILD).
- **Moderate Bar for RELATION**: Link them when there is a clear functional dependency or strong design coupling, not merely thematic similarity.
- **Balance ADD and RELATION**: If clearly independent, choose ADD; if it updates/refines or is tightly coupled, choose RELATION. Avoid overusing either tag.

### Output Format (JSON Only)
{{
  "tag": "EXISTING|ADD|RELATION",
  "relation_type": "REVISE|CHILD|MERGE|SPLIT|DELETE|null",
  "affected_requirements": ["name_string"],
  "reason": "Step 2 analysis: Is it independent? Why? If Step 3 used, explain the relationship decision.",
  "confidence": 0.0-1.0
}}
"""
        
        try:
            response = self.llm_client.call_json([
                {"role": "user", "content": prompt}
            ], temperature=0.0)
            
            if isinstance(response, dict):
                logging.debug(
                    "LLM returned response for Strategist:\n%s",
                    json.dumps(response, indent=2, sort_keys=True),
                )
                return self._normalize_dag_decision(response)
            
            logging.error(
                "LLM returned invalid response for Strategist:\n%s",
                json.dumps(response, indent=2, sort_keys=True),
            )
            return self._fallback_dag_operation(new_requirement, dag)
        
        except Exception:
            return self._fallback_dag_operation(new_requirement, dag)
    
    def _fallback_dag_operation(
        self,
        new_requirement: Dict[str, Any],
        dag: RequirementDAG
    ) -> Dict[str, Any]:
        """Fallback heuristic for DAG operations."""
        req_name = new_requirement.get("name", "")
        matches: list[str] = []
        
        # Check for exact match
        if req_name in dag.nodes:
            return self._normalize_dag_decision({
                "tag": "RELATION",
                "relation_type": "REVISE",
                "operation": "revise",
                "target": req_name,
                "affected_requirements": [req_name],
                "reason": "Exact name match",
                "confidence": 0.7,
            })
        
        # Check for partial matches
        for existing_name in dag.nodes.keys():
            if req_name.lower() in existing_name.lower() or existing_name.lower() in req_name.lower():
                matches.append(existing_name)

        if matches:
            if len(matches) == 1:
                return self._normalize_dag_decision({
                    "tag": "RELATION",
                    "relation_type": "REVISE",
                    "operation": "revise",
                    "target": matches[0],
                    "affected_requirements": matches,
                    "reason": "Similar name detected",
                    "confidence": 0.5,
                })
            return self._normalize_dag_decision({
                "tag": "RELATION",
                "relation_type": "MERGE",
                "operation": "merge",
                "targets": matches,
                "affected_requirements": matches,
                "reason": "Multiple similar names detected",
                "confidence": 0.5,
            })
        
        # Default: add
        return self._normalize_dag_decision({
            "tag": "ADD",
            "operation": "add",
            "reason": "No conflicts",
            "confidence": 0.6,
            "suggested_parents": [],
        })

    def _normalize_dag_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure tag-based decision fields are consistent and complete."""
        allowed_tags = {"EXISTING", "ADD", "RELATION"}
        allowed_relation_types = {"REVISE", "CHILD", "MERGE", "SPLIT", "DELETE"}

        tag = str(decision.get("tag") or "").upper()
        relation_type = decision.get("relation_type")
        operation = decision.get("operation")

        # Tolerate common tag variants
        if tag in {"EXIST", "NOOP", "NONE"}:
            tag = "EXISTING"

        if relation_type:
            relation_type = str(relation_type).upper()
            if relation_type not in allowed_relation_types:
                relation_type = None
            else:
                tag = "RELATION"

        if tag == "RELATION" and not relation_type:
            relation_type = "REVISE"

        affected = decision.get("affected_requirements")
        if not isinstance(affected, list):
            affected = []
        affected = [str(item) for item in affected if str(item).strip()]
        affected = list(dict.fromkeys(affected))

        if tag == "RELATION" and not affected:
            if relation_type in {"REVISE", "SPLIT"}:
                target = decision.get("target")
                if target:
                    affected = [target]
            elif relation_type in {"MERGE", "DELETE"}:
                targets = decision.get("targets") or []
                affected = list(targets)

        # Correct obvious tag/operation mismatches
        if tag == "ADD" and (decision.get("target") or decision.get("targets") or relation_type):
            tag = "RELATION"
        if tag == "RELATION" and not (relation_type or affected):
            tag = "ADD"
            relation_type = None

        if operation:
            operation = str(operation).lower()
        else:
            if tag == "ADD":
                operation = "add"
            elif tag == "RELATION":
                operation = {
                    "REVISE": "revise",
                    "MERGE": "merge",
                    "SPLIT": "split",
                    "CHILD": "add",
                    "DELETE": "delete",
                }.get(relation_type, "revise")
            else:
                operation = "add"

        decision["tag"] = tag
        decision["operation"] = operation
        if tag == "RELATION":
            decision["relation_type"] = relation_type
            decision["affected_requirements"] = affected
        else:
            decision.pop("relation_type", None)
            decision.pop("affected_requirements", None)
        return decision
    
    def _fallback_choose_actions(self, architecture: Dict[str, object]) -> List[dict[str, str]]:
        """Fallback to heuristic-based action selection when LLM is unavailable."""
        components = architecture.get("components", [])
        actions: list[dict[str, str]] = []
        for component in components:
            name = str(component.get("name", "component"))
            operation = self._map_component_to_action(name)
            actions.append({"component": name, "action": operation})
        if not actions:
            actions.append({"component": "repository", "action": "save"})
        return actions

    def _map_component_to_action(self, component_name: str) -> str:
        lowered = component_name.lower()
        if "builder" in lowered or "add" in lowered:
            return "add"
        if "planner" in lowered or "router" in lowered:
            return "split"
        if "memory" in lowered or "cache" in lowered:
            return "save"
        if "merge" in lowered:
            return "merge"
        if "revise" in lowered or "refactor" in lowered:
            return "revise"
        if "cleanup" in lowered or "remove" in lowered:
            return "delete"
        return "save"
