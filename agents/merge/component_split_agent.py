"""Agent that splits overly broad architecture components into smaller cohesive ones."""

from __future__ import annotations

import copy
import json
import logging
import itertools
from typing import Any, Dict, List, Optional, Tuple

from agents.infra.llm_client import LLMClient

from .component_merge_agent import _dedupe_text_list


class ComponentSplitAgent:
    """Split metric-selected broad components using requirement partition evidence."""

    def __init__(
        self,
        api_config: Dict[str, Any] | None = None,
        output_dir: str = ".",
        *,
        enable_llm_split: bool = False,
        split_trigger_responsibility_count: int = 8,
        split_trigger_subrequirement_count: int = 5,
        split_max_output_components: int = 3,
        split_min_confidence: float = 0.70,
    ) -> None:
        self.api_config = api_config or {}
        self.llm_client = (
            LLMClient(self.api_config, output_dir, agent_name="component_split")
            if self.api_config.get("api_key")
            else None
        )
        self.enable_llm_split = bool(enable_llm_split)
        self.split_trigger_responsibility_count = max(2, int(split_trigger_responsibility_count))
        self.split_trigger_subrequirement_count = max(2, int(split_trigger_subrequirement_count))
        self.split_max_output_components = max(2, int(split_max_output_components))
        self.split_min_confidence = float(split_min_confidence)

    def split_architecture_components(
        self,
        parent_task: str,
        architecture: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        components = architecture.get("components", [])
        if not isinstance(components, list):
            components = []
        split_components, split_report = self._split_components_if_needed(parent_task, components)
        arch_copy = copy.deepcopy(architecture)
        arch_copy["components"] = split_components
        arch_copy["component_count"] = len(split_components)
        return arch_copy, split_report

    def _split_components_if_needed(
        self,
        parent_task: str,
        components: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        report: Dict[str, Any] = {
            "enabled": True,
            "input_count": len(components),
            "output_count": len(components),
            "triggered_count": 0,
            "split_groups": [],
            "stats": {
                "split_group_count": 0,
                "split_component_count": 0,
            },
        }
        updated: List[Dict[str, Any]] = []
        for idx, comp in enumerate(components, start=1):
            if not isinstance(comp, dict):
                updated.append(comp)
                continue
            if not self._should_split_component(comp):
                updated.append(comp)
                continue

            report["triggered_count"] += 1
            comp_name = str(comp.get("name", "")).strip() or f"Component{idx}"
            split_components, split_detail = self._split_component_by_metrics(parent_task, comp, idx)

            if split_components:
                updated.extend(split_components)
            else:
                updated.append(comp)
            report["split_groups"].append(split_detail)

        report["output_count"] = len(updated)
        report["stats"]["split_group_count"] = sum(
            1 for row in report["split_groups"]
            if isinstance(row, dict) and row.get("decision") == "split"
        )
        report["stats"]["split_component_count"] = max(0, len(updated) - len(components))
        return updated, report

    def _should_split_component(self, comp: Dict[str, Any]) -> bool:
        recommended_action = str(
            comp.get("recommended_action")
            or comp.get("action_hint")
            or comp.get("suggested_action")
            or ""
        ).strip().lower()
        return recommended_action == "split"

    def _split_component_by_metrics(
        self,
        parent_task: str,
        component: Dict[str, Any],
        source_index: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        source_name = str(component.get("name", "")).strip() or f"Component{source_index}"
        responsibilities = _dedupe_text_list(
            component.get("responsibilities", [])
            if isinstance(component.get("responsibilities", []), list)
            else []
        )
        serves_subrequirements = _dedupe_text_list(
            component.get("serves_subrequirements", [])
            if isinstance(component.get("serves_subrequirements", []), list)
            else []
        )
        if len(serves_subrequirements) < 2:
            return [], {
                "component_name": source_name,
                "decision": "keep",
                "reason": "metric_split_requires_multiple_subrequirements",
                "confidence": 1.0,
                "source_component_index": source_index,
            }

        partition_groups = self._partition_subrequirements(component, serves_subrequirements)
        component_with_evidence = dict(component)
        component_with_evidence["split_partition_groups"] = partition_groups
        if self.enable_llm_split and self.llm_client is not None:
            return self._split_component_with_llm(
                parent_task,
                component_with_evidence,
                source_index,
            )

        raw_split = []
        for idx, group in enumerate(partition_groups, start=1):
            if not group:
                continue
            raw_split.append(
                {
                    "name": f"{source_name} Part {idx}",
                    "responsibilities": responsibilities or [f"Implement {item}" for item in group],
                    "serves_subrequirements": group,
                }
            )
        normalized = self._validate_split_components(
            source_component=component,
            raw_split_components=raw_split,
        )
        if len(normalized) < 2:
            return [], {
                "component_name": source_name,
                "decision": "keep",
                "reason": "metric_split_validation_failed",
                "confidence": 1.0,
                "source_component_index": source_index,
            }
        return normalized, {
            "component_name": source_name,
            "decision": "split",
            "reason": "Metric split accepted from requirement partition evidence.",
            "confidence": 1.0,
            "source_component_index": source_index,
            "partition_groups": partition_groups,
            "split_into": [str(item.get("name", "")).strip() for item in normalized],
        }

    def _partition_subrequirements(
        self,
        component: Dict[str, Any],
        serves_subrequirements: List[str],
    ) -> List[List[str]]:
        max_split_count = min(self.split_max_output_components, max(2, min(3, len(serves_subrequirements))))
        evidence = component.get("split_partition_evidence", {})
        induced_edges = evidence.get("induced_edges", []) if isinstance(evidence, dict) else []
        edges = set()
        for edge in induced_edges if isinstance(induced_edges, list) else []:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if source and target and source in serves_subrequirements and target in serves_subrequirements and source != target:
                edges.add(tuple(sorted((source, target))))
        if not edges:
            return self._round_robin_groups(serves_subrequirements, max_split_count)

        best_groups: Optional[List[List[str]]] = None
        best_score: Optional[Tuple[int, int, int, Tuple[Tuple[str, ...], ...]]] = None
        for split_count in range(2, max_split_count + 1):
            assignments = itertools.product(range(split_count), repeat=len(serves_subrequirements))
            for assignment in assignments:
                if len(set(assignment)) < split_count:
                    continue
                groups = [[] for _ in range(split_count)]
                for subreq, group_idx in zip(serves_subrequirements, assignment):
                    groups[group_idx].append(subreq)
                canonical = tuple(sorted(tuple(group) for group in groups if group))
                group_by_subreq = {
                    subreq: group_idx
                    for group_idx, group in enumerate(groups)
                    for subreq in group
                }
                cut_edges = sum(
                    1 for left, right in edges
                    if group_by_subreq.get(left) != group_by_subreq.get(right)
                )
                size_imbalance = max(len(group) for group in groups) - min(len(group) for group in groups)
                group_count_penalty = len(groups)
                score = (cut_edges, group_count_penalty, size_imbalance, canonical)
                if best_score is None or score < best_score:
                    best_score = score
                    best_groups = [list(group) for group in groups if group]
        return best_groups or self._round_robin_groups(serves_subrequirements, max_split_count)

    @staticmethod
    def _round_robin_groups(serves_subrequirements: List[str], split_count: int) -> List[List[str]]:
        groups = [[] for _ in range(split_count)]
        for idx, subreq in enumerate(serves_subrequirements):
            groups[idx % split_count].append(subreq)
        return [group for group in groups if group]

    def _split_component_with_llm(
        self,
        parent_task: str,
        component: Dict[str, Any],
        source_index: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        source_name = str(component.get("name", "")).strip() or f"Component{source_index}"
        responsibilities = _dedupe_text_list(
            component.get("responsibilities", [])
            if isinstance(component.get("responsibilities", []), list)
            else []
        )
        serves_subrequirements = _dedupe_text_list(
            component.get("serves_subrequirements", [])
            if isinstance(component.get("serves_subrequirements", []), list)
            else []
        )
        action = str(component.get("recommended_action") or "").strip()
        rationale = str(component.get("recommended_action_rationale") or "").strip()
        action_origin = str(component.get("recommended_action_origin") or "").strip()
        partition_evidence = component.get("split_partition_evidence", {})
        partition_groups = component.get("split_partition_groups", [])
        prompt = f"""
You are a senior software architect focused on splitting overly broad architecture components.

Parent requirement: "{parent_task}"
Source component:
{json.dumps({
    "name": source_name,
    "responsibilities": responsibilities,
    "serves_subrequirements": serves_subrequirements,
    "recommended_action": action,
    "recommended_action_rationale": rationale,
    "recommended_action_origin": action_origin,
    "split_partition_evidence": partition_evidence,
    "split_partition_groups": partition_groups,
}, ensure_ascii=False, indent=2)}

Task:
Split this component into smaller, cohesive components using the provided partition evidence.

Additional signal:
- `recommended_action` and `recommended_action_rationale` come from an upstream structural candidate generator.
- `split_partition_evidence` contains the induced requirement-level subgraph for the component.
- `split_partition_groups` are graph-partitioning groups obtained from a minimum-cut objective over that induced subgraph.
- Treat the partition groups as structural evidence for the split, but rewrite component names, responsibilities, and interface assumptions so each child is a coherent implementation responsibility.

Be conservative. Keep the component as-is by default.
Split only when the component clearly mixes multiple stable module boundaries or distinct responsibilities that should be owned and evolved separately.
Do NOT split into tiny helpers, utility buckets, or thin wrappers. Prefer the most compact set of coherent child components justified by the partition evidence.

Return JSON only:
{{
  "reason": "short explanation",
  "confidence": 0.0,
  "split_components": [
    {{
      "name": "SubComponentName",
      "responsibilities": ["..."],
      "serves_subrequirements": ["..."]
    }}
  ]
}}

`confidence` must be a JSON number between 0 and 1.
Valid examples: 0.70, 0.78, 0.91
Do NOT use words such as high/medium/low or percentages.

Decision criteria:
1) Use the partition groups as the starting point for assigning served sub-requirements to child components.
2) Split only when the partition evidence corresponds to clearly distinct clusters with low overlap.
3) Reject splitting when the responsibilities mostly describe one workflow, one public API surface, or one stable subsystem with internal helper steps.
4) Reject splitting when the result would create helper-like, adapter-only, metadata-only, or orchestration-only fragments.
5) Each proposed split component must still be a meaningful module that could plausibly justify its own file and tests.
6) If uncertain, choose "keep".
""".strip()
        response = self.llm_client.call_json(
            [
                {"role": "system", "content": "You are an expert in architecture normalization. Return strict JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=32768,
        )
        if not isinstance(response, dict):
            raise RuntimeError("Component split LLM response must be a JSON object")

        try:
            confidence = float(response.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        reason = str(response.get("reason", "")).strip()
        raw_split = response.get("split_components", [])
        if confidence < self.split_min_confidence or not isinstance(raw_split, list):
            return [], {
                "component_name": source_name,
                "decision": "keep",
                "reason": reason or "below_split_threshold",
                "confidence": confidence,
                "source_component_index": source_index,
            }

        normalized = self._validate_split_components(
            source_component=component,
            raw_split_components=raw_split,
        )
        if len(normalized) < 2:
            return [], {
                "component_name": source_name,
                "decision": "keep",
                "reason": reason or "invalid_split_payload",
                "confidence": confidence,
                "source_component_index": source_index,
            }

        return normalized, {
            "component_name": source_name,
            "decision": "split",
            "reason": reason,
            "confidence": confidence,
            "source_component_index": source_index,
            "split_into": [str(item.get("name", "")).strip() for item in normalized],
        }

    def _validate_split_components(
        self,
        *,
        source_component: Dict[str, Any],
        raw_split_components: List[Any],
    ) -> List[Dict[str, Any]]:
        source_subreq = _dedupe_text_list(
            source_component.get("serves_subrequirements", [])
            if isinstance(source_component.get("serves_subrequirements", []), list)
            else []
        )
        normalized: List[Dict[str, Any]] = []
        seen_names = set()

        for row in raw_split_components[: self.split_max_output_components]:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            responsibilities = _dedupe_text_list(
                row.get("responsibilities", [])
                if isinstance(row.get("responsibilities", []), list)
                else []
            )
            if not responsibilities:
                continue
            serves_subreq = _dedupe_text_list(
                row.get("serves_subrequirements", [])
                if isinstance(row.get("serves_subrequirements", []), list)
                else []
            )
            filtered_subreq = [sub for sub in serves_subreq if sub in source_subreq]
            if not filtered_subreq:
                filtered_subreq = list(source_subreq)

            payload = copy.deepcopy(source_component)
            payload["name"] = name
            payload["responsibilities"] = responsibilities
            payload["serves_subrequirements"] = filtered_subreq
            payload["split_from_name"] = str(source_component.get("name", "")).strip()
            payload["split_from_ids"] = list(source_component.get("merged_from_ids", [])) or [str(source_component.get("name", "")).strip()]
            payload["split_source_action"] = {
                "component": str(source_component.get("name", "")).strip(),
                "action": str(source_component.get("recommended_action") or "").strip(),
                "rationale": str(source_component.get("recommended_action_rationale") or "").strip(),
            }
            normalized.append(payload)

        if len(normalized) < 2:
            return []
        return normalized


__all__ = ["ComponentSplitAgent"]
