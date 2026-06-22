"""Agent for inferring component dependencies and aggregating them into requirement edges."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.infra.llm_client import LLMClient


class DependencyGraphAgent:
    """Infer implementation dependencies between planned components."""

    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.llm_client = (
            LLMClient(self.api_config, output_dir, agent_name="dependency_graph")
            if self.api_config.get("api_key")
            else None
        )

    def infer_component_dependencies(
        self,
        components: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None,
        context: str = "",
        requirement_edges: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Infer component dependency edges using LLM with a deterministic fallback."""
        constraints = constraints or {
            "must_use_only_component_ids": True,
            "disallow_self_dependency": True,
            "allow_same_requirement_edges": False,
        }

        if not self.llm_client:
            return self._fallback_component_dependencies(components, constraints)

        prompt_payload = self._build_prompt_payload(
            components,
            constraints=constraints,
            context=context,
            requirement_edges=requirement_edges or [],
        )
        prompt = self._build_dependency_prompt(prompt_payload)

        try:
            logging.info(
                "DependencyGraphAgent request starting: components=%d soft_requirement_edges=%d constraints_keys=%s context_chars=%d",
                len(components),
                len(requirement_edges or []),
                sorted(list(constraints.keys())),
                len(context or ""),
            )
            response = self.llm_client.call_json(
                [
                    {
                        "role": "system",
                        "content": "You are a rigorous software architecture analyst. Return valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=32768,
                operation_name="dependency_graph",
            )
        except Exception as exc:
            logging.warning("LLM dependency inference failed: %s", exc)
            return self._fallback_component_dependencies(components, constraints)

        normalized = self._normalize_inference_response(response, components, constraints)
        logging.info(
            "DependencyGraphAgent normalized response: cross=%d same=%d uncertain=%d",
            len(normalized.get("cross_requirement_component_edges", [])),
            len(normalized.get("same_requirement_component_edges", [])),
            len(normalized.get("uncertain_edges", [])),
        )
        return normalized

    def prune_component_edges(
        self,
        component_edges: List[Dict[str, Any]],
        components: List[Dict[str, Any]],
        requirement_edges: Optional[List[Dict[str, Any]]] = None,
        min_confidence: float = 0.6,
    ) -> Dict[str, Any]:
        """Validate component edges and annotate whether they align with the planning DAG."""
        allowed_pairs = {
            (edge.get("source"), edge.get("target"))
            for edge in (requirement_edges or [])
            if edge.get("source") and edge.get("target")
        }
        component_map, name_map = self._build_component_maps(components)
        kept: List[Dict[str, Any]] = []
        pruned: List[Dict[str, Any]] = []

        for edge in component_edges:
            source = edge.get("source")
            target = edge.get("target")
            confidence = float(edge.get("confidence", 0.0) or 0.0)
            if confidence < min_confidence:
                pruned.append({**edge, "prune_reason": "low_confidence"})
                continue
            source_id = self._resolve_component_id(source, component_map, name_map)
            target_id = self._resolve_component_id(target, component_map, name_map)
            if not source_id or not target_id:
                pruned.append({**edge, "prune_reason": "unknown_component"})
                continue
            source_req = component_map[source_id].get("requirement_node")
            target_req = component_map[target_id].get("requirement_node")
            if not source_req or not target_req:
                pruned.append({**edge, "prune_reason": "missing_requirement_node"})
                continue
            kept.append(
                {
                    **edge,
                    "source": source_id,
                    "target": target_id,
                    "planning_edge_present": (
                        (source_req, target_req) in allowed_pairs if allowed_pairs else None
                    ),
                }
            )

        return {"edges": kept, "pruned": pruned}

    def aggregate_requirement_dependencies(
        self,
        components: List[Dict[str, Any]],
        component_edges: List[Dict[str, Any]],
        allow_same_requirement: bool = False,
    ) -> Dict[str, Any]:
        """Aggregate component dependency edges into requirement-level dependency edges."""
        component_map, name_map = self._build_component_maps(components)
        aggregated: Dict[Tuple[str, str], Dict[str, Any]] = {}
        skipped: List[Dict[str, Any]] = []

        for edge in component_edges:
            source_id = self._resolve_component_id(edge.get("source"), component_map, name_map)
            target_id = self._resolve_component_id(edge.get("target"), component_map, name_map)
            if not source_id or not target_id:
                skipped.append({"edge": edge, "reason": "unknown component id"})
                continue

            source_req = component_map[source_id].get("requirement_node")
            target_req = component_map[target_id].get("requirement_node")
            if not source_req or not target_req:
                skipped.append({"edge": edge, "reason": "missing requirement_node"})
                continue

            if not allow_same_requirement and source_req == target_req:
                skipped.append({"edge": edge, "reason": "same requirement"})
                continue

            key = (source_req, target_req)
            entry = aggregated.setdefault(
                key,
                {
                    "source": source_req,
                    "target": target_req,
                    "supporting_edges": [],
                    "confidence_sum": 0.0,
                    "planning_edge_present": edge.get("planning_edge_present"),
                },
            )
            entry["supporting_edges"].append(
                {
                    "source": source_id,
                    "target": target_id,
                    "reason": edge.get("reason", ""),
                    "confidence": float(edge.get("confidence", 0.0) or 0.0),
                    "dependency_type": edge.get("dependency_type", "must_have"),
                    "evidence": list(edge.get("evidence", []) or []),
                }
            )
            entry["confidence_sum"] += float(edge.get("confidence", 0.0) or 0.0)
            if entry.get("planning_edge_present") is None:
                entry["planning_edge_present"] = edge.get("planning_edge_present")

        requirement_edges = []
        for entry in aggregated.values():
            support = entry["supporting_edges"]
            count = len(support)
            confidence = entry["confidence_sum"] / count if count else 0.0
            requirement_edges.append(
                {
                    "source": entry["source"],
                    "target": entry["target"],
                    "count": count,
                    "confidence": round(confidence, 4),
                    "planning_edge_present": entry.get("planning_edge_present"),
                    "supporting_edges": support,
                }
            )

        return {
            "edges": requirement_edges,
            "skipped_edges": skipped,
        }

    def build_requirement_dependency_edges(
        self,
        components: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None,
        context: str = "",
        requirement_edges: Optional[List[Dict[str, Any]]] = None,
        min_confidence: float = 0.6,
    ) -> Dict[str, Any]:
        """Run dependency inference and aggregate requirement-level dependencies."""
        component_result = self.infer_component_dependencies(
            components,
            constraints=constraints,
            context=context,
            requirement_edges=requirement_edges,
        )
        validated_cross = self.prune_component_edges(
            component_result.get("cross_requirement_component_edges", []),
            components,
            requirement_edges=requirement_edges,
            min_confidence=min_confidence,
        )
        validated_same = self.prune_component_edges(
            component_result.get("same_requirement_component_edges", []),
            components,
            requirement_edges=requirement_edges,
            min_confidence=min_confidence,
        )
        aggregated = self.aggregate_requirement_dependencies(
            components,
            validated_cross.get("edges", []),
            allow_same_requirement=bool(
                (constraints or {}).get("allow_same_requirement_edges", False)
            ),
        )
        component_edges = validated_cross.get("edges", [])
        same_requirement_edges = validated_same.get("edges", [])
        uncertain_edges = component_result.get("uncertain_edges", [])
        pruned_component_edges = validated_cross.get("pruned", []) + validated_same.get("pruned", [])
        return {
            "edges": aggregated.get("edges", []),
            "skipped_edges": aggregated.get("skipped_edges", []),
            "component_edges": component_edges,
            "same_requirement_component_edges": same_requirement_edges,
            "pruned_component_edges": pruned_component_edges,
            "uncertain_edges": uncertain_edges,
            "unresolved": uncertain_edges,
            "summary": {
                "cross_requirement_component_edges": len(component_edges),
                "same_requirement_component_edges": len(same_requirement_edges),
                "requirement_edges": len(aggregated.get("edges", [])),
                "uncertain_edges": len(uncertain_edges),
                "pruned_component_edges": len(pruned_component_edges),
            },
        }

    def _fallback_component_dependencies(
        self,
        components: List[Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        component_map, name_map = self._build_component_maps(components)
        cross_edges: List[Dict[str, Any]] = []
        same_edges: List[Dict[str, Any]] = []

        for comp in components:
            comp_id = self._component_id(comp)
            if not comp_id:
                continue
            raw_deps = (
                comp.get("known_dependencies")
                or comp.get("dependencies")
                or []
            )
            source_req = comp.get("requirement_node")
            for dep in raw_deps:
                target_id = self._resolve_component_id(dep, component_map, name_map)
                if not target_id:
                    continue
                if constraints.get("disallow_self_dependency") and target_id == comp_id:
                    continue
                target_req = component_map.get(target_id, {}).get("requirement_node")
                edge = {
                    "source": comp_id,
                    "target": target_id,
                    "dependency_type": "declared_dependency",
                    "reason": "declared dependency",
                    "confidence": 0.6,
                    "evidence": [],
                }
                if source_req and target_req and source_req == target_req:
                    same_edges.append(edge)
                else:
                    cross_edges.append(edge)

        return {
            "cross_requirement_component_edges": cross_edges,
            "same_requirement_component_edges": same_edges,
            "uncertain_edges": [],
        }

    def _normalize_inference_response(
        self,
        response: Any,
        components: List[Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        component_map, name_map = self._build_component_maps(components)

        if isinstance(response, dict):
            cross_raw = response.get("cross_requirement_edges", [])
            same_raw = response.get("same_requirement_edges", [])
            uncertain_raw = response.get("uncertain_edges", response.get("unresolved", []))
            if not cross_raw and not same_raw and isinstance(response.get("edges"), list):
                cross_raw = response.get("edges", [])
        elif isinstance(response, list):
            cross_raw = response
            same_raw = []
            uncertain_raw = []
        else:
            cross_raw = []
            same_raw = []
            uncertain_raw = []

        cross_edges: List[Dict[str, Any]] = []
        same_edges: List[Dict[str, Any]] = []
        for raw_bucket, target_bucket in (
            (cross_raw, cross_edges),
            (same_raw, same_edges),
        ):
            for row in raw_bucket if isinstance(raw_bucket, list) else []:
                if not isinstance(row, dict):
                    continue
                source_id = self._resolve_component_id(row.get("source"), component_map, name_map)
                target_id = self._resolve_component_id(row.get("target"), component_map, name_map)
                if not source_id or not target_id:
                    continue
                if constraints.get("disallow_self_dependency") and source_id == target_id:
                    continue
                edge = {
                    "source": source_id,
                    "target": target_id,
                    "dependency_type": str(row.get("dependency_type") or "must_have").strip() or "must_have",
                    "reason": str(row.get("reason") or "").strip(),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "evidence": self._normalize_string_list(row.get("evidence"), limit=4),
                }
                source_req = component_map[source_id].get("requirement_node")
                target_req = component_map[target_id].get("requirement_node")
                if source_req and target_req and source_req == target_req:
                    same_edges.append(edge)
                else:
                    target_bucket.append(edge)

        uncertain_edges: List[Dict[str, Any]] = []
        for row in uncertain_raw if isinstance(uncertain_raw, list) else []:
            if not isinstance(row, dict):
                continue
            uncertain_edges.append(
                {
                    "source_hint": str(row.get("source_hint") or row.get("source") or "").strip(),
                    "target_hint": str(row.get("target_hint") or row.get("target") or "").strip(),
                    "reason": str(row.get("reason") or "").strip(),
                    "evidence": self._normalize_string_list(row.get("evidence"), limit=4),
                }
            )

        return {
            "cross_requirement_component_edges": cross_edges,
            "same_requirement_component_edges": same_edges,
            "uncertain_edges": uncertain_edges,
        }

    def _build_prompt_payload(
        self,
        components: List[Dict[str, Any]],
        *,
        constraints: Dict[str, Any],
        context: str,
        requirement_edges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        component_rows = [self._component_prompt_row(comp) for comp in components]
        soft_requirement_edges = [
            {
                "source": str(edge.get("source") or "").strip(),
                "target": str(edge.get("target") or "").strip(),
            }
            for edge in requirement_edges
            if isinstance(edge, dict) and str(edge.get("source") or "").strip() and str(edge.get("target") or "").strip()
        ]
        return {
            "components": component_rows,
            "constraints": constraints,
            "soft_requirement_edges": soft_requirement_edges[:200],
            "context": str(context or "").strip(),
        }

    def _build_dependency_prompt(self, prompt_payload: Dict[str, Any]) -> str:
        return (
            "You are analyzing implementation dependencies between planned software components.\n\n"
            "Goal:\n"
            "Infer MUST-HAVE implementation dependencies only. An edge source -> target means the source component "
            "cannot be implemented correctly without importing, invoking, consuming, or structurally depending on the target.\n\n"
            "Important rules:\n"
            "1. Cross-requirement edges are allowed and expected when implementation reuse is necessary.\n"
            "2. Same-requirement edges are also valuable, but return them in a separate list.\n"
            "3. The provided requirement DAG edges are SOFT hints, not hard constraints. Do not suppress a necessary implementation edge just because the planning DAG lacks it.\n"
            "4. Prefer omission over weak guesses.\n"
            "5. Do not invent components. Only use component ids from the input.\n"
            "6. Good reasons include API consumption, canonical data/model contract usage, result-object dependencies, configuration/schema ownership, orchestration callbacks, or backend service usage.\n"
            "7. Bad reasons include topical similarity, shared vocabulary, being in the same subsystem, or likely future reuse.\n\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "cross_requirement_edges": [\n'
            '    {"source": "component_id", "target": "component_id", "dependency_type": "api_call|data_contract|result_contract|config_schema|orchestration|backend_service|shared_metadata|declared_dependency", "reason": "why this dependency must exist", "confidence": 0.0, "evidence": ["short evidence 1"]}\n'
            "  ],\n"
            '  "same_requirement_edges": [\n'
            '    {"source": "component_id", "target": "component_id", "dependency_type": "api_call|data_contract|result_contract|config_schema|orchestration|backend_service|shared_metadata|declared_dependency", "reason": "why this dependency must exist", "confidence": 0.0, "evidence": ["short evidence 1"]}\n'
            "  ],\n"
            '  "uncertain_edges": [\n'
            '    {"source_hint": "component id or name", "target_hint": "component id or name", "reason": "why it might exist but is not certain", "evidence": ["short evidence 1"]}\n'
            "  ],\n"
            '  "summary": {"notes": "optional short note"}\n'
            "}\n\n"
            f"INPUT:\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
        )

    def _component_prompt_row(self, component: Dict[str, Any]) -> Dict[str, Any]:
        class_names = self._normalize_string_list(component.get("class_names"), limit=5)
        function_names = []
        for row in component.get("function_signatures", []) if isinstance(component.get("function_signatures"), list) else []:
            if isinstance(row, dict):
                name = str(row.get("name") or "").strip()
                if name:
                    function_names.append(name)
        contracts = class_names + self._normalize_string_list(function_names, limit=6) + self._normalize_string_list(component.get("exports"), limit=6)
        return {
            "component_id": str(self._component_id(component) or "").strip(),
            "component_name": str(component.get("name") or "").strip(),
            "parent_requirement": str(component.get("parent_requirement") or component.get("requirement_node") or "").strip(),
            "responsibilities": self._normalize_string_list(component.get("responsibilities"), limit=6),
            "serves_subrequirements": self._normalize_string_list(component.get("serves_subrequirements"), limit=6),
            "public_contracts": self._normalize_string_list(contracts, limit=8),
            "declared_dependencies": self._normalize_string_list(component.get("dependencies"), limit=8),
            "file_path": str(component.get("file_path") or "").strip(),
            "status": str(component.get("status") or component.get("metadata", {}).get("status") or "").strip(),
        }

    def _normalize_string_list(self, values: Any, limit: int = 6) -> List[str]:
        if not isinstance(values, list):
            values = list(values) if isinstance(values, tuple) else []
        result: List[str] = []
        seen = set()
        for item in values:
            text = str(item or "").strip()
            if not text:
                continue
            text = " ".join(text.split())
            if len(text) > 220:
                text = text[:217].rstrip() + "..."
            if text in seen:
                continue
            seen.add(text)
            result.append(text)
            if len(result) >= limit:
                break
        return result

    def _build_component_maps(
        self, components: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        component_map: Dict[str, Dict[str, Any]] = {}
        name_map: Dict[str, str] = {}
        for comp in components:
            comp_id = self._component_id(comp)
            if not comp_id:
                continue
            component_map[comp_id] = comp
            comp_name = comp.get("name")
            if comp_name:
                name_map[str(comp_name)] = comp_id
        return component_map, name_map

    def _resolve_component_id(
        self,
        reference: Any,
        component_map: Dict[str, Dict[str, Any]],
        name_map: Dict[str, str],
    ) -> Optional[str]:
        if reference is None:
            return None
        ref = str(reference).strip()
        if ref in component_map:
            return ref
        if ref in name_map:
            return name_map[ref]
        return None

    def _component_id(self, component: Dict[str, Any]) -> Optional[str]:
        return (
            component.get("id")
            or component.get("component_id")
            or component.get("name")
        )
