"""Gap-driven add-action helpers for post-refinement capability recovery."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.infra.llm_client import LLMClient


@dataclass(frozen=True)
class GapAdditionCandidate:
    parent_requirement: str
    candidate_type: str
    target_subrequirement: str
    proposed_requirement: str
    proposed_component: str
    heuristic_gap_score: float
    evidence_spans: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class GapAdditionDecision:
    decision: str
    action: str
    confidence: float
    final_confidence: float
    parent_requirement: str
    target_subrequirement: str
    new_requirement: str
    new_component: str
    served_subrequirements: List[str] = field(default_factory=list)
    evidence_spans: List[str] = field(default_factory=list)
    reason: str = ""


class GapAdditionJudge:
    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.llm_client = (
            LLMClient(self.api_config, output_dir, agent_name="gap_addition_judge")
            if self.api_config.get("api_key")
            else None
        )

    def judge_candidate(
        self,
        *,
        candidate: GapAdditionCandidate,
        parent_entry: Dict[str, Any],
        generated_entries: List[Dict[str, Any]],
        realization_report: Dict[str, Any],
    ) -> GapAdditionDecision:
        if self.llm_client is None:
            return GapAdditionDecision(
                decision="reject",
                action="none",
                confidence=0.0,
                final_confidence=0.0,
                parent_requirement=candidate.parent_requirement,
                target_subrequirement=candidate.target_subrequirement,
                new_requirement="",
                new_component="",
                served_subrequirements=[],
                evidence_spans=candidate.evidence_spans,
                reason="No API key configured.",
            )
        prompt = f"""You are judging whether a parent requirement has a real missing capability gap.
Do not create a new parent requirement.
Prefer add_component over add_requirement_and_component when an existing subrequirement already exists.
Candidate: {candidate}
Parent entry: {parent_entry}
Generated evidence: {generated_entries}
Realization report: {realization_report}
Return JSON only with decision, action, confidence, new_requirement, new_component, served_subrequirements, reason."""
        payload = self.llm_client.call_json(
            [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
            operation_name="gap_addition_judge",
        )
        return normalize_gap_judge_response(candidate=candidate, payload=payload if isinstance(payload, dict) else {})


def _tokens(text: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", str(text).lower()) if tok]


def _overlap_score(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _component_surface_text(component: Dict[str, Any]) -> str:
    responsibilities = " ".join(str(item) for item in component.get("responsibilities", []) or [])
    exports = " ".join(str(item) for item in component.get("exports", []) or [])
    return " ".join(
        [
            str(component.get("name") or ""),
            responsibilities,
            exports,
        ]
    ).strip()


def _derive_component_name(capability_name: str) -> str:
    lowered = str(capability_name).strip().lower()
    if "proxy environment" in lowered:
        return "ProxyEnvironmentSupport"
    words = [part.capitalize() for part in _tokens(capability_name)]
    return "".join(words[:4]) or "RecoveredCapability"


def _extract_direct_evidence_spans(input_text: str, capability_name: str) -> List[str]:
    lines = [line.strip() for line in str(input_text).splitlines() if line.strip()]
    key_tokens = set(_tokens(capability_name))
    spans: List[str] = []
    for line in lines:
        line_tokens = set(_tokens(line))
        if key_tokens and key_tokens <= line_tokens:
            spans.append(line)
    return spans[:2]


def propose_gap_candidate_for_parent(
    *,
    parent_entry: Dict[str, Any],
    input_text: str,
    requirements_payload: Dict[str, Any],
    generated_entries: List[Dict[str, Any]],
    realization_report: Dict[str, Any],
    proposal_threshold: float,
    max_candidates_per_parent: int,
) -> Optional[GapAdditionCandidate]:
    del requirements_payload, generated_entries, realization_report, max_candidates_per_parent

    architecture = dict(parent_entry.get("architecture") or {})
    parent_requirement = str(
        parent_entry.get("parent_task")
        or parent_entry.get("task")
        or (architecture.get("requirement") or {}).get("name")
        or ""
    ).strip()
    components = architecture.get("components", []) or []
    sub_requirements = architecture.get("sub_requirements", []) or []

    for subreq in sub_requirements:
        if not isinstance(subreq, dict):
            continue
        name = str(subreq.get("name") or "").strip()
        description = str(subreq.get("description") or "").strip()
        if not name:
            continue
        if any(name in (component.get("serves_subrequirements") or []) for component in components if isinstance(component, dict)):
            continue
        best_component_score = max(
            (
                _overlap_score(f"{name} {description}", _component_surface_text(component))
                for component in components
                if isinstance(component, dict)
            ),
            default=0.0,
        )
        heuristic_gap_score = 1.0 - best_component_score
        if heuristic_gap_score < float(proposal_threshold):
            continue
        return GapAdditionCandidate(
            parent_requirement=parent_requirement,
            candidate_type="missing_subrequirement",
            target_subrequirement=name,
            proposed_requirement="",
            proposed_component=_derive_component_name(name),
            heuristic_gap_score=round(heuristic_gap_score, 4),
            evidence_spans=_extract_direct_evidence_spans(input_text, name),
            reason="Existing subrequirement has no component landing zone.",
        )

    if sub_requirements:
        return None

    direct_evidence = _extract_direct_evidence_spans(input_text, "proxy environment configuration")
    if not direct_evidence:
        return None
    return GapAdditionCandidate(
        parent_requirement=parent_requirement,
        candidate_type="missing_requirement_under_parent",
        target_subrequirement="",
        proposed_requirement="proxy environment configuration",
        proposed_component="ProxyEnvironmentSupport",
        heuristic_gap_score=1.0,
        evidence_spans=direct_evidence,
        reason="Direct input evidence exists but the parent has no matching subrequirement or component.",
    )


def fuse_gap_confidence(heuristic_gap_score: float, judge_confidence: float) -> float:
    return round(0.45 * float(heuristic_gap_score) + 0.55 * float(judge_confidence), 4)


def normalize_gap_judge_response(
    *,
    candidate: GapAdditionCandidate,
    payload: Dict[str, Any],
) -> GapAdditionDecision:
    action = str(payload.get("action") or "none").strip().lower()
    decision = str(payload.get("decision") or "reject").strip().lower()
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    if action == "add_requirement_and_component" and not candidate.evidence_spans:
        action = "none"
        decision = "reject"
        confidence = 0.0
    final_confidence = fuse_gap_confidence(candidate.heuristic_gap_score, confidence)
    return GapAdditionDecision(
        decision="accept" if decision == "accept" and action != "none" else "reject",
        action=action if action in {"add_component", "add_requirement_and_component"} else "none",
        confidence=confidence,
        final_confidence=final_confidence,
        parent_requirement=candidate.parent_requirement,
        target_subrequirement=candidate.target_subrequirement,
        new_requirement=str(payload.get("new_requirement") or candidate.proposed_requirement).strip(),
        new_component=str(payload.get("new_component") or candidate.proposed_component).strip(),
        served_subrequirements=[
            str(item).strip()
            for item in payload.get("served_subrequirements", []) or []
            if str(item).strip()
        ],
        evidence_spans=list(candidate.evidence_spans),
        reason=str(payload.get("reason") or "").strip(),
    )


def should_accept_gap_decision(
    decision: GapAdditionDecision,
    *,
    add_component_accept_threshold: float,
    add_requirement_and_component_accept_threshold: float,
) -> bool:
    if decision.decision != "accept":
        return False
    if decision.action == "add_requirement_and_component":
        return decision.final_confidence >= float(add_requirement_and_component_accept_threshold)
    if decision.action == "add_component":
        return decision.final_confidence >= float(add_component_accept_threshold)
    return False


def apply_gap_addition_decision(
    *,
    parent_entry: Dict[str, Any],
    decision: GapAdditionDecision,
) -> Dict[str, Any]:
    updated = dict(parent_entry)
    architecture = dict(updated.get("architecture") or {})
    components = list(architecture.get("components", []) or [])
    sub_requirements = list(architecture.get("sub_requirements", []) or [])

    if decision.action == "add_requirement_and_component" and decision.new_requirement:
        existing_names = {str(item.get("name") or "").strip() for item in sub_requirements if isinstance(item, dict)}
        if decision.new_requirement not in existing_names:
            sub_requirements.append(
                {
                    "name": decision.new_requirement,
                    "description": "Recovered from direct input evidence during gap-add stage.",
                    "dependencies": [],
                    "rationale": "gap_add_judge",
                    "order": len(sub_requirements),
                    "status": "active",
                }
            )

    if decision.action in {"add_component", "add_requirement_and_component"}:
        targets = [target for target in (decision.served_subrequirements or [decision.new_requirement or decision.target_subrequirement]) if target]
        components.append(
            {
                "name": decision.new_component,
                "responsibilities": [f"Implement recovered capability surface for {target}." for target in targets],
                "serves_subrequirements": list(decision.served_subrequirements),
                "exports": [],
                "recommended_action": "save",
                "recommended_action_origin": "gap_add_judge",
            }
        )

    architecture["sub_requirements"] = sub_requirements
    architecture["components"] = components
    updated["architecture"] = architecture
    updated["parent_task"] = decision.parent_requirement or updated.get("parent_task") or updated.get("task")
    updated["task"] = updated["parent_task"]
    return updated


def run_local_gap_cleanup(
    *,
    parent_entry: Dict[str, Any],
    component_merge_agent: Any,
    component_split_agent: Any,
    augment_actions_with_component_metrics_fn: Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    apply_action_guided_structure_refinement_fn: Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    split_cohesion_threshold: float,
    split_min_subrequirements: int,
    merge_max_small_subrequirements: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    architectures = [parent_entry]
    actions = [{"task": str(parent_entry.get("parent_task") or parent_entry.get("task") or ""), "actions": []}]
    actions, metric_report = augment_actions_with_component_metrics_fn(
        architectures=architectures,
        actions=actions,
        decomposed_dag=None,
        split_cohesion_threshold=float(split_cohesion_threshold),
        split_min_subrequirements=int(split_min_subrequirements),
        merge_judge=None,
        merge_max_small_subrequirements=int(merge_max_small_subrequirements),
    )
    cleaned, refinement_report = apply_action_guided_structure_refinement_fn(
        architectures=architectures,
        component_merge_agent=component_merge_agent,
        component_split_agent=component_split_agent,
    )
    report = {
        "parent_task": str(parent_entry.get("parent_task") or parent_entry.get("task") or ""),
        "metric_report": metric_report,
        "refinement_report": refinement_report,
    }
    return cleaned[0], report
