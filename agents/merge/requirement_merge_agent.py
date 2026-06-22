"""Agent that merges redundant requirements into consolidated ones."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Sequence, Set, Tuple

from agents.infra.llm_client import LLMClient

_GENERIC_TOKENS = {
    "support",
    "system",
    "module",
    "component",
    "feature",
    "functionality",
    "capability",
    "workflow",
    "pipeline",
    "platform",
    "framework",
    "tool",
    "tools",
    "interface",
    "integration",
    "management",
    "handling",
    "processing",
    "analysis",
}


@dataclass
class RequirementItem:
    """Normalized requirement entry used by merge logic."""

    req_id: int
    name: str
    description: str


class RequirementMergeAgent:
    """Merge semantically redundant requirements using two-stage LLM-only review."""

    def __init__(
        self,
        api_config: Dict[str, Any] | None = None,
        output_dir: str = ".",
        *,
        merge_validate_best: float = 0.55,
        merge_validate_avg: float = 0.48,
        merge_validate_min_pair: float = 0.30,
        merge_validate_dominance_gap: float = 0.2,
        llm_review_min_confidence: float = 0.75,
    ) -> None:
        self.api_config = api_config or {}
        self.llm_client = (
            LLMClient(self.api_config, output_dir, agent_name="requirement_merge")
            if self.api_config.get("api_key")
            else None
        )
        self.merge_validate_best = float(merge_validate_best)
        self.merge_validate_avg = float(merge_validate_avg)
        self.merge_validate_min_pair = float(merge_validate_min_pair)
        self.merge_validate_dominance_gap = float(merge_validate_dominance_gap)
        self.llm_review_min_confidence = float(llm_review_min_confidence)
        self._last_validation_rejections: List[Dict[str, Any]] = []

    def merge_requirements(
        self,
        requirements_input: Sequence[Dict[str, Any]] | Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge requirements and return merged groups + old->new name mapping."""
        requirements = self._normalize_requirements(requirements_input)
        self._last_validation_rejections = []
        if not requirements:
            return {
                "merge_groups": [],
                "merged_name_mapping": [],
                "requirements_after_merge": [],
                "stats": {
                    "input_count": 0,
                    "merged_group_count": 0,
                    "merged_source_count": 0,
                    "output_count": 0,
                },
                "validation_rejections": [],
            }

        if not self.llm_client:
            raise RuntimeError("Requirement merge requires LLM configuration; fallback is disabled")

        merge_groups = self._merge_with_llm(requirements)
        self._log_merge_groups(requirements, merge_groups, stage="proposal")
        merge_groups, rejected_groups = self._review_merge_groups_with_llm(requirements, merge_groups)
        self._log_merge_groups(requirements, merge_groups, stage="approved")
        self._log_rejected_groups(requirements, rejected_groups)
        self._last_validation_rejections = rejected_groups
        return self._build_output(requirements, merge_groups, rejected_groups)

    def merge_from_file(self, input_json: str | Path, output_json: str | Path | None = None) -> Dict[str, Any]:
        """Load requirements from JSON, merge them, optionally save merge result."""
        input_path = Path(input_json)
        data = json.loads(input_path.read_text(encoding="utf-8"))
        result = self.merge_requirements(data)

        if output_json is not None:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        return result

    def _merge_with_llm(self, requirements: List[RequirementItem]) -> List[Dict[str, Any]]:
        payload = [
            {
                "id": req.req_id,
                "name": req.name,
                "description": req.description,
            }
            for req in requirements
        ]

        prompt = dedent(
            f"""
            You are a Senior Product Analyst specializing in Requirements Engineering and Deduplication.

            ### Task
            Analyze the provided list of requirements and identify items that are functionally redundant or significantly overlapping.
            Merge them into a single, comprehensive requirement.

            ### Input Data (JSON)
            {json.dumps(payload, ensure_ascii=False, indent=2)}

            ### Merging Criteria
            1. Redundancy: Merge if two or more requirements describe the same user goal or system behavior using different wording.
            2. Subsumption: If Requirement A is a subset of Requirement B, merge them into a more complete version.
            3. Keep Separate: Do NOT merge requirements that are merely related (for example, User Login and User Logout).
            4. Conflict Resolution: If overlapping requirements have conflicting details, prioritize the most detailed and technically feasible description.

            ### Output Requirements
            Return ONLY a valid JSON object.
            If no redundancies are found, return:
            {{"merge_groups": []}}

            ### Response Schema
            {{
              "merge_groups": [
                {{
                  "reasoning": "Brief explanation of why these specific IDs are being merged.",
                  "merged_name": "A concise, professional name for the unified requirement.",
                  "merged_description": "A high-quality, comprehensive description incorporating all key details from source requirements.",
                  "source_ids": [1, 2]
                }}
              ]
            }}

            ### Rules
            - Unique Mapping: Each original id can belong to at most one merge group.
            - Minimum Group Size: Each source_ids list must contain at least 2 IDs.
            - Language Consistency: merged_name and merged_description must use the same language as the input.
            """
        ).strip()
        response = self.llm_client.call_json(
            [
                {
                    "role": "system",
                    "content": "You are an expert software requirements analyst. Always return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=32768,
        )

        if not isinstance(response, dict):
            raise RuntimeError("Invalid merge response format: expected object")

        raw_groups = response.get("merge_groups", [])
        if not isinstance(raw_groups, list):
            raise RuntimeError("Invalid merge response format: merge_groups must be a list")

        valid_ids = {req.req_id for req in requirements}
        used_ids: set[int] = set()
        groups: List[Dict[str, Any]] = []

        for group in raw_groups:
            if not isinstance(group, dict):
                continue

            source_ids = group.get("source_ids", [])
            if not isinstance(source_ids, list):
                continue

            unique_ids = []
            for value in source_ids:
                if isinstance(value, int) and value in valid_ids and value not in unique_ids:
                    unique_ids.append(value)

            if len(unique_ids) < 2:
                continue
            if any(src_id in used_ids for src_id in unique_ids):
                continue

            used_ids.update(unique_ids)
            merged_name = str(group.get("merged_name", "")).strip()
            merged_desc = str(group.get("merged_description", "")).strip()
            groups.append(
                {
                    "source_ids": unique_ids,
                    "merged_name": merged_name,
                    "merged_description": merged_desc,
                }
            )

        return groups

    def _review_merge_groups_with_llm(
        self,
        requirements: List[RequirementItem],
        merge_groups: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not merge_groups:
            return [], []

        req_by_id = {req.req_id: req for req in requirements}
        review_payload = []
        for idx, group in enumerate(merge_groups, start=1):
            source_ids = [sid for sid in group.get("source_ids", []) if sid in req_by_id]
            if len(source_ids) < 2:
                continue
            review_payload.append(
                {
                    "group_id": idx,
                    "source_ids": source_ids,
                    "merged_name": str(group.get("merged_name", "")).strip(),
                    "merged_description": str(group.get("merged_description", "")).strip(),
                    "source_requirements": [
                        {
                            "id": sid,
                            "name": req_by_id[sid].name,
                            "description": req_by_id[sid].description,
                        }
                        for sid in source_ids
                    ],
                }
            )

        if not review_payload:
            return [], []

        prompt = dedent(
            f"""
            You are a strict requirements deduplication reviewer.

            Task:
            Review each proposed merge group and decide whether it should be approved.

            Review criteria:
            1. Approve only if all source requirements describe the same capability or one is clearly subsumed by another.
            2. Reject if the items are merely related, adjacent, frequently co-mentioned, or belong to the same subsystem/tooling area but still represent distinct responsibilities.
            3. Reject if one item is infrastructure/platform support and another is a concrete feature implemented on top of it.
            4. Reject if one item is computation/execution and another is interpretation/guidance/reporting built on top of it.
            5. Reject if the proposed merged name/description loses important distinctions or would create a requirement that is too broad to be owned clearly.
            6. Be conservative. If uncertain, reject.

            Approval rule of thumb:
            - Approve: "same capability expressed twice" or "one is a narrower restatement of the other"
            - Reject: "same area, different job"

            Proposed merge groups:
            {json.dumps(review_payload, ensure_ascii=False, indent=2)}

            Return ONLY valid JSON in this schema:
            {{
              "reviews": [
                {{
                  "group_id": 1,
                  "decision": "approve",
                  "confidence": 0.84,
                  "reason": "Why this merge is or is not valid",
                  "suggested_split_groups": [[1, 2]]
                }}
              ]
            }}

            Confidence guidance:
            - Use confidence >= 0.85 only when the overlap is very strong and the merge is clearly safe.
            - Use lower confidence when there is any meaningful scope difference.
            - If you reject a 3+ item group because only part of it should be merged, provide suggested_split_groups containing only the subset(s) that should still be merged.
            - Each suggested split group must contain at least 2 source IDs taken from the original group only.
            - `confidence` must be a JSON number between 0 and 1.
            - Valid examples: 0.72, 0.95
            - Do not use words or mixed formats such as high, low, nine, ninety percent, or 0. nine.
            """
        ).strip()

        response = self.llm_client.call_json(
            [
                {
                    "role": "system",
                    "content": "You are an expert software requirements reviewer. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=32768,
        )

        if not isinstance(response, dict):
            raise RuntimeError("Invalid merge review response format: expected object")

        raw_reviews = response.get("reviews", [])
        if not isinstance(raw_reviews, list):
            raise RuntimeError("Invalid merge review response format: reviews must be a list")

        group_by_id = {idx: group for idx, group in enumerate(merge_groups, start=1)}
        approved_groups: List[Dict[str, Any]] = []
        rejected_groups: List[Dict[str, Any]] = []
        seen_group_ids: Set[int] = set()
        used_source_ids: Set[int] = set()

        for review in raw_reviews:
            if not isinstance(review, dict):
                continue
            group_id = review.get("group_id")
            if not isinstance(group_id, int) or group_id not in group_by_id or group_id in seen_group_ids:
                continue
            seen_group_ids.add(group_id)
            decision = str(review.get("decision", "")).strip().lower()
            try:
                confidence = float(review.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            reason = str(review.get("reason", "")).strip()
            group = dict(group_by_id[group_id])
            source_ids = list(group.get("source_ids", []))
            suggested_split_groups = review.get("suggested_split_groups", [])
            if decision == "approve" and confidence >= self.llm_review_min_confidence:
                if any(src_id in used_source_ids for src_id in source_ids):
                    rejected_groups.append(
                        {
                            "source_ids": source_ids,
                            "reason": "llm_review_id_conflict",
                            "review": {
                                "decision": "reject",
                                "confidence": confidence,
                                "reason": "Approved group overlaps with an already accepted group",
                            },
                        }
                    )
                    continue
                used_source_ids.update(source_ids)
                group["review"] = {
                    "decision": "approve",
                    "confidence": confidence,
                    "reason": reason,
                }
                approved_groups.append(group)
            else:
                salvaged_groups = []
                if isinstance(suggested_split_groups, list):
                    valid_group_source_ids = set(source_ids)
                    for subset in suggested_split_groups:
                        if not isinstance(subset, list):
                            continue
                        normalized_subset = []
                        for value in subset:
                            if (
                                isinstance(value, int)
                                and value in valid_group_source_ids
                                and value not in normalized_subset
                            ):
                                normalized_subset.append(value)
                        if len(normalized_subset) < 2:
                            continue
                        if any(src_id in used_source_ids for src_id in normalized_subset):
                            continue
                        used_source_ids.update(normalized_subset)
                        salvaged_groups.append(
                            {
                                "source_ids": normalized_subset,
                                "merged_name": group.get("merged_name", ""),
                                "merged_description": group.get("merged_description", ""),
                                "review": {
                                    "decision": "approve_via_split",
                                    "confidence": confidence,
                                    "reason": reason,
                                    "suggested_split_groups": suggested_split_groups,
                                },
                            }
                        )
                approved_groups.extend(salvaged_groups)
                rejected_groups.append(
                    {
                        "source_ids": source_ids,
                        "reason": "llm_review_rejected",
                        "review": {
                            "decision": decision or "reject",
                            "confidence": confidence,
                            "reason": reason,
                            "suggested_split_groups": suggested_split_groups if isinstance(suggested_split_groups, list) else [],
                        },
                    }
                )

        for group_id, group in group_by_id.items():
            if group_id in seen_group_ids:
                continue
            rejected_groups.append(
                {
                    "source_ids": list(group.get("source_ids", [])),
                    "reason": "llm_review_missing",
                    "review": {
                        "decision": "reject",
                        "confidence": 0.0,
                        "reason": "No review decision returned by LLM",
                    },
                }
            )

        return approved_groups, rejected_groups

    def _log_merge_groups(
        self,
        requirements: List[RequirementItem],
        merge_groups: List[Dict[str, Any]],
        *,
        stage: str,
    ) -> None:
        req_by_id = {req.req_id: req for req in requirements}
        if not merge_groups:
            logging.info("Requirement merge %s produced 0 LLM merge groups", stage)
            return

        logging.info("Requirement merge %s produced %d LLM merge groups", stage, len(merge_groups))
        for idx, group in enumerate(merge_groups, start=1):
            source_ids = [sid for sid in group.get("source_ids", []) if sid in req_by_id]
            source_names = [req_by_id[sid].name for sid in source_ids]
            review = group.get("review") if isinstance(group.get("review"), dict) else {}
            logging.info(
                "Requirement merge %s group %d: source_ids=%s source_names=%s -> merged_name=%s review=%s",
                stage,
                idx,
                source_ids,
                source_names,
                str(group.get("merged_name", "")).strip(),
                review,
            )

    def _log_rejected_groups(
        self,
        requirements: List[RequirementItem],
        rejected_groups: List[Dict[str, Any]],
    ) -> None:
        req_by_id = {req.req_id: req for req in requirements}
        if not rejected_groups:
            logging.info("Requirement merge review rejected 0 groups")
            return
        logging.info("Requirement merge review rejected %d groups", len(rejected_groups))
        for idx, group in enumerate(rejected_groups, start=1):
            source_ids = [sid for sid in group.get("source_ids", []) if sid in req_by_id]
            source_names = [req_by_id[sid].name for sid in source_ids]
            logging.info(
                "Requirement merge rejected group %d: source_ids=%s source_names=%s reason=%s review=%s",
                idx,
                source_ids,
                source_names,
                group.get("reason"),
                group.get("review"),
            )

    def _validate_merge_groups(
        self,
        raw_groups: List[Dict[str, Any]],
        requirements: List[RequirementItem],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        req_by_id = {req.req_id: req for req in requirements}
        used: Set[int] = set()
        valid_groups: List[Dict[str, Any]] = []
        rejected_groups: List[Dict[str, Any]] = []

        for raw in raw_groups:
            if not isinstance(raw, dict):
                continue
            source_ids_raw = raw.get("source_ids", [])
            if not isinstance(source_ids_raw, list):
                continue

            source_ids: List[int] = []
            for value in source_ids_raw:
                if isinstance(value, int) and value in req_by_id and value not in source_ids:
                    source_ids.append(value)

            if len(source_ids) < 2:
                continue
            if any(req_id in used for req_id in source_ids):
                rejected_groups.append(
                    {
                        "source_ids": source_ids,
                        "reason": "id_conflict",
                    }
                )
                continue

            score_summary = self._group_score_summary(source_ids, req_by_id)
            if (
                score_summary["best_pair_score"] < self.merge_validate_best
                or score_summary["avg_pair_score"] < self.merge_validate_avg
                or score_summary["worst_pair_score"] < self.merge_validate_min_pair
            ):
                rejected_groups.append(
                    {
                        "source_ids": source_ids,
                        "reason": "low_semantic_similarity",
                        "score_summary": score_summary,
                    }
                )
                continue

            if len(source_ids) >= 3 and (
                score_summary["best_pair_score"] - score_summary["worst_pair_score"]
            ) > self.merge_validate_dominance_gap:
                strongest_pair = score_summary.get("strongest_pair", [])
                if len(strongest_pair) == 2 and not any(req_id in used for req_id in strongest_pair):
                    source_ids = [int(strongest_pair[0]), int(strongest_pair[1])]
                    score_summary = self._group_score_summary(source_ids, req_by_id)
                else:
                    rejected_groups.append(
                        {
                            "source_ids": source_ids,
                            "reason": "chain_merge_instability",
                            "score_summary": score_summary,
                        }
                    )
                    continue

            used.update(source_ids)
            group = dict(raw)
            group["source_ids"] = source_ids
            group["score_summary"] = score_summary
            valid_groups.append(group)

        valid_groups.sort(key=lambda row: min(row.get("source_ids", [10**9])))
        return valid_groups, rejected_groups

    def _group_score_summary(
        self,
        source_ids: List[int],
        req_by_id: Dict[int, RequirementItem],
    ) -> Dict[str, Any]:
        if len(source_ids) < 2:
            return {
                "pair_count": 0,
                "avg_pair_score": 0.0,
                "best_pair_score": 0.0,
                "worst_pair_score": 0.0,
                "strongest_pair": [],
            }

        pair_scores: List[Tuple[float, int, int]] = []
        for i in range(len(source_ids)):
            for j in range(i + 1, len(source_ids)):
                left_id = source_ids[i]
                right_id = source_ids[j]
                left = req_by_id[left_id]
                right = req_by_id[right_id]
                score = self._pair_similarity(left, right)
                pair_scores.append((float(score), left_id, right_id))

        pair_scores.sort(key=lambda row: row[0], reverse=True)
        values = [row[0] for row in pair_scores]
        best = values[0] if values else 0.0
        worst = values[-1] if values else 0.0
        avg = float(sum(values) / len(values)) if values else 0.0
        strongest_pair = [pair_scores[0][1], pair_scores[0][2]] if pair_scores else []
        return {
            "pair_count": len(values),
            "avg_pair_score": round(avg, 6),
            "best_pair_score": round(best, 6),
            "worst_pair_score": round(worst, 6),
            "strongest_pair": strongest_pair,
        }

    def _pair_similarity(self, left: RequirementItem, right: RequirementItem) -> float:
        left_name = self._normalize_text(left.name)
        right_name = self._normalize_text(right.name)
        left_desc = self._normalize_text(left.description)
        right_desc = self._normalize_text(right.description)

        name_ratio = self._ratio(left_name, right_name)
        name_jaccard = self._jaccard(left_name, right_name)
        desc_ratio = self._ratio(left_desc, right_desc)
        desc_jaccard = self._jaccard(left_desc, right_desc)
        anchor_overlap = self._content_token_jaccard(left.name, right.name)
        # Prefer requirement titles while keeping description evidence.
        score = (
            0.4 * (0.55 * name_ratio + 0.45 * name_jaccard)
            + 0.45 * (0.55 * desc_ratio + 0.45 * desc_jaccard)
            + 0.15 * anchor_overlap
        )
        return float(score)

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _ratio(left: str, right: str) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return SequenceMatcher(a=left, b=right).ratio()

    @staticmethod
    def _jaccard(left: str, right: str) -> float:
        lt = set(left.split())
        rt = set(right.split())
        if not lt and not rt:
            return 1.0
        if not lt or not rt:
            return 0.0
        return len(lt & rt) / len(lt | rt)

    @staticmethod
    def _normalize_requirements(
        requirements_input: Sequence[Dict[str, Any]] | Dict[str, Any],
    ) -> List[RequirementItem]:
        if isinstance(requirements_input, dict):
            raw = requirements_input.get("requirements", [])
        else:
            raw = requirements_input

        normalized: List[RequirementItem] = []
        for idx, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            description = str(item.get("description", "")).strip()
            normalized.append(RequirementItem(req_id=idx, name=name, description=description))
        return normalized

    @staticmethod
    def _build_output(
        requirements: List[RequirementItem],
        merge_groups: List[Dict[str, Any]],
        validation_rejections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        req_by_id = {req.req_id: req for req in requirements}
        merged_ids: set[int] = set()
        mapping_rows: List[Dict[str, Any]] = []
        output_groups: List[Dict[str, Any]] = []
        score_summaries: List[Dict[str, Any]] = []

        for group in merge_groups:
            source_ids = [sid for sid in group.get("source_ids", []) if sid in req_by_id]
            if len(source_ids) < 2:
                continue

            source_reqs = [req_by_id[sid] for sid in source_ids]
            merged_name = str(group.get("merged_name", "")).strip() or source_reqs[0].name
            merged_desc = str(group.get("merged_description", "")).strip() or source_reqs[0].description

            merged_ids.update(source_ids)
            output_groups.append(
                {
                    "merged_name": merged_name,
                    "merged_description": merged_desc,
                    "merged_from_ids": source_ids,
                    "merged_from_names": [req.name for req in source_reqs],
                }
            )
            if isinstance(group.get("score_summary"), dict):
                score_summaries.append(
                    {
                        "merged_name": merged_name,
                        "source_ids": source_ids,
                        "score_summary": group.get("score_summary"),
                    }
                )

            for req in source_reqs:
                mapping_rows.append(
                    {
                        "source_id": req.req_id,
                        "source_name": req.name,
                        "merged_name": merged_name,
                    }
                )

        kept = [
            {
                "id": req.req_id,
                "name": req.name,
                "description": req.description,
            }
            for req in requirements
            if req.req_id not in merged_ids
        ]

        merged_output = [
            {"name": group["merged_name"], "description": group["merged_description"]}
            for group in output_groups
        ]
        requirements_after_merge = merged_output + [
            {"name": item["name"], "description": item["description"]} for item in kept
        ]

        return {
            "merge_groups": output_groups,
            "merged_name_mapping": mapping_rows,
            "requirements_after_merge": requirements_after_merge,
            "kept_requirements": kept,
            "quality_report": {
                "accepted_groups": score_summaries,
                "accepted_group_count": len(score_summaries),
                "rejected_group_count": len(validation_rejections),
            },
            "validation_rejections": validation_rejections,
            "stats": {
                "input_count": len(requirements),
                "merged_group_count": len(output_groups),
                "merged_source_count": len(merged_ids),
                "output_count": len(requirements_after_merge),
            },
        }

    @staticmethod
    def _content_token_jaccard(left: str, right: str) -> float:
        left_set = {
            tok
            for tok in RequirementMergeAgent._normalize_text(left).split()
            if len(tok) > 2 and tok not in _GENERIC_TOKENS
        }
        right_set = {
            tok
            for tok in RequirementMergeAgent._normalize_text(right).split()
            if len(tok) > 2 and tok not in _GENERIC_TOKENS
        }
        if not left_set and not right_set:
            return 0.0
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / len(left_set | right_set)


__all__ = ["RequirementMergeAgent"]
