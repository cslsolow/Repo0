"""Agent that merges duplicate architecture components and provides optional embedding diagnostics."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

try:
    from sklearn.cluster import KMeans
    from sklearn.neighbors import LocalOutlierFactor
except Exception:  # pragma: no cover - optional dependency
    KMeans = None  # type: ignore[assignment]
    LocalOutlierFactor = None  # type: ignore[assignment]

from agents.infra.llm_client import LLMClient


_IDENTIFIER_WORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")
_QWEN3_EMBED_LOCAL_PATH = os.environ.get("REPO0_QWEN3_EMBED_LOCAL_PATH", "").strip()
_QWEN3_EMBED_HF_ID = "Qwen/Qwen3-Embedding-0.6B"
_GENERIC_TOKENS = {
    "module",
    "component",
    "system",
    "service",
    "support",
    "manager",
    "handler",
    "engine",
    "core",
    "base",
    "common",
    "utils",
    "utility",
    "helper",
    "framework",
    "pipeline",
    "process",
    "integration",
    "adapter",
}


def _humanize_identifier(text: str) -> str:
    cleaned = str(text or "").strip().replace("-", "_").replace(".", "_")
    if not cleaned:
        return ""
    parts: List[str] = []
    for chunk in cleaned.split("_"):
        if not chunk:
            continue
        parts.extend(_IDENTIFIER_WORD_RE.findall(chunk))
    return " ".join(part.lower() for part in parts if part).strip()


def _to_token_set(text: str) -> Set[str]:
    return {tok for tok in _humanize_identifier(text).split() if len(tok) >= 2 and tok not in _GENERIC_TOKENS}


def _safe_jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def _dedupe_text_list(values: Sequence[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


@dataclass
class ComponentItem:
    component_id: str
    index: int
    parent_task: str
    name: str
    responsibilities: List[str]
    serves_subrequirements: List[str]
    raw: Dict[str, Any]

    def normalized_name_text(self) -> str:
        return _humanize_identifier(self.name) or self.name.lower()

    def normalized_resp_text(self) -> str:
        joined = " ".join(self.responsibilities)
        return _humanize_identifier(joined) or joined.lower()


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class _QwenEmbedding:
    """Embedding helper aligned with existing project setting."""

    def __init__(self, model_id: Optional[str] = None, batch_size: int = 64) -> None:
        self.model_id = model_id
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._available: Optional[bool] = None
        self.loaded_model_id = ""

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._try_load()
        return bool(self._available)

    def _try_load(self) -> bool:
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except Exception as exc:
            logging.warning("ComponentMergeAgent embedding backend unavailable: %s", exc)
            self.loaded_model_id = ""
            return False

        candidates = [self.model_id] if self.model_id else [c for c in (_QWEN3_EMBED_LOCAL_PATH, _QWEN3_EMBED_HF_ID) if c]
        last_exc: Exception | None = None
        for candidate in candidates:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=False)
                self._model = AutoModel.from_pretrained(candidate, trust_remote_code=True)
                self._model.eval()
                self._torch = torch
                self.loaded_model_id = str(candidate)
                logging.info("ComponentMergeAgent embedding model loaded: %s", candidate)
                return True
            except Exception as exc:
                last_exc = exc
        logging.warning(
            "ComponentMergeAgent failed to load embedding model candidates %s: %s",
            candidates,
            last_exc,
        )
        self.loaded_model_id = ""
        return False

    def encode(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        if np is None:
            return None
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if not self.available or self._model is None or self._tokenizer is None or self._torch is None:
            return None

        chunks: List[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                tokenized = self._tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                outputs = self._model(**tokenized)
                emb = outputs.last_hidden_state[:, 0].float()
                emb = self._torch.nn.functional.normalize(emb, dim=1).cpu().numpy()
                chunks.append(emb.astype(np.float32))
        return np.vstack(chunks) if chunks else np.zeros((0, 1), dtype=np.float32)


class ComponentMergeAgent:
    """Merge duplicated components with LLM and optional embedding diagnostics."""

    def __init__(
        self,
        api_config: Dict[str, Any] | None = None,
        output_dir: str = ".",
        *,
        enable_embedding_analysis: bool = False,
        emb_threshold: float = 0.78,
        emb_dominance_gap: float = 0.12,
        emb_name_weight: float = 0.5,
        emb_resp_weight: float = 0.35,
        emb_subreq_weight: float = 0.15,
        emb_model_id: Optional[str] = None,
        merge_validate_best: float = 0.78,
        merge_validate_avg: float = 0.70,
        merge_validate_min_pair: float = 0.55,
        merge_validate_dominance_gap: float = 0.18,
        merge_admission_mode: str = "strict",
        merge_relaxed_best: float = 0.30,
        merge_relaxed_avg: float = 0.26,
        merge_relaxed_min_pair: float = 0.20,
        merge_relaxed_dominance_gap: float = 0.28,
    ) -> None:
        self.api_config = api_config or {}
        self.output_dir = Path(output_dir)
        self.relaxed_review_log_path = self.output_dir / "component_merge_relaxed_review.jsonl"
        self.llm_client = (
            LLMClient(self.api_config, output_dir, agent_name="component_merge")
            if self.api_config.get("api_key")
            else None
        )
        self.enable_embedding_analysis = bool(enable_embedding_analysis)
        self.emb_threshold = float(emb_threshold)
        self.emb_dominance_gap = float(emb_dominance_gap)
        self.emb_name_weight = float(emb_name_weight)
        self.emb_resp_weight = float(emb_resp_weight)
        self.emb_subreq_weight = float(emb_subreq_weight)
        self.embedding = _QwenEmbedding(model_id=emb_model_id) if self.enable_embedding_analysis else None
        self.merge_validate_best = float(merge_validate_best)
        self.merge_validate_avg = float(merge_validate_avg)
        self.merge_validate_min_pair = float(merge_validate_min_pair)
        self.merge_validate_dominance_gap = float(merge_validate_dominance_gap)
        self.merge_admission_mode = str(merge_admission_mode or "strict").strip() or "strict"
        self.merge_relaxed_best = float(merge_relaxed_best)
        self.merge_relaxed_avg = float(merge_relaxed_avg)
        self.merge_relaxed_min_pair = float(merge_relaxed_min_pair)
        self.merge_relaxed_dominance_gap = float(merge_relaxed_dominance_gap)
        self._last_validation_rejections: List[Dict[str, Any]] = []

    def merge_architecture_components(
        self,
        parent_task: str,
        architecture: Dict[str, Any],
        *,
        require_split_origin: bool = False,
        include_rule_candidates: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Merge duplicated components in one architecture and return updated architecture + report."""
        self._last_validation_rejections = []
        components = architecture.get("components", [])
        if not isinstance(components, list):
            components = []
        component_items = self._normalize_components(parent_task, components)
        if len(component_items) < 2:
            report = {
                "parent_task": parent_task,
                "input_count": len(component_items),
                "output_count": len(component_items),
                "merge_groups": [],
                "applied": False,
                "reason": "not_enough_components",
            }
            arch_copy = copy.deepcopy(architecture)
            arch_copy["component_count"] = len(component_items)
            return arch_copy, report

        explicit_groups = self._merge_groups_from_action_hints(component_items)
        if self.llm_client is not None:
            try:
                generated_groups = self._merge_with_llm(parent_task, component_items)
                if include_rule_candidates:
                    generated_groups.extend(self._merge_with_rules(component_items))
            except Exception as exc:
                logging.warning("Component LLM merge failed for '%s': %s. fallback to rules.", parent_task, exc)
                generated_groups = self._merge_with_rules(component_items)
                merge_groups, rejected_groups = self._validate_merge_groups(
                    explicit_groups + generated_groups,
                    component_items,
                    require_split_origin=require_split_origin,
                )
                self._last_validation_rejections = rejected_groups
            else:
                merge_groups, rejected_groups = self._validate_merge_groups(
                    explicit_groups + generated_groups,
                    component_items,
                    require_split_origin=require_split_origin,
                )
                self._last_validation_rejections = rejected_groups
        else:
            generated_groups = self._merge_with_rules(component_items)
            merge_groups, rejected_groups = self._validate_merge_groups(
                explicit_groups + generated_groups,
                component_items,
                require_split_origin=require_split_origin,
            )
            self._last_validation_rejections = rejected_groups

        merged_components, mapping = self._apply_merge_groups(component_items, merge_groups)
        arch_copy = copy.deepcopy(architecture)
        arch_copy["components"] = merged_components
        arch_copy["component_count"] = len(merged_components)

        report = {
            "parent_task": parent_task,
            "input_count": len(component_items),
            "output_count": len(merged_components),
            "applied": len(merged_components) < len(component_items),
            "merge_groups": merge_groups,
            "validation_rejections": list(self._last_validation_rejections),
            "validation_summary": self._summarize_rejections(self._last_validation_rejections),
            "id_mapping": mapping,
            "stats": {
                "merged_component_count": max(0, len(component_items) - len(merged_components)),
                "accepted_group_count": len(merge_groups),
                "rejected_group_count": len(self._last_validation_rejections),
            },
        }
        return arch_copy, report

    def _merge_groups_from_action_hints(
        self,
        components: List[ComponentItem],
    ) -> List[Dict[str, Any]]:
        if len(components) < 2:
            return []
        by_name = {item.name: item for item in components}
        groups: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str]] = set()
        for item in components:
            action = str(item.raw.get("recommended_action") or "").strip().lower()
            target_name = str(item.raw.get("recommended_target_component") or "").strip()
            if action != "merge" or not target_name:
                continue
            target = by_name.get(target_name)
            if target is None or target.component_id == item.component_id:
                continue
            pair = tuple(sorted((item.component_id, target.component_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            groups.append(
                {
                    "source_ids": [item.component_id, target.component_id],
                    "merged_name": target.name,
                    "merged_responsibilities": _dedupe_text_list(
                        list(item.responsibilities) + list(target.responsibilities)
                    ),
                    "merged_serves_subrequirements": _dedupe_text_list(
                        list(item.serves_subrequirements) + list(target.serves_subrequirements)
                    ),
                    "reasoning": str(item.raw.get("recommended_action_rationale") or "").strip(),
                    "confidence": 1.0,
                    "admission_mode_hint": "metric_judged",
                }
            )
        return groups

    def embedding_diagnostic(
        self,
        parent_task: str,
        architecture: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Produce optional embedding-based merge diagnostics. This output is not applied downstream."""
        components = architecture.get("components", [])
        if not isinstance(components, list):
            components = []
        items = self._normalize_components(parent_task, components)
        if not self.enable_embedding_analysis:
            return {
                "enabled": False,
                "parent_task": parent_task,
                "input_count": len(items),
                "message": "embedding analysis disabled",
            }

        if len(items) < 2:
            return {
                "enabled": True,
                "parent_task": parent_task,
                "input_count": len(items),
                "pair_scores": [],
                "clusters_raw": [],
                "clusters_after_policy": [],
                "message": "not_enough_components",
            }

        name_texts = [item.normalized_name_text() for item in items]
        resp_texts = [item.normalized_resp_text() for item in items]
        name_embeddings = self.embedding.encode(name_texts) if self.embedding else None
        resp_embeddings = self.embedding.encode(resp_texts) if self.embedding else None

        pair_scores: List[Dict[str, Any]] = []
        threshold_edges: List[Tuple[int, int, float]] = []
        for i, j in combinations(range(len(items)), 2):
            score_detail = self._weighted_pair_score(
                items[i],
                items[j],
                i,
                j,
                name_embeddings,
                resp_embeddings,
            )
            pair_scores.append(score_detail)
            total = float(score_detail["weighted_score"])
            if total >= self.emb_threshold:
                threshold_edges.append((i, j, total))

        clusters_raw = self._connected_components(len(items), threshold_edges)
        clusters_after = self._apply_dominance_policy(items, threshold_edges, clusters_raw)
        gt_sub_requirements = architecture.get("sub_requirements", [])
        gt_leaf_payload = self._normalize_gt_leaves(gt_sub_requirements)
        pred_leaf_payload = [
            {
                "component_id": item.component_id,
                "name": item.name,
                "text": f"{item.name}. {'; '.join(item.responsibilities)}",
            }
            for item in items
        ]
        gt_cluster_assignment = self._assign_predicted_to_gt_clusters(
            gt_leaves=gt_leaf_payload,
            predicted_leaves=pred_leaf_payload,
        )
        return {
            "enabled": True,
            "parent_task": parent_task,
            "input_count": len(items),
            "embedding_backend": (
                self.embedding.loaded_model_id if self.embedding and self.embedding.available else "lexical_fallback"
            ),
            "weights": {
                "name": self.emb_name_weight,
                "responsibility": self.emb_resp_weight,
                "subrequirements": self.emb_subreq_weight,
            },
            "threshold": self.emb_threshold,
            "dominance_gap": self.emb_dominance_gap,
            "pair_scores": sorted(pair_scores, key=lambda row: float(row["weighted_score"]), reverse=True),
            "clusters_raw": [
                [items[idx].component_id for idx in cluster]
                for cluster in clusters_raw
                if len(cluster) >= 2
            ],
            "clusters_after_policy": [
                [items[idx].component_id for idx in cluster]
                for cluster in clusters_after
                if len(cluster) >= 2
            ],
            "gt_leaf_cluster_assignment": gt_cluster_assignment,
        }

    def _normalize_gt_leaves(self, sub_requirements: Any) -> List[Dict[str, str]]:
        if not isinstance(sub_requirements, list):
            return []
        normalized: List[Dict[str, str]] = []
        for idx, sub in enumerate(sub_requirements):
            if isinstance(sub, dict):
                name = str(sub.get("name", "")).strip() or f"GT_{idx + 1}"
                desc = str(sub.get("description", "")).strip()
            else:
                name = str(sub).strip() or f"GT_{idx + 1}"
                desc = ""
            text = f"{name}. {desc}".strip()
            normalized.append(
                {
                    "gt_id": f"GT{idx + 1}",
                    "name": name,
                    "description": desc,
                    "text": text,
                }
            )
        return normalized

    def _assign_predicted_to_gt_clusters(
        self,
        gt_leaves: List[Dict[str, str]],
        predicted_leaves: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        if not gt_leaves or not predicted_leaves or self.embedding is None:
            return {
                "enabled": False,
                "reason": "missing_gt_or_pred_or_embedding",
                "gt_count": len(gt_leaves),
                "pred_count": len(predicted_leaves),
            }

        if np is None:
            return {
                "enabled": False,
                "reason": "numpy_unavailable",
                "gt_count": len(gt_leaves),
                "pred_count": len(predicted_leaves),
            }

        gt_texts = [row.get("text", "") for row in gt_leaves]
        pred_texts = [row.get("text", "") for row in predicted_leaves]
        gt_emb = self.embedding.encode(gt_texts)
        pred_emb = self.embedding.encode(pred_texts)
        if gt_emb is None or pred_emb is None or gt_emb.size == 0 or pred_emb.size == 0:
            return {
                "enabled": False,
                "reason": "embedding_encode_failed",
                "gt_count": len(gt_leaves),
                "pred_count": len(predicted_leaves),
            }

        outlier_mask = np.zeros(len(predicted_leaves), dtype=bool)
        lof_backend = "disabled"
        if LocalOutlierFactor is not None and len(gt_leaves) >= 3 and len(predicted_leaves) > 0:
            try:
                n_neighbors = min(max(2, len(gt_leaves) - 1), 10)
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    novelty=True,
                    contamination="auto",
                    metric="cosine",
                )
                lof.fit(gt_emb)
                pred_flag = lof.predict(pred_emb)  # 1 inlier, -1 outlier
                outlier_mask = pred_flag == -1
                lof_backend = "sklearn_lof_novelty"
            except Exception as exc:
                logging.warning("ComponentMergeAgent LOF detection failed, fallback to no outlier filtering: %s", exc)
                lof_backend = "failed_fallback"

        inlier_idx = [idx for idx in range(len(predicted_leaves)) if not bool(outlier_mask[idx])]
        outlier_idx = [idx for idx in range(len(predicted_leaves)) if bool(outlier_mask[idx])]

        clusters: Dict[int, List[int]] = {idx: [] for idx in range(len(gt_leaves))}
        assignment_backend = "nearest_center"

        if inlier_idx:
            inlier_emb = pred_emb[inlier_idx]
            if KMeans is not None and len(inlier_idx) >= len(gt_leaves):
                try:
                    km = KMeans(
                        n_clusters=len(gt_leaves),
                        init=gt_emb,
                        n_init=1,
                        random_state=0,
                    )
                    km.fit(inlier_emb)
                    assignment_backend = "kmeans_init_gt_centroids"

                    # Map learned clusters to gt slots by cosine similarity between centers.
                    center_sim = self._cosine_matrix(km.cluster_centers_, gt_emb)
                    cluster_to_gt = self._greedy_center_alignment(center_sim)
                    for local_idx, cluster_idx in enumerate(km.labels_):
                        gt_idx = cluster_to_gt.get(int(cluster_idx), int(cluster_idx) % len(gt_leaves))
                        clusters[int(gt_idx)].append(inlier_idx[local_idx])
                except Exception as exc:
                    logging.warning("ComponentMergeAgent KMeans assignment failed, fallback to nearest GT center: %s", exc)
                    assignment_backend = "kmeans_failed_nearest_center"
                    nearest = self._nearest_center_indices(inlier_emb, gt_emb)
                    for local_idx, gt_idx in enumerate(nearest):
                        clusters[int(gt_idx)].append(inlier_idx[local_idx])
            else:
                nearest = self._nearest_center_indices(inlier_emb, gt_emb)
                for local_idx, gt_idx in enumerate(nearest):
                    clusters[int(gt_idx)].append(inlier_idx[local_idx])

        cluster_rows: List[Dict[str, Any]] = []
        for gt_idx, pred_indices in clusters.items():
            gt_item = gt_leaves[gt_idx]
            members = [
                {
                    "component_id": predicted_leaves[p_idx]["component_id"],
                    "name": predicted_leaves[p_idx]["name"],
                }
                for p_idx in pred_indices
            ]
            cluster_rows.append(
                {
                    "gt_id": gt_item["gt_id"],
                    "gt_name": gt_item["name"],
                    "assigned_predicted": members,
                    "assigned_count": len(members),
                }
            )

        outlier_rows = [
            {
                "component_id": predicted_leaves[idx]["component_id"],
                "name": predicted_leaves[idx]["name"],
            }
            for idx in outlier_idx
        ]
        return {
            "enabled": True,
            "embedding_model": self.embedding.loaded_model_id if self.embedding.available else "lexical_fallback",
            "use_cls_pooling": True,
            "normalization": "l2",
            "lof_backend": lof_backend,
            "assignment_backend": assignment_backend,
            "gt_count": len(gt_leaves),
            "pred_count": len(predicted_leaves),
            "outlier_count": len(outlier_rows),
            "outliers": outlier_rows,
            "clusters": cluster_rows,
        }

    @staticmethod
    def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
        return a_norm @ b_norm.T

    @staticmethod
    def _greedy_center_alignment(sim: np.ndarray) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        used_gt: Set[int] = set()
        if sim.size == 0:
            return mapping
        flat_pairs: List[Tuple[float, int, int]] = []
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                flat_pairs.append((float(sim[i, j]), i, j))
        flat_pairs.sort(key=lambda row: row[0], reverse=True)
        for _, cluster_idx, gt_idx in flat_pairs:
            if cluster_idx in mapping:
                continue
            if gt_idx in used_gt:
                continue
            mapping[cluster_idx] = gt_idx
            used_gt.add(gt_idx)
        return mapping

    @staticmethod
    def _nearest_center_indices(samples: np.ndarray, centers: np.ndarray) -> List[int]:
        if samples.size == 0 or centers.size == 0:
            return []
        sim = ComponentMergeAgent._cosine_matrix(samples, centers)
        return [int(np.argmax(row)) for row in sim]

    def _normalize_components(self, parent_task: str, components: Sequence[Dict[str, Any]]) -> List[ComponentItem]:
        normalized: List[ComponentItem] = []
        for idx, comp in enumerate(components):
            if not isinstance(comp, dict):
                continue
            name = str(comp.get("name", "")).strip()
            if not name:
                name = f"Component{idx + 1}"
            responsibilities = comp.get("responsibilities", [])
            serves_subreq = comp.get("serves_subrequirements", [])
            normalized.append(
                ComponentItem(
                    component_id=f"C{idx + 1}",
                    index=idx,
                    parent_task=parent_task,
                    name=name,
                    responsibilities=_dedupe_text_list(responsibilities if isinstance(responsibilities, list) else []),
                    serves_subrequirements=_dedupe_text_list(serves_subreq if isinstance(serves_subreq, list) else []),
                    raw=dict(comp),
                )
            )
        return normalized

    def _merge_with_llm(self, parent_task: str, components: List[ComponentItem]) -> List[Dict[str, Any]]:
        payload = [
            {
                "id": item.component_id,
                "parent_task": item.parent_task,
                "name": item.name,
                "responsibilities": item.responsibilities,
                "serves_subrequirements": item.serves_subrequirements,
                "recommended_action": str(item.raw.get("recommended_action") or "").strip(),
            }
            for item in components
        ]
        prompt = f"""
You are a senior software architect focused on component normalization.

Task:
Merge duplicated or near-duplicated components for parent requirement "{parent_task}".

Input components (JSON):
{json.dumps(payload, ensure_ascii=False, indent=2)}

Rules:
1) Merge only when two or more components implement essentially the same capability, or when they are thin adjacent slices of the same stable module boundary.
2) Keep components separate when they are only related, but have different owners, interfaces, runtimes, data lifecycles, or change independently.
3) Prefer fewer, broader, clearly ownable components over many tiny helper-like components.
4) Good merge candidates include thin wrappers such as API+orchestrator, registry+metadata, persistence+provenance, plotting+export, validation+preprocessing.
5) Do NOT merge if it would create a vague catch-all module with weak cohesion.
6) IMPORTANT transitive rule:
   If A-B and B-C are similar but A-C is weak, do NOT merge all three.
   Prefer merging the strongest coherent clique only.
7) Each source id can appear in at most one merge group.
8) Keep merged names concise and preserve key responsibilities/subrequirements.
9) `confidence` must be a JSON number between 0 and 1.
10) Valid confidence examples: 0.72, 0.95
11) Do not use words or mixed formats such as high, low, nine, ninety percent, or 0. nine.
12) Do NOT merge any component whose recommended_action is "save" or "revise"; those actions mean the component boundary should remain stable for this round.

Return JSON only:
{{
  "merge_groups": [
    {{
      "source_ids": ["C1", "C3"],
      "merged_name": "CanonicalComponentName",
      "merged_responsibilities": ["..."],
      "merged_serves_subrequirements": ["..."],
      "reasoning": "short reason",
      "confidence": 0.0
    }}
  ]
}}
""".strip()
        response = self.llm_client.call_json(
            [
                {"role": "system", "content": "You are an expert in repository architecture deduplication. Return strict JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=32768,
        )
        if not isinstance(response, dict):
            raise RuntimeError("Component merge LLM response must be a JSON object")
        raw_groups = response.get("merge_groups", [])
        if not isinstance(raw_groups, list):
            raise RuntimeError("Component merge LLM response.merge_groups must be a list")
        valid_groups, rejected_groups = self._validate_merge_groups(raw_groups, components)
        self._last_validation_rejections = rejected_groups
        return valid_groups


    def _merge_with_rules(self, components: List[ComponentItem]) -> List[Dict[str, Any]]:
        if len(components) < 2:
            return []
        pairs: List[Tuple[int, int, float]] = []
        for i, j in combinations(range(len(components)), 2):
            name_sim = self._name_similarity_enhanced(components[i].name, components[j].name)
            resp_sim = self._resp_similarity(components[i], components[j])
            subreq_sim = _safe_jaccard(
                {item.lower() for item in components[i].serves_subrequirements},
                {item.lower() for item in components[j].serves_subrequirements},
            )
            total = 0.55 * name_sim + 0.35 * resp_sim + 0.10 * subreq_sim
            if total >= 0.82:
                pairs.append((i, j, total))
        if not pairs:
            return []

        components_idx = self._connected_components(len(components), pairs)
        merge_groups: List[Dict[str, Any]] = []
        for group in components_idx:
            if len(group) < 2:
                continue
            source_ids = [components[idx].component_id for idx in sorted(group)]
            merge_groups.append(
                {
                    "source_ids": source_ids,
                    "merged_name": components[min(group)].name,
                    "merged_responsibilities": _dedupe_text_list(
                        [resp for idx in group for resp in components[idx].responsibilities]
                    ),
                    "merged_serves_subrequirements": _dedupe_text_list(
                        [sub for idx in group for sub in components[idx].serves_subrequirements]
                    ),
                    "reasoning": "Rule-based high similarity merge.",
                    "confidence": 0.5,
                }
            )
        return merge_groups

    def _validate_merge_groups(
        self,
        raw_groups: List[Dict[str, Any]],
        components: List[ComponentItem],
        *,
        require_split_origin: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        id_to_item = {item.component_id: item for item in components}
        used: Set[str] = set()
        valid_groups: List[Dict[str, Any]] = []
        rejected_groups: List[Dict[str, Any]] = []

        for raw in raw_groups:
            if not isinstance(raw, dict):
                continue
            src_raw = raw.get("source_ids", [])
            if not isinstance(src_raw, list):
                continue
            source_ids = []
            for item in src_raw:
                comp_id = str(item).strip()
                if comp_id in id_to_item and comp_id not in source_ids:
                    source_ids.append(comp_id)
            if len(source_ids) < 2:
                continue
            if require_split_origin and not any(
                str(id_to_item[comp_id].raw.get("split_from_name") or "").strip()
                for comp_id in source_ids
            ):
                continue
            admission_mode_hint = str(raw.get("admission_mode_hint") or "").strip()
            protected_sources = [
                comp_id
                for comp_id in source_ids
                if str(id_to_item[comp_id].raw.get("recommended_action") or "").strip().lower()
                in {"save", "revise"}
            ]
            if protected_sources and admission_mode_hint != "metric_judged":
                rejected_groups.append(
                    {
                        "source_ids": source_ids,
                        "reason": "stable_action_protected",
                        "protected_source_ids": protected_sources,
                    }
                )
                continue
            if any(comp_id in used for comp_id in source_ids):
                rejected_groups.append(
                    {
                        "source_ids": source_ids,
                        "reason": "id_conflict",
                    }
                )
                continue

            score_summary = self._group_score_summary(source_ids, id_to_item)
            admission_mode = "strict"
            dominance_gap = self.merge_validate_dominance_gap
            if admission_mode_hint == "metric_judged":
                admission_mode = "metric_judged"
            elif not self._score_summary_passes(
                score_summary,
                self.merge_validate_best,
                self.merge_validate_avg,
                self.merge_validate_min_pair,
            ):
                review = None
                if self.merge_admission_mode == "llm_review_relaxed" and self._score_summary_passes(
                    score_summary,
                    self.merge_relaxed_best,
                    self.merge_relaxed_avg,
                    self.merge_relaxed_min_pair,
                ):
                    review = self._review_relaxed_merge(raw, source_ids, id_to_item, score_summary)
                    if review.get("approved") is True:
                        admission_mode = "llm_review_relaxed"
                        dominance_gap = self.merge_relaxed_dominance_gap
                    else:
                        rejected_groups.append(
                            {
                                "source_ids": source_ids,
                                "reason": "llm_relaxed_review_rejected",
                                "score_summary": score_summary,
                                "review": review,
                            }
                        )
                        continue
                else:
                    rejected_groups.append(
                        {
                            "source_ids": source_ids,
                            "reason": "low_semantic_similarity",
                            "score_summary": score_summary,
                            "admission_mode": self.merge_admission_mode,
                        }
                    )
                    continue

            # Chain-split guard: if group score spread is large, only keep strongest pair.
            if len(source_ids) >= 3 and (
                score_summary["best_pair_score"] - score_summary["worst_pair_score"]
            ) > dominance_gap:
                strongest_pair = score_summary.get("strongest_pair", [])
                if len(strongest_pair) == 2:
                    source_ids = list(strongest_pair)
                    score_summary = self._group_score_summary(source_ids, id_to_item)
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

            merged_name = str(raw.get("merged_name", "")).strip()
            if not merged_name:
                merged_name = id_to_item[source_ids[0]].name
            merged_resp = raw.get("merged_responsibilities", [])
            merged_sub = raw.get("merged_serves_subrequirements", [])
            if not isinstance(merged_resp, list):
                merged_resp = []
            if not isinstance(merged_sub, list):
                merged_sub = []
            valid_groups.append(
                {
                    "source_ids": source_ids,
                    "merged_name": merged_name,
                    "merged_responsibilities": _dedupe_text_list(merged_resp),
                    "merged_serves_subrequirements": _dedupe_text_list(merged_sub),
                    "reasoning": str(raw.get("reasoning", "")).strip(),
                    "confidence": float(raw.get("confidence", 0.0) or 0.0),
                    "score_summary": score_summary,
                    "admission_mode": admission_mode_hint or admission_mode,
                }
            )

        valid_groups.sort(
            key=lambda grp: min(id_to_item[comp_id].index for comp_id in grp.get("source_ids", []))
        )
        return valid_groups, rejected_groups


    @staticmethod
    def _score_summary_passes(
        score_summary: Dict[str, Any],
        best_threshold: float,
        avg_threshold: float,
        min_pair_threshold: float,
    ) -> bool:
        return (
            float(score_summary.get("best_pair_score", 0.0)) >= float(best_threshold)
            and float(score_summary.get("avg_pair_score", 0.0)) >= float(avg_threshold)
            and float(score_summary.get("worst_pair_score", 0.0)) >= float(min_pair_threshold)
        )

    def _review_relaxed_merge(
        self,
        raw_group: Dict[str, Any],
        source_ids: List[str],
        id_to_item: Dict[str, ComponentItem],
        score_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.llm_client is None:
            review = {"approved": False, "reason": "missing_llm_client"}
            self._append_relaxed_review_jsonl(source_ids, score_summary, {}, review)
            return review
        payload = {
            "candidate_group": raw_group,
            "score_summary": score_summary,
            "relaxed_thresholds": {
                "best_pair_score": self.merge_relaxed_best,
                "avg_pair_score": self.merge_relaxed_avg,
                "worst_pair_score": self.merge_relaxed_min_pair,
            },
            "components": [
                {
                    "id": comp_id,
                    "parent_task": id_to_item[comp_id].parent_task,
                    "name": id_to_item[comp_id].name,
                    "responsibilities": id_to_item[comp_id].responsibilities,
                    "serves_subrequirements": id_to_item[comp_id].serves_subrequirements,
                    "recommended_action": str(id_to_item[comp_id].raw.get("recommended_action") or "").strip(),
                }
                for comp_id in source_ids
            ],
        }
        prompt = f"""
Review whether this relaxed-threshold component merge is safe.

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Approve only if the components have the same core responsibility, no incompatible public interface,
no conflicting lifecycle/runtime ownership, and the merged component will remain cohesive.

Return JSON only:
{{
  "approved": true,
  "same_responsibility": true,
  "interface_conflict": false,
  "behavior_conflict": false,
  "risk": "low",
  "reason": "short reason"
}}
""".strip()
        try:
            response = self.llm_client.call_json(
                [
                    {"role": "system", "content": "You are a strict software architecture merge reviewer. Return strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
                operation_name="component_merge_relaxed_review",
            )
        except Exception as exc:
            review = {"approved": False, "reason": "review_call_failed", "error": str(exc)}
            self._append_relaxed_review_jsonl(source_ids, score_summary, payload, review)
            return review
        if not isinstance(response, dict):
            review = {"approved": False, "reason": "review_response_not_object"}
            self._append_relaxed_review_jsonl(source_ids, score_summary, payload, review)
            return review
        risk = str(response.get("risk") or "").strip().lower()
        approved = (
            response.get("approved") is True
            and response.get("same_responsibility") is True
            and response.get("interface_conflict") is False
            and response.get("behavior_conflict") is False
            and risk in {"low", "medium"}
        )
        response["approved"] = bool(approved)
        self._append_relaxed_review_jsonl(source_ids, score_summary, payload, response)
        return response

    def _append_relaxed_review_jsonl(
        self,
        source_ids: List[str],
        score_summary: Dict[str, Any],
        payload: Dict[str, Any],
        review: Dict[str, Any],
    ) -> None:
        row = {
            "source_ids": source_ids,
            "score_summary": score_summary,
            "admission_mode": "llm_review_relaxed",
            "relaxed_thresholds": {
                "best_pair_score": self.merge_relaxed_best,
                "avg_pair_score": self.merge_relaxed_avg,
                "worst_pair_score": self.merge_relaxed_min_pair,
                "dominance_gap": self.merge_relaxed_dominance_gap,
            },
            "candidate": payload,
            "review": review,
        }
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with self.relaxed_review_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:
            logging.warning("Failed to write relaxed merge review JSONL: %s", exc)

    @staticmethod
    def _summarize_rejections(rejected_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_reason: Dict[str, int] = {}
        for row in rejected_groups:
            reason = str(row.get("reason", "unknown")).strip() or "unknown"
            by_reason[reason] = by_reason.get(reason, 0) + 1
        return {
            "total_rejected": len(rejected_groups),
            "by_reason": by_reason,
        }

    def _group_score_summary(
        self,
        source_ids: List[str],
        id_to_item: Dict[str, ComponentItem],
    ) -> Dict[str, Any]:
        if len(source_ids) < 2:
            return {
                "pair_count": 0,
                "avg_pair_score": 0.0,
                "best_pair_score": 0.0,
                "worst_pair_score": 0.0,
                "strongest_pair": [],
            }

        pair_scores: List[Tuple[float, str, str]] = []
        for left_id, right_id in combinations(source_ids, 2):
            left = id_to_item[left_id]
            right = id_to_item[right_id]
            name_score = self._name_similarity_enhanced(left.name, right.name)
            resp_score = self._resp_similarity(left, right)
            subreq_score = _safe_jaccard(
                {item.lower() for item in left.serves_subrequirements},
                {item.lower() for item in right.serves_subrequirements},
            )
            total = 0.45 * name_score + 0.45 * resp_score + 0.10 * subreq_score
            pair_scores.append((float(total), left_id, right_id))

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

    def _apply_merge_groups(
        self,
        components: List[ComponentItem],
        merge_groups: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        id_to_item = {item.component_id: item for item in components}
        canonical_by_id: Dict[str, str] = {}
        merged_payload_by_canonical: Dict[str, Dict[str, Any]] = {}
        for group in merge_groups:
            source_ids = [comp_id for comp_id in group.get("source_ids", []) if comp_id in id_to_item]
            if len(source_ids) < 2:
                continue
            canonical_id = min(source_ids, key=lambda comp_id: id_to_item[comp_id].index)
            base_items = [id_to_item[comp_id] for comp_id in source_ids]
            merged_resp = _dedupe_text_list(
                list(group.get("merged_responsibilities", []))
                + [resp for item in base_items for resp in item.responsibilities]
            )
            merged_subreq = _dedupe_text_list(
                list(group.get("merged_serves_subrequirements", []))
                + [sub for item in base_items for sub in item.serves_subrequirements]
            )
            merged_name = str(group.get("merged_name", "")).strip() or id_to_item[canonical_id].name
            merged_payload_by_canonical[canonical_id] = {
                "name": merged_name,
                "responsibilities": merged_resp,
                "serves_subrequirements": merged_subreq,
                "merged_from_ids": source_ids,
                "merged_source_actions": [
                    {
                        "component": item.name,
                        "action": str(item.raw.get("recommended_action") or "").strip(),
                        "rationale": str(item.raw.get("recommended_action_rationale") or "").strip(),
                    }
                    for item in base_items
                ],
                "merge_reasoning": str(group.get("reasoning", "")).strip(),
                "merge_confidence": float(group.get("confidence", 0.0) or 0.0),
            }
            for comp_id in source_ids:
                canonical_by_id[comp_id] = canonical_id

        merged_components: List[Dict[str, Any]] = []
        mapping: List[Dict[str, Any]] = []
        for item in sorted(components, key=lambda row: row.index):
            canonical_id = canonical_by_id.get(item.component_id)
            if canonical_id is None:
                raw_copy = dict(item.raw)
                raw_copy.setdefault("merged_from_ids", [item.component_id])
                merged_components.append(raw_copy)
                mapping.append(
                    {
                        "source_id": item.component_id,
                        "source_name": item.name,
                        "canonical_id": item.component_id,
                        "canonical_name": item.name,
                    }
                )
                continue
            if item.component_id != canonical_id:
                mapping.append(
                    {
                        "source_id": item.component_id,
                        "source_name": item.name,
                        "canonical_id": canonical_id,
                        "canonical_name": merged_payload_by_canonical[canonical_id]["name"],
                    }
                )
                continue

            payload = dict(item.raw)
            payload.update(merged_payload_by_canonical[canonical_id])
            merged_components.append(payload)
            mapping.append(
                {
                    "source_id": item.component_id,
                    "source_name": item.name,
                    "canonical_id": canonical_id,
                    "canonical_name": payload["name"],
                }
            )
        return merged_components, mapping

    def _weighted_pair_score(
        self,
        left: ComponentItem,
        right: ComponentItem,
        i: int,
        j: int,
        name_embeddings: Optional[np.ndarray],
        resp_embeddings: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        name_lex = self._name_similarity_enhanced(left.name, right.name)
        name_emb = self._cosine_by_index(name_embeddings, i, j)
        if name_emb is None:
            name_score = name_lex
        else:
            # Improved name signal: embed + lexical hybrid.
            name_score = 0.6 * name_emb + 0.4 * name_lex

        resp_lex = self._resp_similarity(left, right)
        resp_emb = self._cosine_by_index(resp_embeddings, i, j)
        resp_score = resp_lex if resp_emb is None else (0.7 * resp_emb + 0.3 * resp_lex)

        subreq_score = _safe_jaccard(
            {item.lower() for item in left.serves_subrequirements},
            {item.lower() for item in right.serves_subrequirements},
        )
        weighted = (
            self.emb_name_weight * name_score
            + self.emb_resp_weight * resp_score
            + self.emb_subreq_weight * subreq_score
        )
        return {
            "left_id": left.component_id,
            "right_id": right.component_id,
            "left_name": left.name,
            "right_name": right.name,
            "name_score": round(float(name_score), 6),
            "resp_score": round(float(resp_score), 6),
            "subreq_score": round(float(subreq_score), 6),
            "weighted_score": round(float(weighted), 6),
        }

    def _apply_dominance_policy(
        self,
        items: List[ComponentItem],
        edges: List[Tuple[int, int, float]],
        raw_clusters: List[List[int]],
    ) -> List[List[int]]:
        if not edges:
            return []
        edge_map: Dict[Tuple[int, int], float] = {}
        for i, j, score in edges:
            a, b = (i, j) if i < j else (j, i)
            edge_map[(a, b)] = max(edge_map.get((a, b), 0.0), float(score))

        out_clusters: List[List[int]] = []
        for cluster in raw_clusters:
            if len(cluster) < 2:
                continue
            cluster_sorted = sorted(cluster)
            internal_scores: List[float] = []
            for i, j in combinations(cluster_sorted, 2):
                score = edge_map.get((i, j))
                if score is not None:
                    internal_scores.append(float(score))
            if not internal_scores:
                continue
            max_score = max(internal_scores)
            min_score = min(internal_scores)
            if (max_score - min_score) <= self.emb_dominance_gap:
                out_clusters.append(cluster_sorted)
                continue

            high_cut = max_score - self.emb_dominance_gap
            high_edges = [(i, j, s) for i, j, s in edges if i in cluster_sorted and j in cluster_sorted and s >= high_cut]
            high_clusters = self._connected_components(len(items), high_edges, subset=set(cluster_sorted))
            high_clusters = [sorted(group) for group in high_clusters if len(group) >= 2]
            if high_clusters:
                out_clusters.extend(high_clusters)
                continue

            # Fallback: keep only the strongest pair when chained similarities are imbalanced.
            best_pair = None
            best_score = -1.0
            for i, j in combinations(cluster_sorted, 2):
                score = edge_map.get((i, j), -1.0)
                if score > best_score:
                    best_score = score
                    best_pair = [i, j]
            if best_pair is not None and len(best_pair) == 2:
                out_clusters.append(sorted(best_pair))

        # Deduplicate clusters.
        seen: Set[Tuple[int, ...]] = set()
        deduped: List[List[int]] = []
        for cluster in out_clusters:
            key = tuple(sorted(cluster))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(list(key))
        return deduped

    def _connected_components(
        self,
        total_nodes: int,
        edges: List[Tuple[int, int, float]],
        subset: Optional[Set[int]] = None,
    ) -> List[List[int]]:
        uf = _UnionFind(total_nodes)
        for i, j, _ in edges:
            if subset is not None and (i not in subset or j not in subset):
                continue
            uf.union(i, j)
        groups: Dict[int, List[int]] = {}
        nodes = sorted(subset) if subset is not None else list(range(total_nodes))
        for idx in nodes:
            root = uf.find(idx)
            groups.setdefault(root, []).append(idx)
        return [group for group in groups.values() if len(group) >= 2]

    def _name_similarity_enhanced(self, left_name: str, right_name: str) -> float:
        left_norm = _humanize_identifier(left_name)
        right_norm = _humanize_identifier(right_name)
        ratio = SequenceMatcher(a=left_norm, b=right_norm).ratio() if left_norm or right_norm else 0.0
        token_sim = _safe_jaccard(_to_token_set(left_norm), _to_token_set(right_norm))
        # Give lexical tokens more weight to reduce accidental chain merges.
        return float(0.55 * token_sim + 0.45 * ratio)

    def _resp_similarity(self, left: ComponentItem, right: ComponentItem) -> float:
        left_text = " ".join(left.responsibilities)
        right_text = " ".join(right.responsibilities)
        ratio = SequenceMatcher(a=_humanize_identifier(left_text), b=_humanize_identifier(right_text)).ratio()
        token_sim = _safe_jaccard(_to_token_set(left_text), _to_token_set(right_text))
        return float(0.5 * ratio + 0.5 * token_sim)

    @staticmethod
    def _cosine_by_index(embeddings: Optional[np.ndarray], i: int, j: int) -> Optional[float]:
        if np is None or embeddings is None or embeddings.size == 0:
            return None
        return float(np.dot(embeddings[i], embeddings[j]))


__all__ = ["ComponentMergeAgent"]
