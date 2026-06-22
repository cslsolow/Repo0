"""Plan candidate module families from architecture + actions."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agents.infra.llm_client import LLMClient
from agents.package_root import normalize_python_package_root

_STOPWORDS = {
    "module", "modules", "system", "systems", "component", "components",
    "analysis", "support", "supporting", "service", "services", "manager",
    "managers", "engine", "engines", "layer", "layers", "model", "models",
    "core", "api", "runtime", "integration", "feature", "features",
}
_GENERIC_PACKAGES = {
    "core", "common", "utility", "utils", "runtime", "api", "service", "services",
    "platform", "framework", "integration", "layer", "layers",
}
_MAX_PLANNING_INPUT_EST_TOKENS = 1400
_MAX_COMPONENTS_PER_CHUNK = 3


def _snake(text: str) -> str:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(text or "").strip())
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return normalized


def _slug(text: str, max_tokens: int = 2) -> str:
    tokens = [tok for tok in _snake(text).split("_") if tok and tok not in _STOPWORDS and len(tok) >= 3]
    if not tokens:
        return ""
    return "_".join(tokens[:max_tokens])


def _best_overlap_slug(texts: List[str], candidates: List[str]) -> str:
    best = ""
    best_score = 0.0
    bags = [set(_snake(text).split("_")) for text in texts if text]
    for cand in candidates:
        cand_norm = _snake(cand)
        cand_tokens = {tok for tok in cand_norm.split("_") if tok}
        if not cand_tokens:
            continue
        for bag in bags:
            if not bag:
                continue
            score = len(cand_tokens & bag) / len(cand_tokens | bag)
            if score > best_score:
                best_score = score
                best = cand_norm
    return best


class ModulePlanningAgent:
    """Plan candidate module families without assigning components to paths."""

    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.output_dir = Path(output_dir)
        self.llm_client = (
            LLMClient(self.api_config, output_dir, agent_name="module_planner")
            if self.api_config.get("api_key")
            else None
        )

    def _primary_generated_package_root(self) -> Path | None:
        generated_root = self.output_dir / "generated_code"
        if not generated_root.exists():
            return None

        repo_name = normalize_python_package_root(self.api_config.get("repo", ""), default="")
        if repo_name:
            candidate = generated_root / repo_name
            if candidate.exists() and candidate.is_dir():
                return candidate

        candidates = [
            p for p in generated_root.iterdir()
            if p.is_dir() and p.name != "__pycache__"
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def plan_modules(
        self,
        architectures: List[dict],
        actions: List[dict],
        layout_policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        action_index = self._build_action_index(actions)
        module_families: List[Dict[str, Any]] = []
        canonical_candidates = [str(item) for item in (layout_policy.get("canonical_packages") or []) if str(item).strip()]
        default_package = str(layout_policy.get("default_subpackage") or "core").strip() or "core"

        for arch in architectures if isinstance(architectures, list) else []:
            if not isinstance(arch, dict):
                continue
            parent_task = str(arch.get("parent_task") or arch.get("task") or "").strip()
            architecture = arch.get("architecture", {})
            if not parent_task or not isinstance(architecture, dict):
                continue
            logging.info("Module planning parent: %s", parent_task)
            parent_result = self._plan_parent_modules(
                parent_task=parent_task,
                architecture=architecture,
                action_index=action_index,
                canonical_candidates=canonical_candidates,
                default_package=default_package,
            )
            module_families.extend(parent_result.get("module_families", []))

        return {
            "default_package": default_package,
            "module_families": module_families,
            "stats": {
                "module_family_count": len(module_families),
                "parent_count": len(
                    [arch for arch in architectures if isinstance(arch, dict) and str(arch.get("parent_task") or arch.get("task") or "").strip()]
                ),
            },
        }

    def _plan_parent_modules(
        self,
        *,
        parent_task: str,
        architecture: Dict[str, Any],
        action_index: Dict[str, Dict[str, str]],
        canonical_candidates: List[str],
        default_package: str,
    ) -> Dict[str, Any]:
        components = architecture.get("components", [])
        if not isinstance(components, list):
            return {"module_families": []}
        if self.llm_client:
            try:
                planned = self._plan_parent_modules_with_llm(
                    parent_task=parent_task,
                    architecture=architecture,
                    action_index=action_index,
                    canonical_candidates=canonical_candidates,
                    default_package=default_package,
                )
                if planned.get("module_families"):
                    return planned
            except Exception as exc:
                logging.warning("Module planning LLM failed for '%s': %s", parent_task, exc)
        return self._plan_parent_modules_with_rules(
            parent_task=parent_task,
            architecture=architecture,
            action_index=action_index,
            canonical_candidates=canonical_candidates,
            default_package=default_package,
        )

    def _plan_parent_modules_with_llm(
        self,
        *,
        parent_task: str,
        architecture: Dict[str, Any],
        action_index: Dict[str, Dict[str, str]],
        canonical_candidates: List[str],
        default_package: str,
    ) -> Dict[str, Any]:
        planning_input = self._build_parent_planning_input(
            parent_task=parent_task,
            architecture=architecture,
            action_index=action_index,
            canonical_candidates=canonical_candidates,
            default_package=default_package,
        )
        response = self._run_llm_module_planning_rounds(
            parent_task=parent_task,
            planning_input=planning_input,
        )
        module_families_raw = response.get("module_families", []) if isinstance(response, dict) else []
        module_families: List[Dict[str, Any]] = []
        for row in module_families_raw if isinstance(module_families_raw, list) else []:
            if not isinstance(row, dict):
                continue
            family = _snake(str(row.get("module_family", "")).strip())
            package_subpath = self._normalize_package_subpath(str(row.get("package_subpath", "")).strip(), default_package)
            if not family:
                continue
            module_families.append(
                {
                    "parent_task": parent_task,
                    "module_family": family,
                    "covers_features": row.get("covers_features", []),
                    "components": row.get("components", []),
                    "package_subpath": package_subpath,
                    "rationale": str(row.get("rationale", "")).strip(),
                    "source": "llm",
                }
            )
        return {"module_families": module_families}

    def _build_parent_planning_input(
        self,
        *,
        parent_task: str,
        architecture: Dict[str, Any],
        action_index: Dict[str, Dict[str, str]],
        canonical_candidates: List[str],
        default_package: str,
    ) -> Dict[str, Any]:
        components = architecture.get("components", [])
        features = architecture.get("sub_requirements", [])
        requirement_info = architecture.get("requirement", {}) if isinstance(architecture.get("requirement", {}), dict) else {}
        existing_context = self._collect_existing_package_context()

        feature_payload: List[Dict[str, Any]] = []
        feature_names: List[str] = []
        for feature in features if isinstance(features, list) else []:
            if not isinstance(feature, dict):
                continue
            feature_name = str(feature.get("name", "")).strip()
            if not feature_name:
                continue
            feature_names.append(feature_name)
            feature_payload.append(
                {
                    "name": feature_name,
                    "description": str(feature.get("description", "")).strip(),
                }
            )

        component_payload: List[Dict[str, Any]] = []
        component_names: List[str] = []
        feature_component_links: List[Dict[str, str]] = []
        for comp in components if isinstance(components, list) else []:
            if not isinstance(comp, dict):
                continue
            name = str(comp.get("name", "")).strip()
            if not name:
                continue
            component_names.append(name)
            serves = comp.get("serves_subrequirements", []) if isinstance(comp.get("serves_subrequirements", []), list) else []
            component_payload.append(
                {
                    "name": name,
                    "responsibilities": comp.get("responsibilities", []),
                    "serves_subrequirements": serves,
                    "action": action_index.get(parent_task, {}).get(name, "save"),
                }
            )
            for feature_name in serves:
                feature_component_links.append(
                    {
                        "feature": str(feature_name).strip(),
                        "component": name,
                        "relation": "served_by",
                    }
                )

        component_component_links: List[Dict[str, Any]] = []
        for i, left in enumerate(component_payload):
            left_features = set(left.get("serves_subrequirements", []) or [])
            if not left_features:
                continue
            for right in component_payload[i + 1 :]:
                right_features = set(right.get("serves_subrequirements", []) or [])
                if not right_features:
                    continue
                overlap = sorted(left_features & right_features)
                if overlap:
                    component_component_links.append(
                        {
                            "left_component": left["name"],
                            "right_component": right["name"],
                            "shared_features": overlap,
                            "relation": "feature_overlap",
                        }
                    )

        return {
            "parent_requirement": {
                "name": parent_task,
                "description": str(requirement_info.get("description", "")).strip(),
            },
            "feature_families": feature_payload,
            "components": component_payload,
            "relations": {
                "feature_component_links": feature_component_links,
                "component_component_links": component_component_links,
            },
            "canonical_package_candidates": canonical_candidates or [default_package],
            "default_package": default_package,
            "suggested_module_budget": self._suggest_module_budget(
                feature_count=len(feature_payload),
                component_count=len(component_payload),
            ),
            "existing_package_context": existing_context,
            "component_names": component_names,
            "feature_names": feature_names,
        }

    def _run_llm_module_planning_rounds(
        self,
        *,
        parent_task: str,
        planning_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        drafts = self._run_planning_draft_maybe_chunked(
            parent_task=parent_task,
            planning_input=planning_input,
        )
        if not drafts:
            return {}
        draft: Dict[str, Any]
        if len(drafts) == 1:
            draft = drafts[0]
        else:
            merge_prompt = f"""
You are consolidating candidate module families from multiple planning chunks for one parent requirement.

Global planning input:
{json.dumps({k: v for k, v in planning_input.items() if k not in {"components", "relations", "component_names"}}, ensure_ascii=False, indent=2)}

Chunk candidate drafts:
{json.dumps(drafts, ensure_ascii=False, indent=2)}

Task:
1. Merge overlapping or duplicate candidate module families.
2. Keep the final set small and realistic for a mature Python scientific repository.
3. Preserve subdirectory structure where it improves clarity, e.g. time_series/models, regression/inference, core/optimizers.

Return ONLY JSON:
{{
  "module_families": [
    {{
      "module_family": "time_series_models",
      "covers_features": ["state-space-core", "classical-models"],
      "package_subpath": "time_series/models",
      "rationale": "short explanation"
    }}
  ]
}}
""".strip()
            draft = self.llm_client.call_json(
                [
                    {"role": "system", "content": "You output strict JSON only."},
                    {"role": "user", "content": merge_prompt},
                ],
                temperature=0.0,
                max_tokens=20000,
                operation_name=f"module_planner.merge:{parent_task}",
            )
            draft = draft if isinstance(draft, dict) else {}

        review_prompt = f"""
You are a module planning architect. Infer stable Python module families for one parent requirement.

Planning input:
{json.dumps(planning_input, ensure_ascii=False, indent=2)}

Task:
1. Infer 2-5 stable module families by jointly considering feature families, components, and their explicit relations.
2. Choose package_subpath values that preserve domain semantics.
3. Do NOT assign components to modules yet; only propose candidate module families.

Constraints:
- Do NOT create one module per feature by default.
- Multiple nearby features should share the same module family when they evolve together.
- Prefer realistic package counts and names similar to production scientific Python repositories.
- Prefer package subdirectories when they clarify structure, e.g. time_series/models rather than a flat pile.
- Reuse previously generated good package vocabulary when it remains semantically correct.
- Avoid bare generic buckets like core, common, runtime, api, services when a domain family is identifiable.
- If a generic root is still appropriate, refine it to core/<domain_family> rather than core alone.
- Do not change component ownership; the output is a module plan, not a component revise.

Return ONLY JSON:
{{
  "module_families": [
    {{
      "module_family": "time_series_models",
      "covers_features": ["state-space-core", "classical-models"],
      "package_subpath": "core/time_series_models",
      "rationale": "short explanation"
    }}
  ]
}}
You are reviewing a draft module plan for semantic quality and granularity.

Planning input:
{json.dumps(planning_input, ensure_ascii=False, indent=2)}

Draft plan:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Review goals:
1. Catch over-fragmentation: one feature -> one module is usually wrong.
2. Catch under-segmentation: unrelated domain families collapsed into one generic bucket.
3. Verify feature coverage: important feature families should be represented by some module family.
4. Verify candidate quality: module families should be useful for a later assignment step.

Return ONLY JSON:
{{
  "approved": true,
  "issues": ["short issue"],
  "recommended_changes": ["short change"]
}}
""".strip()
        review = self.llm_client.call_json(
            [
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": review_prompt},
            ],
            temperature=0.0,
            max_tokens=12000,
            operation_name=f"module_planner.review:{parent_task}",
        )

        if isinstance(review, dict) and review.get("approved") is True:
            return draft if isinstance(draft, dict) else {}

        refine_prompt = f"""
You are refining a module plan after review.

Planning input:
{json.dumps(planning_input, ensure_ascii=False, indent=2)}

Previous draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Review feedback:
{json.dumps(review, ensure_ascii=False, indent=2)}

Instructions:
- Keep the strong parts of the previous draft.
- Fix the review issues.
- Prefer fewer, stronger module families over over-fragmented layouts.
- Keep multiple nearby features in the same module family when appropriate.

Return ONLY JSON in the same schema as the draft.
""".strip()
        refined = self.llm_client.call_json(
            [
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": refine_prompt},
            ],
            temperature=0.0,
            max_tokens=32768,
            operation_name=f"module_planner.refine:{parent_task}",
        )
        return refined if isinstance(refined, dict) else (draft if isinstance(draft, dict) else {})

    def _run_planning_draft_maybe_chunked(
        self,
        *,
        parent_task: str,
        planning_input: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        components = planning_input.get("components", []) if isinstance(planning_input.get("components", []), list) else []
        est_tokens = self._estimate_tokens(json.dumps(planning_input, ensure_ascii=False, indent=2))
        if est_tokens <= _MAX_PLANNING_INPUT_EST_TOKENS or len(components) <= _MAX_COMPONENTS_PER_CHUNK:
            draft = self._run_planning_draft_once(parent_task=parent_task, planning_input=planning_input)
            return [draft] if draft else []

        results: List[Dict[str, Any]] = []
        for idx in range(0, len(components), _MAX_COMPONENTS_PER_CHUNK):
            chunk = components[idx: idx + _MAX_COMPONENTS_PER_CHUNK]
            chunk_names = {str(item.get("name", "")).strip() for item in chunk if isinstance(item, dict)}
            chunk_input = dict(planning_input)
            chunk_input["components"] = chunk
            chunk_input["component_names"] = [name for name in planning_input.get("component_names", []) if name in chunk_names]
            relations = planning_input.get("relations", {}) if isinstance(planning_input.get("relations", {}), dict) else {}
            chunk_input["relations"] = {
                "feature_component_links": [
                    row for row in relations.get("feature_component_links", [])
                    if isinstance(row, dict) and str(row.get("component", "")).strip() in chunk_names
                ],
                "component_component_links": [
                    row for row in relations.get("component_component_links", [])
                    if isinstance(row, dict)
                    and str(row.get("left_component", "")).strip() in chunk_names
                    and str(row.get("right_component", "")).strip() in chunk_names
                ],
            }
            draft = self._run_planning_draft_once(
                parent_task=f"{parent_task}#chunk{idx // _MAX_COMPONENTS_PER_CHUNK + 1}",
                planning_input=chunk_input,
            )
            if draft:
                results.append(draft)
        return results

    def _run_planning_draft_once(
        self,
        *,
        parent_task: str,
        planning_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        draft_prompt = f"""
You are a repository module planning architect. Propose candidate module families for one parent requirement.

Planning input:
{json.dumps(planning_input, ensure_ascii=False, indent=2)}

Task:
1. Infer a small set of candidate module families by jointly considering feature families, sub-features, components, and their explicit relations.
2. Choose package_subpath values that preserve domain semantics.
3. Do NOT assign components to modules yet; only propose candidate module families that a later assignment step can use.

Constraints:
- Do NOT create one module per feature by default.
- Multiple nearby features should share the same module family when they evolve together.
- Keep the number of candidate families close to a realistic scientific Python repository.
- Prefer reusing the existing package vocabulary when it is semantically correct.
- Prefer package subdirectories when they clarify structure, e.g. time_series/models, regression/inference, core/optimizers.
- Avoid bare generic buckets like core, common, runtime, api, services when a domain family is identifiable.
- If a generic root is still appropriate, refine it to core/<domain_family> rather than core alone.
- Candidate families that end up with no assigned components later should simply not materialize as directories.

Return ONLY JSON:
{{
  "module_families": [
    {{
      "module_family": "time_series_models",
      "covers_features": ["state-space-core", "classical-models"],
      "package_subpath": "time_series/models",
      "rationale": "short explanation"
    }}
  ]
}}
""".strip()
        draft = self.llm_client.call_json(
            [
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": draft_prompt},
            ],
            temperature=0.0,
            max_tokens=20000,
            operation_name=f"module_planner.draft:{parent_task}",
        )
        return draft if isinstance(draft, dict) else {}

    def _plan_parent_modules_with_rules(
        self,
        *,
        parent_task: str,
        architecture: Dict[str, Any],
        action_index: Dict[str, Dict[str, str]],
        canonical_candidates: List[str],
        default_package: str,
    ) -> Dict[str, Any]:
        module_families_by_name: Dict[str, Dict[str, Any]] = {}
        parent_slug = _slug(parent_task)
        parent_family = _best_overlap_slug([parent_task], canonical_candidates) or parent_slug or "domain"
        for comp in architecture.get("components", []) if isinstance(architecture.get("components", []), list) else []:
            if not isinstance(comp, dict):
                continue
            component_name = str(comp.get("name", "")).strip()
            if not component_name:
                continue
            action = action_index.get(parent_task, {}).get(component_name, "save")
            serves = comp.get("serves_subrequirements", [])
            responsibilities = comp.get("responsibilities", [])
            family_sources: List[str] = []
            if isinstance(serves, list):
                family_sources.extend(str(item) for item in serves)
            if isinstance(responsibilities, list):
                family_sources.extend(str(item) for item in responsibilities[:4])
            family_sources.append(component_name)
            family_sources.append(parent_task)

            family = (
                _best_overlap_slug(family_sources, canonical_candidates)
                or _slug(" ".join(family_sources))
                or parent_family
            )
            family = family or "domain"
            base_package = _best_overlap_slug([component_name, parent_task], canonical_candidates) or default_package
            base_package = _snake(base_package or default_package) or default_package
            if base_package in _GENERIC_PACKAGES:
                package_subpath = f"{base_package}/{family}"
            elif action == "split" and family and family != base_package:
                package_subpath = f"{base_package}/{family}"
            else:
                package_subpath = base_package
            module_family = family or _snake(base_package) or "domain"
            family_entry = module_families_by_name.setdefault(
                module_family,
                {
                    "parent_task": parent_task,
                    "module_family": module_family,
                    "covers_features": [],
                    "package_subpath": self._normalize_package_subpath(package_subpath, default_package),
                    "rationale": f"fallback family for {module_family}",
                    "source": "rules",
                },
            )
            for sub in serves if isinstance(serves, list) else []:
                if sub not in family_entry["covers_features"]:
                    family_entry["covers_features"].append(sub)
        return {"module_families": list(module_families_by_name.values())}

    def _build_action_index(self, actions: List[dict]) -> Dict[str, Dict[str, str]]:
        index: Dict[str, Dict[str, str]] = {}
        for row in actions if isinstance(actions, list) else []:
            if not isinstance(row, dict):
                continue
            parent = str(row.get("parent_task") or row.get("task") or "").strip()
            if not parent:
                continue
            parent_index = index.setdefault(parent, {})
            for item in row.get("actions", []) if isinstance(row.get("actions", []), list) else []:
                if not isinstance(item, dict):
                    continue
                component = str(item.get("component", "")).strip()
                action = str(item.get("action", "")).strip() or "save"
                if component:
                    parent_index[component] = action
        return index

    def _normalize_package_subpath(self, value: str, default_package: str) -> str:
        parts = [_snake(part) for part in str(value or "").replace("\\", "/").split("/") if _snake(part)]
        if not parts:
            parts = [_snake(default_package) or "core"]
        if parts[0] in _GENERIC_PACKAGES and len(parts) == 1:
            parts.append("domain")
        return "/".join(parts[:2])

    def _estimate_tokens(self, text: str) -> int:
        return max(1, round(len(str(text or "")) / 4))

    def _suggest_module_budget(self, *, feature_count: int, component_count: int) -> Dict[str, int]:
        if component_count <= 2:
            return {"min": 1, "max": 2}
        if component_count <= 4:
            return {"min": 1, "max": 3}
        return {"min": 2, "max": 4}

    def _collect_existing_package_context(self) -> Dict[str, Any]:
        top_level_packages: List[str] = []
        sample_subpackages: List[str] = []
        package_root = self._primary_generated_package_root()
        if package_root is not None:
            top_level_packages = sorted(
                [p.name for p in package_root.iterdir() if p.is_dir() and p.name != "__pycache__"]
            )[:20]
            for pkg in top_level_packages[:8]:
                pkg_dir = package_root / pkg
                if pkg_dir.is_dir():
                    sample_subpackages.extend(
                        [f"{pkg}/{child.name}" for child in pkg_dir.iterdir() if child.is_dir() and child.name != "__pycache__"]
                    )
        return {
            "top_level_packages": top_level_packages,
            "sample_subpackages": sample_subpackages[:20],
        }
