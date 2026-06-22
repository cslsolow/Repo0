"""Assign components into candidate module families."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from agents.infra.llm_client import LLMClient
from agents.package_root import normalize_python_package_root

_STOPWORDS = {
    "module", "modules", "system", "systems", "component", "components",
    "analysis", "support", "supporting", "service", "services", "manager",
    "managers", "engine", "engines", "layer", "layers", "model", "models",
    "core", "api", "runtime", "integration", "feature", "features",
}
_MAX_ASSIGNMENT_INPUT_EST_TOKENS = 1400
_MAX_COMPONENTS_PER_ASSIGNMENT_CHUNK = 4
_GENERIC_PACKAGES = {
    "core", "common", "utility", "utils", "runtime", "api", "service", "services",
    "platform", "framework", "integration", "layer", "layers",
}


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


class ModuleAssignmentAgent:
    """Assign components to candidate module families and package subpaths."""

    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.output_dir = Path(output_dir)
        self.llm_client = (
            LLMClient(self.api_config, output_dir, agent_name="module_assignment")
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

    def assign_modules(
        self,
        architectures: List[dict],
        actions: List[dict],
        layout_policy: Dict[str, Any],
        module_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        action_index = self._build_action_index(actions)
        rows: List[Dict[str, Any]] = []
        path_index: Dict[str, str] = {}
        canonical_candidates = [str(item) for item in (layout_policy.get("canonical_packages") or []) if str(item).strip()]
        default_package = str(layout_policy.get("default_subpackage") or "core").strip() or "core"
        module_families = module_plan.get("module_families", []) if isinstance(module_plan, dict) else []

        for arch in architectures if isinstance(architectures, list) else []:
            if not isinstance(arch, dict):
                continue
            parent_task = str(arch.get("parent_task") or arch.get("task") or "").strip()
            architecture = arch.get("architecture", {})
            if not parent_task or not isinstance(architecture, dict):
                continue
            logging.info("Module assignment parent: %s", parent_task)
            parent_module_families = [
                row for row in module_families
                if isinstance(row, dict) and str(row.get("parent_task", "")).strip() == parent_task
            ]
            parent_rows = self._assign_parent_modules(
                parent_task=parent_task,
                architecture=architecture,
                action_index=action_index,
                parent_module_families=parent_module_families,
                canonical_candidates=canonical_candidates,
                default_package=default_package,
            )
            rows.extend(parent_rows)
            for row in parent_rows:
                path_index[f"{parent_task}::{row['component']}"] = row["package_subpath"]

        return {
            "component_package_path_index": path_index,
            "assignments": rows,
            "stats": {
                "assigned_components": len(rows),
                "non_generic_subpaths": sum(1 for row in rows if "/" in row.get("package_subpath", "")),
            },
        }

    def _assign_parent_modules(
        self,
        *,
        parent_task: str,
        architecture: Dict[str, Any],
        action_index: Dict[str, Dict[str, str]],
        parent_module_families: List[Dict[str, Any]],
        canonical_candidates: List[str],
        default_package: str,
    ) -> List[Dict[str, Any]]:
        components = architecture.get("components", [])
        if not isinstance(components, list):
            return []
        if self.llm_client and parent_module_families:
            try:
                rows = self._assign_parent_modules_with_llm(
                    parent_task=parent_task,
                    architecture=architecture,
                    action_index=action_index,
                    parent_module_families=parent_module_families,
                    canonical_candidates=canonical_candidates,
                    default_package=default_package,
                )
                if rows:
                    return rows
            except Exception as exc:
                logging.warning("Module assignment LLM failed for '%s': %s", parent_task, exc)
        return self._assign_parent_modules_with_rules(
            parent_task=parent_task,
            architecture=architecture,
            action_index=action_index,
            parent_module_families=parent_module_families,
            canonical_candidates=canonical_candidates,
            default_package=default_package,
        )

    def _assign_parent_modules_with_llm(
        self,
        *,
        parent_task: str,
        architecture: Dict[str, Any],
        action_index: Dict[str, Dict[str, str]],
        parent_module_families: List[Dict[str, Any]],
        canonical_candidates: List[str],
        default_package: str,
    ) -> List[Dict[str, Any]]:
        assignment_input = self._build_parent_assignment_input(
            parent_task=parent_task,
            architecture=architecture,
            action_index=action_index,
            parent_module_families=parent_module_families,
            canonical_candidates=canonical_candidates,
            default_package=default_package,
        )
        response = self._run_llm_module_assignment_rounds(
            parent_task=parent_task,
            assignment_input=assignment_input,
        )
        rows: List[Dict[str, Any]] = []
        valid_components = set(assignment_input["component_names"])
        valid_families = set(assignment_input["module_family_names"])
        multi_component_parent = len(valid_components) > 1
        for row in response.get("assignments", []) if isinstance(response, dict) else []:
            if not isinstance(row, dict):
                continue
            component = str(row.get("component", "")).strip()
            module_family = _snake(str(row.get("module_family", "")).strip())
            package_subpath = self._ensure_subpackage_path(
                str(row.get("package_subpath", "")).strip(),
                module_family,
                force_nested=multi_component_parent,
            )
            if component and component in valid_components and module_family and module_family in valid_families:
                rows.append(
                    {
                        "parent_task": parent_task,
                        "component": component,
                        "module_family": module_family,
                        "package_subpath": package_subpath,
                        "rationale": str(row.get("rationale", "")).strip(),
                        "source": "llm",
                    }
                )
        return rows

    def _build_parent_assignment_input(
        self,
        *,
        parent_task: str,
        architecture: Dict[str, Any],
        action_index: Dict[str, Dict[str, str]],
        parent_module_families: List[Dict[str, Any]],
        canonical_candidates: List[str],
        default_package: str,
    ) -> Dict[str, Any]:
        components = architecture.get("components", [])
        features = architecture.get("sub_requirements", [])
        requirement_info = architecture.get("requirement", {}) if isinstance(architecture.get("requirement", {}), dict) else {}
        existing_context = self._collect_existing_package_context()

        feature_payload: List[Dict[str, Any]] = []
        for feature in features if isinstance(features, list) else []:
            if not isinstance(feature, dict):
                continue
            feature_name = str(feature.get("name", "")).strip()
            if not feature_name:
                continue
            feature_payload.append(
                {
                    "name": feature_name,
                    "description": str(feature.get("description", "")).strip(),
                }
            )

        component_payload: List[Dict[str, Any]] = []
        component_names: List[str] = []
        for comp in components if isinstance(components, list) else []:
            if not isinstance(comp, dict):
                continue
            name = str(comp.get("name", "")).strip()
            if not name:
                continue
            component_names.append(name)
            component_payload.append(
                {
                    "name": name,
                    "responsibility_summary": [
                        str(item).strip()
                        for item in (comp.get("responsibilities", []) if isinstance(comp.get("responsibilities", []), list) else [])[:3]
                        if str(item).strip()
                    ],
                    "serves_subrequirements": comp.get("serves_subrequirements", []),
                    "action": action_index.get(parent_task, {}).get(name, "save"),
                }
            )

        normalized_families: List[Dict[str, Any]] = []
        family_names: List[str] = []
        for row in parent_module_families:
            family = _snake(str(row.get("module_family", "")).strip())
            if not family:
                continue
            family_names.append(family)
            normalized_families.append(
                {
                    "module_family": family,
                    "covers_features": row.get("covers_features", []),
                    "package_subpath": self._normalize_package_subpath(str(row.get("package_subpath", "")).strip(), default_package),
                    "rationale": str(row.get("rationale", "")).strip(),
                }
            )

        return {
            "parent_requirement": {
                "name": parent_task,
                "description": str(requirement_info.get("description", "")).strip(),
            },
            "feature_families": feature_payload,
            "candidate_module_families": normalized_families,
            "components": component_payload,
            "canonical_package_candidates": canonical_candidates or [default_package],
            "default_package": default_package,
            "existing_package_context": existing_context,
            "component_names": component_names,
            "module_family_names": family_names,
        }

    def _run_llm_module_assignment_rounds(
        self,
        *,
        parent_task: str,
        assignment_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        drafts = self._run_assignment_draft_maybe_chunked(
            parent_task=parent_task,
            assignment_input=assignment_input,
        )
        if not drafts:
            return {}
        if len(drafts) == 1:
            draft = drafts[0]
        else:
            merge_prompt = f"""
You are consolidating chunked component-to-module assignments.

Assignment input:
{json.dumps({k: v for k, v in assignment_input.items() if k != "components" and k != "component_names"}, ensure_ascii=False, indent=2)}

Chunk drafts:
{json.dumps(drafts, ensure_ascii=False, indent=2)}

Task:
1. Merge the chunk drafts into one consistent assignment.
2. Keep assignments compact and realistic.
3. Prefer stable package subdirectories over flat package roots when multiple components share a family.

Return ONLY JSON:
{{
  "assignments": [
    {{
      "component": "ComponentName",
      "module_family": "time_series_models",
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
                operation_name=f"module_assignment.merge:{parent_task}",
            )
            draft = draft if isinstance(draft, dict) else {}

        draft_prompt = f"""
You are assigning components into previously planned candidate module families.

Assignment input:
{json.dumps(assignment_input, ensure_ascii=False, indent=2)}

Task:
1. Assign each component to exactly one existing candidate module family.
2. Reuse the candidate family package_subpath unless a clearly better subpath under the same family is needed.
3. Keep nearby components in the same family when their served subrequirements and responsibilities overlap.

Constraints:
- Do NOT invent new module families.
- Do NOT revise components.
- This step assigns components into candidate modules; it does not redesign the module plan.
- Prefer realistic package subdirectories rather than dumping everything directly under the package root.
- If multiple components share one family, keep them under one stable subdirectory family instead of scattering them.

Return ONLY JSON:
{{
  "assignments": [
    {{
      "component": "ComponentName",
      "module_family": "time_series_models",
      "package_subpath": "core/time_series_models",
      "rationale": "short explanation"
    }}
  ]
}}
""".strip()
        review_prompt = f"""
You are reviewing a draft component-to-module assignment.

Assignment input:
{json.dumps(assignment_input, ensure_ascii=False, indent=2)}

Draft assignment:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Review goals:
1. Every component must be assigned.
2. No component should be assigned to an unrelated family.
3. Assignments should respect served subrequirements and responsibility summaries.
4. The result should not collapse into flat package roots when subdirectories would improve clarity.

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
            operation_name=f"module_assignment.review:{parent_task}",
        )

        if isinstance(review, dict) and review.get("approved") is True:
            return draft if isinstance(draft, dict) else {}

        refine_prompt = f"""
You are refining a component-to-module assignment after review.

Assignment input:
{json.dumps(assignment_input, ensure_ascii=False, indent=2)}

Previous draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Review feedback:
{json.dumps(review, ensure_ascii=False, indent=2)}

Instructions:
- Keep the strong assignments from the previous draft.
- Fix missing or semantically weak assignments.
- Do not invent new module families.

Return ONLY JSON in the same schema as the draft.
""".strip()
        refined = self.llm_client.call_json(
            [
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": refine_prompt},
            ],
            temperature=0.0,
            max_tokens=20000,
            operation_name=f"module_assignment.refine:{parent_task}",
        )
        return refined if isinstance(refined, dict) else (draft if isinstance(draft, dict) else {})

    def _run_assignment_draft_maybe_chunked(
        self,
        *,
        parent_task: str,
        assignment_input: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        components = assignment_input.get("components", []) if isinstance(assignment_input.get("components", []), list) else []
        est_tokens = self._estimate_tokens(json.dumps(assignment_input, ensure_ascii=False, indent=2))
        if est_tokens <= _MAX_ASSIGNMENT_INPUT_EST_TOKENS or len(components) <= _MAX_COMPONENTS_PER_ASSIGNMENT_CHUNK:
            draft = self._run_assignment_draft_once(parent_task=parent_task, assignment_input=assignment_input)
            return [draft] if draft else []

        results: List[Dict[str, Any]] = []
        for idx in range(0, len(components), _MAX_COMPONENTS_PER_ASSIGNMENT_CHUNK):
            chunk = components[idx: idx + _MAX_COMPONENTS_PER_ASSIGNMENT_CHUNK]
            chunk_input = dict(assignment_input)
            chunk_input["components"] = chunk
            chunk_input["component_names"] = [str(item.get("name", "")).strip() for item in chunk if isinstance(item, dict)]
            draft = self._run_assignment_draft_once(
                parent_task=f"{parent_task}#chunk{idx // _MAX_COMPONENTS_PER_ASSIGNMENT_CHUNK + 1}",
                assignment_input=chunk_input,
            )
            if draft:
                results.append(draft)
        return results

    def _run_assignment_draft_once(
        self,
        *,
        parent_task: str,
        assignment_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        draft_prompt = f"""
You are assigning components into previously planned candidate module families.

Assignment input:
{json.dumps(assignment_input, ensure_ascii=False, indent=2)}

Task:
1. Assign each component to exactly one existing candidate module family.
2. Reuse the candidate family package_subpath unless a clearly better subpath under the same family is needed.
3. Keep nearby components in the same family when their served subrequirements and responsibilities overlap.

Constraints:
- Do NOT invent new module families.
- Do NOT revise components.
- This step assigns components into candidate modules; it does not redesign the module plan.
- Prefer realistic package subdirectories rather than dumping everything directly under the package root.
- If multiple components share one family, keep them under one stable subdirectory family instead of scattering them.

Return ONLY JSON:
{{
  "assignments": [
    {{
      "component": "ComponentName",
      "module_family": "time_series_models",
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
            operation_name=f"module_assignment.draft:{parent_task}",
        )
        return draft if isinstance(draft, dict) else {}

    def _assign_parent_modules_with_rules(
        self,
        *,
        parent_task: str,
        architecture: Dict[str, Any],
        action_index: Dict[str, Dict[str, str]],
        parent_module_families: List[Dict[str, Any]],
        canonical_candidates: List[str],
        default_package: str,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        normalized_families = []
        component_count = len([comp for comp in architecture.get("components", []) if isinstance(comp, dict) and str(comp.get("name", "")).strip()])
        for row in parent_module_families:
            family = _snake(str(row.get("module_family", "")).strip())
            if not family:
                continue
            normalized_families.append(
                {
                    "module_family": family,
                    "covers_features": row.get("covers_features", []) if isinstance(row.get("covers_features", []), list) else [],
                    "package_subpath": self._normalize_package_subpath(str(row.get("package_subpath", "")).strip(), default_package),
                }
            )

        for comp in architecture.get("components", []) if isinstance(architecture.get("components", []), list) else []:
            if not isinstance(comp, dict):
                continue
            component_name = str(comp.get("name", "")).strip()
            if not component_name:
                continue
            serves = comp.get("serves_subrequirements", []) if isinstance(comp.get("serves_subrequirements", []), list) else []
            responsibilities = comp.get("responsibilities", []) if isinstance(comp.get("responsibilities", []), list) else []
            family_sources: List[str] = list(serves) + [str(item) for item in responsibilities[:3]] + [component_name, parent_task]

            best_family = None
            best_score = -1.0
            for family in normalized_families:
                family_tokens = family.get("covers_features", []) + [family.get("module_family", "")]
                score_basis = _best_overlap_slug(family_sources, [str(x) for x in family_tokens if str(x).strip()])
                score = 1.0 if score_basis else 0.0
                overlap = set(_snake(" ".join(family_sources)).split("_")) & set(_snake(" ".join(str(x) for x in family_tokens)).split("_"))
                score += len([tok for tok in overlap if tok]) / 10.0
                if score > best_score:
                    best_score = score
                    best_family = family

            if best_family is None:
                fallback_family = (
                    _best_overlap_slug(family_sources, canonical_candidates)
                    or _slug(" ".join(family_sources))
                    or _slug(parent_task)
                    or "domain"
                )
                base_package = _best_overlap_slug([component_name, parent_task], canonical_candidates) or default_package
                base_package = _snake(base_package or default_package) or default_package
                package_subpath = f"{base_package}/{fallback_family}" if base_package in _GENERIC_PACKAGES else base_package
                rows.append(
                    {
                        "parent_task": parent_task,
                        "component": component_name,
                        "module_family": fallback_family,
                        "package_subpath": self._normalize_package_subpath(package_subpath, default_package),
                        "rationale": "fallback_no_candidate_family",
                        "source": "rules",
                    }
                )
                continue

            action = action_index.get(parent_task, {}).get(component_name, "save")
            package_subpath = str(best_family.get("package_subpath", "")).strip() or default_package
            package_subpath = self._ensure_subpackage_path(
                package_subpath,
                best_family["module_family"],
                force_nested=(action == "split" or component_count > 1),
            )
            rows.append(
                {
                    "parent_task": parent_task,
                    "component": component_name,
                    "module_family": best_family["module_family"],
                    "package_subpath": self._normalize_package_subpath(package_subpath, default_package),
                    "rationale": f"fallback_assign:{best_family['module_family']}",
                    "source": "rules",
                }
            )
        return rows

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

    def _ensure_subpackage_path(self, package_subpath: str, module_family: str, *, force_nested: bool = False) -> str:
        normalized = self._normalize_package_subpath(package_subpath, "core")
        parts = normalized.split("/")
        if len(parts) >= 2:
            return normalized
        root = parts[0]
        family = _snake(module_family)
        family_parts = [part for part in family.split("_") if part]
        root_parts = [part for part in root.split("_") if part]
        suffix = [part for part in family_parts if part not in root_parts]
        nested = "_".join(suffix[:2]) if suffix else ("domain" if root in _GENERIC_PACKAGES or force_nested else "")
        if not nested:
            return normalized
        return f"{root}/{nested}"

    def _estimate_tokens(self, text: str) -> int:
        return max(1, round(len(str(text or "")) / 4))

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
