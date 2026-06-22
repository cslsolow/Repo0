"""Package API plan builder for codegen/export stages."""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple


def _to_snake_case(name: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or "").strip())
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = text.strip("_").lower()
    return text or "unnamed_component"


def _to_pascal_case(name: str) -> str:
    snake = _to_snake_case(name)
    parts = [part for part in snake.split("_") if part]
    if not parts:
        return "UnnamedComponent"
    return "".join(part.capitalize() for part in parts)


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _tokenize(text: str) -> List[str]:
    return [tok for tok in _to_snake_case(text).split("_") if tok]


def _normalize_subpackage(name: str) -> str:
    tokens = [tok for tok in _tokenize(name) if tok]
    if not tokens:
        return ""
    return "_".join(tokens[:3])


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def _safe_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


_TOKEN_STOPWORDS = {
    "feature", "features", "task", "tasks", "requirement", "requirements",
    "component", "components", "module", "modules", "system", "implementation",
    "support", "manager", "engine", "service", "layer", "and", "with", "for",
    "core", "common", "utility", "utils", "pipeline", "framework",
}


_PLACEHOLDER_PACKAGES = {
    "",
    "unnamed",
    "unnamed_component",
    "generated",
    "default",
    "misc",
    "module",
    "modules",
    "component",
    "components",
}

_GENERIC_PACKAGES = {
    "core",
    "common",
    "utility",
    "utils",
    "pipeline",
    "framework",
    "base",
    "shared",
    "platform",
    "integration",
    "runtime",
    "service",
    "services",
    "layer",
    "layers",
    "api",
}


def _is_placeholder_package(name: str) -> bool:
    norm = _normalize_subpackage(name)
    if not norm:
        return True
    if norm in _PLACEHOLDER_PACKAGES:
        return True
    return norm.startswith("unnamed")


def _is_generic_package(name: str) -> bool:
    return _normalize_subpackage(name) in _GENERIC_PACKAGES


def _slug_from_text(text: str, max_tokens: int = 2) -> str:
    tokens = [tok for tok in _tokenize(text) if len(tok) >= 3 and tok not in _TOKEN_STOPWORDS]
    if not tokens:
        return ""
    return "_".join(tokens[:max_tokens])


def _contains_contiguous_tokens(needle: List[str], haystack: List[str]) -> bool:
    if not needle or not haystack or len(needle) > len(haystack):
        return False
    width = len(needle)
    for idx in range(0, len(haystack) - width + 1):
        if haystack[idx : idx + width] == needle:
            return True
    return False


def _preferred_package_anchor(
    parent_task: str,
    component: Dict[str, Any],
    candidates: List[str],
) -> str:
    sources: List[str] = [str(parent_task or "")]
    serves = component.get("serves_subrequirements", [])
    if isinstance(serves, list):
        sources.extend(str(item) for item in serves)
    responsibilities = component.get("responsibilities", [])
    if isinstance(responsibilities, list):
        sources.extend(str(item) for item in responsibilities[:4])

    for source in sources:
        slug = _slug_from_text(source, max_tokens=2)
        if not slug:
            continue
        if slug in candidates and not _is_generic_package(slug):
            return slug
        mapped = _best_canonical_subpackage(slug, candidates)
        if mapped and not _is_generic_package(mapped):
            return mapped
    return ""


def _derive_nested_subpackage(
    parent_task: str,
    component: Dict[str, Any],
    canonical_package: str,
    candidates: List[str],
) -> str:
    recommended_action = str(
        component.get("recommended_action")
        or component.get("action_hint")
        or component.get("suggested_action")
        or ""
    ).strip().lower()
    anchor = _preferred_package_anchor(parent_task, component, candidates)
    if not anchor:
        return ""
    canonical_norm = _normalize_subpackage(canonical_package)
    anchor_norm = _normalize_subpackage(anchor)
    if anchor_norm == canonical_norm:
        if recommended_action != "split":
            return ""
        component_slug = _slug_from_text(str(component.get("name", "")), max_tokens=2)
        component_norm = _normalize_subpackage(component_slug)
        if not component_norm or component_norm == anchor_norm:
            return ""
        return component_norm
    if not _is_generic_package(canonical_package) and recommended_action != "split":
        return ""
    return anchor


def _iter_architecture_components(architectures: List[dict]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for arch in architectures:
        if not isinstance(arch, dict):
            continue
        parent_task = str(arch.get("parent_task", "")).strip()
        architecture = arch.get("architecture", {})
        if not isinstance(architecture, dict):
            continue
        components = architecture.get("components", [])
        if not isinstance(components, list):
            continue
        for component in components:
            if isinstance(component, dict):
                yield parent_task, component


def _component_text_bag(parent_task: str, component: Dict[str, Any]) -> Tuple[str, List[str]]:
    parts: List[str] = [str(parent_task or ""), str(component.get("name", "") or "")]
    responsibilities = component.get("responsibilities", [])
    if isinstance(responsibilities, list):
        parts.extend(str(item) for item in responsibilities)
    serves = component.get("serves_subrequirements", [])
    if isinstance(serves, list):
        parts.extend(str(item) for item in serves)
    joined = " ".join(parts).strip().lower()
    tokens = [tok for tok in _tokenize(joined) if tok and tok not in _TOKEN_STOPWORDS]
    return joined, tokens


def _score_candidate(candidate: str, bag_text: str, bag_tokens: List[str]) -> float:
    cand_tokens = [tok for tok in _tokenize(candidate) if tok]
    if not cand_tokens:
        return 0.0
    bag_snake = _to_snake_case(bag_text)
    overlap = _jaccard(cand_tokens, bag_tokens)
    phrase_hit = 1.0 if candidate and (candidate in bag_snake or _contains_contiguous_tokens(cand_tokens, bag_tokens)) else 0.0
    best_token_match = 0.0
    for c_tok in cand_tokens:
        for b_tok in bag_tokens[:64]:
            best_token_match = max(best_token_match, _safe_ratio(c_tok, b_tok))
    generic_penalty = 0.12 if _is_generic_package(candidate) and any(tok not in _TOKEN_STOPWORDS for tok in bag_tokens) else 0.0
    return max(0.0, 0.55 * overlap + 0.30 * phrase_hit + 0.15 * best_token_match - generic_penalty)


def _pick_best_candidate(
    bag_text: str,
    bag_tokens: List[str],
    candidates: List[str],
) -> Tuple[str, float]:
    scored = [(_score_candidate(cand, bag_text, bag_tokens), cand) for cand in candidates]
    scored.sort(key=lambda row: (-row[0], row[1]))
    if not scored:
        return "core", 0.0
    best_score, best_cand = scored[0]
    return best_cand, float(best_score)


def _best_canonical_subpackage(slug: str, candidates: List[str]) -> str:
    """Map a semantic slug to the closest canonical package candidate."""
    norm = _normalize_subpackage(slug)
    if not norm:
        return ""
    if norm in candidates:
        return norm
    best, score = _pick_best_candidate(norm, _tokenize(norm), candidates)
    if score <= 0.0:
        return ""
    return best


def _build_candidate_packages(
    architectures: List[dict],
    layout_policy: Dict[str, Any],
) -> List[str]:
    alias_map = layout_policy.get("alias_map") or {}
    alias_candidates = [
        _normalize_subpackage(dst)
        for dst in alias_map.values()
        if _normalize_subpackage(dst)
    ]

    parent_counter: Counter[str] = Counter()
    component_counter: Counter[str] = Counter()
    parent_count = 0
    for parent_task, component in _iter_architecture_components(architectures):
        if parent_task:
            parent_count += 1
            slug = _slug_from_text(parent_task, max_tokens=2)
            if slug:
                parent_counter[slug] += 2
        cname = str(component.get("name", "")).strip()
        cslug = _slug_from_text(cname, max_tokens=2)
        if cslug:
            component_counter[cslug] += 1
        declared = _normalize_subpackage(str(component.get("canonical_package", "")).strip())
        if declared and not _is_placeholder_package(declared):
            component_counter[declared] += 2

    target_candidates = int(max(8, min(18, round(math.sqrt(max(parent_count, 1)) * 4))))
    merged_counter: Counter[str] = Counter()
    merged_counter.update(parent_counter)
    merged_counter.update(component_counter)

    ranked = [name for name, _ in merged_counter.most_common(target_candidates)]
    candidates = _dedupe_keep_order(alias_candidates + ranked)
    candidates = [cand for cand in candidates if not _is_placeholder_package(cand)]
    if not candidates:
        candidates = ["core"]
    if "core" not in candidates:
        candidates.append("core")
    return candidates


def build_canonical_package_grouping(
    architectures: List[dict],
    layout_policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Build canonical package groups and assign each component to one canonical package."""
    layout_root = str(layout_policy.get("layout_root") or "").strip().strip("/")
    alias_map = {
        _normalize_subpackage(str(k)): _normalize_subpackage(str(v))
        for k, v in (layout_policy.get("alias_map") or {}).items()
        if _normalize_subpackage(str(k)) and _normalize_subpackage(str(v))
    }
    candidates = _build_candidate_packages(architectures, layout_policy)
    default_package = candidates[0] if candidates else "core"

    assignments: List[Dict[str, Any]] = []
    assignment_index: Dict[str, str] = {}
    package_counter: Counter[str] = Counter()

    for parent_task, component in _iter_architecture_components(architectures):
        component_name = str(component.get("name", "")).strip()
        if not component_name:
            continue
        key = f"{parent_task}::{component_name}"
        bag_text, bag_tokens = _component_text_bag(parent_task, component)
        anchor_package = _preferred_package_anchor(parent_task, component, candidates)
        component_default = anchor_package or default_package

        declared = _normalize_subpackage(str(component.get("canonical_package", "")).strip())
        if _is_placeholder_package(declared):
            declared = ""
        alias_hit = ""
        for alias_key, alias_dst in alias_map.items():
            if alias_key and alias_key in bag_text:
                alias_hit = alias_dst
                break

        reason = "scored"
        score = 0.0
        if declared and declared in candidates:
            chosen = declared
            reason = "declared"
            score = 1.0
        elif alias_hit and alias_hit in candidates:
            chosen = alias_hit
            reason = "alias_map"
            score = 0.95
        else:
            chosen, score = _pick_best_candidate(bag_text, bag_tokens, candidates)
            if anchor_package:
                anchor_score = _score_candidate(anchor_package, bag_text, bag_tokens)
                if chosen == anchor_package:
                    score = max(score, anchor_score)
                else:
                    recommended_action = str(
                        component.get("recommended_action")
                        or component.get("action_hint")
                        or component.get("suggested_action")
                        or ""
                    ).strip().lower()
                    prefers_finer_package = recommended_action == "split"
                    if _is_generic_package(chosen) or score < 0.24 or prefers_finer_package:
                        if anchor_score >= max(0.18, score - 0.08):
                            chosen = anchor_package
                            reason = "parent_anchor"
                            score = anchor_score
            if score < 0.16:
                chosen = component_default
                reason = "default_fallback"

        assignment_index[key] = chosen
        package_counter[chosen] += 1
        assignments.append(
            {
                "parent_task": parent_task,
                "component": component_name,
                "canonical_package": chosen,
                "score": round(float(score), 4),
                "reason": reason,
            }
        )

    assignments.sort(key=lambda row: (row.get("parent_task", ""), row.get("component", "")))
    return {
        "layout_root": layout_root,
        "candidate_packages": candidates,
        "default_package": default_package,
        "component_assignment_index": assignment_index,
        "assignments": assignments,
        "stats": {
            "component_count": len(assignments),
            "candidate_count": len(candidates),
            "package_distribution": dict(sorted(package_counter.items(), key=lambda item: item[0])),
        },
    }


def derive_component_export_symbols(
    component_name: str,
    responsibilities: Any,
    planned_file_path: str,
) -> List[str]:
    symbols: List[str] = []
    name = str(component_name or "").strip()
    if name:
        symbols.append(_to_pascal_case(name))
        symbols.append(_to_snake_case(name))

    planned_path = str(planned_file_path or "").strip().replace("\\", "/")
    if planned_path:
        stem = Path(planned_path).stem
        if stem and stem != "__init__":
            symbols.append(stem)
            symbols.append(_to_pascal_case(stem))

    if isinstance(responsibilities, list):
        for resp in responsibilities:
            text = str(resp or "")
            for token in re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text):
                symbols.append(token)
            for token in re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", text):
                symbols.append(token)
            for token in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
                if not token.startswith("_"):
                    symbols.append(token)

    return _dedupe_keep_order(symbols)


def build_package_api_plan(
    architectures: List[dict],
    layout_policy: Dict[str, Any],
    build_component_file_plan: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, str]],
) -> Dict[str, Any]:
    layout_root = str(layout_policy.get("layout_root") or "").strip().strip("/")
    package_modules: Dict[str, List[Dict[str, Any]]] = {}
    component_rows: List[Dict[str, Any]] = []

    grouping = build_canonical_package_grouping(architectures, layout_policy)
    assignment_index = grouping.get("component_assignment_index", {})
    candidates = grouping.get("candidate_packages", [])
    default_package = grouping.get("default_package", "core")

    policy_with_grouping = dict(layout_policy)
    policy_with_grouping["component_package_index"] = assignment_index
    policy_with_grouping["canonical_packages"] = candidates
    policy_with_grouping["default_subpackage"] = default_package

    assignment_row_index: Dict[str, Dict[str, Any]] = {}
    for row in grouping.get("assignments", []):
        if not isinstance(row, dict):
            continue
        key = f"{row.get('parent_task','')}::{row.get('component','')}"
        assignment_row_index[key] = row

    for arch in architectures:
        if not isinstance(arch, dict):
            continue
        parent_task = str(arch.get("parent_task", "")).strip()
        architecture = arch.get("architecture", {})
        if not isinstance(architecture, dict):
            continue
        components = architecture.get("components", [])
        if not isinstance(components, list):
            continue

        unified_task = {
            "name": parent_task,
            "description": architecture.get("requirement", {}).get("description", ""),
            "sub_requirements": arch.get("sub_tasks", []),
            "parent_node": arch.get("parent_node"),
            "parent_prev_node": arch.get("parent_prev_node"),
        }
        planned_paths = build_component_file_plan(architecture, unified_task, policy_with_grouping)

        for component in components:
            if not isinstance(component, dict):
                continue
            component_name = str(component.get("name", "")).strip()
            if not component_name:
                continue
            key = f"{parent_task}::{component_name}"
            assignment_row = assignment_row_index.get(key, {})
            canonical_package = str(assignment_row.get("canonical_package", "")).strip()
            planned_file_path = str(planned_paths.get(component_name, "")).strip()
            if not planned_file_path:
                module_name_fallback = _to_snake_case(component_name)
                nested_subpackage = _derive_nested_subpackage(
                    parent_task=parent_task,
                    component=component,
                    canonical_package=canonical_package or default_package,
                    candidates=candidates,
                )
                package_subpath = canonical_package or default_package
                if nested_subpackage:
                    package_subpath = f"{package_subpath}/{nested_subpackage}"
                if package_subpath:
                    planned_file_path = f"{layout_root}/{package_subpath}/{module_name_fallback}.py"
                else:
                    planned_file_path = f"{layout_root}/{default_package}/{module_name_fallback}.py"
            package_subpath = str(Path(planned_file_path).parent).replace("\\", "/").strip("/")
            if layout_root and package_subpath.startswith(f"{layout_root}/"):
                package_subpath = package_subpath[len(layout_root) + 1 :]

            responsibilities = component.get("responsibilities", [])
            export_symbols = derive_component_export_symbols(
                component_name=component_name,
                responsibilities=responsibilities,
                planned_file_path=planned_file_path,
            )
            package_dir = str(Path(planned_file_path).parent).replace("\\", "/").strip("/") if planned_file_path else layout_root
            module_name = Path(planned_file_path).stem if planned_file_path else _to_snake_case(component_name)
            if module_name == "__init__":
                module_name = _to_snake_case(component_name)

            row = {
                "parent_task": parent_task,
                "component": component_name,
                "planned_file_path": planned_file_path,
                "package_dir": package_dir,
                "module_name": module_name,
                "canonical_package": canonical_package or default_package,
                "package_subpath": package_subpath,
                "grouping_score": assignment_row.get("score", 0.0),
                "grouping_reason": assignment_row.get("reason", "unknown"),
                "export_symbols": export_symbols,
            }
            component_rows.append(row)
            package_modules.setdefault(package_dir, []).append(
                {
                    "module_name": module_name,
                    "component": component_name,
                    "planned_file_path": planned_file_path,
                    "canonical_package": canonical_package or default_package,
                    "export_symbols": export_symbols,
                }
            )

    package_rows: List[Dict[str, Any]] = []
    for package_dir, modules in sorted(package_modules.items(), key=lambda item: item[0]):
        merged_symbols: List[str] = []
        for module in modules:
            merged_symbols.extend(module.get("export_symbols", []))
        package_rows.append(
            {
                "package_dir": package_dir,
                "module_count": len(modules),
                "modules": modules,
                "planned_exports": _dedupe_keep_order(merged_symbols),
            }
        )

    component_index: Dict[str, Dict[str, Any]] = {}
    for row in component_rows:
        key = f"{row.get('parent_task','')}::{row.get('component','')}"
        component_index[key] = row

    return {
        "layout_root": layout_root,
        "package_count": len(package_rows),
        "component_count": len(component_rows),
        "canonical_packages": candidates,
        "default_package": default_package,
        "grouping": grouping,
        "packages": package_rows,
        "components": component_rows,
        "component_index": component_index,
    }


__all__ = [
    "build_canonical_package_grouping",
    "build_package_api_plan",
    "derive_component_export_symbols",
]
