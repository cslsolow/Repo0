#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_round_files(output_dir: Path, pattern: str) -> list[Path]:
    def round_key(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.rsplit("_", 1)[-1])
        except Exception:
            return 0

    return sorted(output_dir.glob(pattern), key=round_key)


def format_proposed_actions(path: Path) -> list[str]:
    data = load_json(path)
    lines = [f"[proposed] {path.name}"]
    for parent in data if isinstance(data, list) else []:
        task = str(parent.get("task") or parent.get("parent_task") or "").strip()
        rows = parent.get("actions", []) if isinstance(parent, dict) else []
        if not rows:
            continue
        lines.append(f"  - {task}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            component = str(row.get("component") or "").strip()
            action = str(row.get("action") or "").strip()
            origin = str(row.get("action_origin") or "").strip()
            rationale = str(row.get("rationale") or "").strip()
            extra = f" [{origin}]" if origin else ""
            lines.append(f"      {component}: {action}{extra}")
            if rationale:
                lines.append(f"        rationale: {rationale}")
    return lines


def format_round_stats(path: Path) -> list[str]:
    data = load_json(path)
    stats = data.get("stats", {}) if isinstance(data, dict) else {}
    return [
        f"[round] {path.name}",
        "  merge_groups={merge} split_groups={split} action_counts={counts}".format(
            merge=stats.get("merge_group_count", 0),
            split=stats.get("split_group_count", 0),
            counts=stats.get("action_counts", {}),
        ),
    ]


def format_accepted_actions(path: Path) -> list[str]:
    data = load_json(path)
    lines = [f"[accepted] {path.name}"]
    for parent in data.get("parents", []) if isinstance(data, dict) else []:
        if not isinstance(parent, dict):
            continue
        task = str(parent.get("parent_task") or "").strip()
        merge_report = parent.get("merge_report") or {}
        split_report = parent.get("split_report") or {}

        merge_groups = merge_report.get("merge_groups", []) if isinstance(merge_report, dict) else []
        if merge_groups:
            lines.append(f"  - {task} merge:")
            for group in merge_groups:
                merged_name = str(group.get("merged_name") or "").strip()
                source_names = [str(item.get("source_name") or "").strip() for item in group.get("sources", []) if isinstance(item, dict)]
                lines.append(f"      {source_names} -> {merged_name}")

        split_groups = split_report.get("split_groups", []) if isinstance(split_report, dict) else []
        if split_groups:
            lines.append(f"  - {task} split:")
            for group in split_groups:
                component_name = str(group.get("component_name") or "").strip()
                split_into = [str(item).strip() for item in group.get("split_into", []) if str(item).strip()]
                confidence = group.get("confidence")
                lines.append(f"      {component_name} -> {split_into} (confidence={confidence})")
                reason = str(group.get("reason") or "").strip()
                if reason:
                    lines.append(f"        reason: {reason}")
    return lines


def format_gap_add(path: Path) -> list[str]:
    data = load_json(path)
    lines = [f"[gap-add] {path.name}", f"  accepted_count={data.get('accepted_count', 0)}"]
    for parent in data.get("parents", []) if isinstance(data, dict) else []:
        if not isinstance(parent, dict):
            continue
        if not parent.get("accepted"):
            continue
        lines.append(
            "  - {parent}: {decision} final_confidence={score}".format(
                parent=parent.get("parent_requirement", ""),
                decision=parent.get("decision", ""),
                score=parent.get("final_confidence", ""),
            )
        )
    return lines


def format_architecture_summary(path: Path) -> list[str]:
    data = load_json(path)
    if not isinstance(data, list):
        return []
    parent_count = len(data)
    component_count = 0
    subreq_count = 0
    lines = [f"[architecture-summary] {path.name}"]
    for parent in data:
        if not isinstance(parent, dict):
            continue
        architecture = parent.get("architecture", {}) or {}
        components = architecture.get("components", []) or []
        subreqs = architecture.get("sub_requirements", []) or []
        component_count += len(components)
        subreq_count += len(subreqs)
    lines.append(f"  parents={parent_count} components={component_count} subrequirements={subreq_count}")
    return lines


def build_summary(output_dir: Path) -> str:
    proposed = iter_round_files(output_dir, "actions_round_*.json")
    rounds = iter_round_files(output_dir, "action_refinement_round_*.json")
    accepted = iter_round_files(output_dir, "component_refinement_report_round_*.json")
    lines: list[str] = []

    for path in proposed:
        lines.extend(format_proposed_actions(path))
    for path in rounds:
        lines.extend(format_round_stats(path))
    for path in accepted:
        lines.extend(format_accepted_actions(path))

    gap_add = output_dir / "gap_addition_report.json"
    if gap_add.exists():
        lines.extend(format_gap_add(gap_add))

    final_actions = output_dir / "actions.json"
    if final_actions.exists():
        lines.append(f"[final-actions] {final_actions}")
    summary = output_dir / "action_refinement_report.json"
    if summary.exists():
        lines.append(f"[final-summary] {summary}")

    if lines:
        return "\n".join(lines)

    architectures = output_dir / "architectures.json"
    if architectures.exists():
        arch_lines = format_architecture_summary(architectures)
        if arch_lines:
            return "\n".join(arch_lines)

    return f"No round artifacts found under {output_dir}"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: print_action_rounds.py <agents_output_dir>", file=sys.stderr)
        return 1
    output_dir = Path(sys.argv[1]).expanduser().resolve()
    if not output_dir.is_dir():
        print(f"Missing agents_output dir: {output_dir}", file=sys.stderr)
        return 1

    print(build_summary(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
