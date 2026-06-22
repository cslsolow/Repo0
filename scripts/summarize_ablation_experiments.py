#!/usr/bin/env python3
"""Summarize Repo0 ablation coverage, novelty, and test-pass results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_REPOS = ("statsmodels", "django", "requests")
DEFAULT_METHODS = (
    "full",
    "no_decomposition",
    "no_dependency",
    "no_graph_module",
    "no_strategist",
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


def infer_method(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("ablation_gpt5mini_fix_data_"):
            method = part.removeprefix("ablation_gpt5mini_fix_data_")
            return re.sub(r"_\d{8}_\d{6}$", "", method)
        if part.startswith("repos_gpt5mini_fix_data_"):
            return part.removeprefix("repos_gpt5mini_fix_data_")
        if part == "repos_gpt5mini_fix_data":
            return "full"
    return None


def latest_coverage(root: Path, repos: set[str], methods: set[str]) -> dict[tuple[str, str], Path]:
    latest: dict[tuple[str, str], Path] = {}
    candidates: list[Path] = []
    for method in methods:
        if method == "full":
            candidates.extend(root.glob("repos_gpt5mini_fix_data/*/coverage_eval/*/coverage_result.json"))
            candidates.extend(root.glob("repos_gpt5mini_fix_data/*/rpg_checkpoints/coverage_result.json"))
            continue
        candidates.extend(root.glob(f"ablation_gpt5mini_fix_data_{method}_*/*/coverage_eval/*/coverage_result.json"))
        candidates.extend(root.glob(f"repos_gpt5mini_fix_data_{method}/*/coverage_eval/*/coverage_result.json"))
    for path in candidates:
        method = infer_method(path)
        if method not in methods:
            continue
        parts = set(path.parts)
        repo = next((item for item in repos if item in parts), None)
        if repo is None:
            continue
        key = (repo, method)
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path
    return latest


def pass_from_result(path: Path) -> tuple[bool, bool]:
    data = load_json(path)
    answer = str(data.get("answer", "")).lower()
    output = str(data.get("test_output", "")).lower()
    passed = "passed" in answer or re.search(r"\b[1-9]\d*\s+passed\b", output) is not None
    return passed, bool(data.get("voting"))


def summarize_test_pass(root: Path, repos: set[str], methods: set[str]) -> dict[tuple[str, str], dict[str, int]]:
    summary: dict[tuple[str, str], dict[str, int]] = {}
    ablation_root = root / "ablation"
    for repo in repos:
        repo_dir = ablation_root / repo
        if not repo_dir.is_dir():
            continue
        for method_dir in repo_dir.iterdir():
            if not method_dir.is_dir() or method_dir.name not in methods:
                continue
            latest_by_task: dict[str, Path] = {}
            for path in (method_dir / "cache" / "sample30").glob("task_*/results/result.json"):
                task_id = path.parent.parent.name
                if task_id not in latest_by_task or path.stat().st_mtime > latest_by_task[task_id].stat().st_mtime:
                    latest_by_task[task_id] = path
            passed = 0
            voted = 0
            for path in latest_by_task.values():
                ok, vote = pass_from_result(path)
                passed += int(ok)
                voted += int(vote)
            summary[(repo, method_dir.name)] = {
                "completed": len(latest_by_task),
                "passed": passed,
                "voted": voted,
            }
    return summary


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    repos = set(args.repos)
    methods = set(args.methods)
    coverage = latest_coverage(args.coverage_root, repos, methods)
    test_pass = summarize_test_pass(args.repocraft_runs, repos, methods)
    rows = []
    for repo in args.repos:
        for method in args.methods:
            cov_path = coverage.get((repo, method))
            cov_data = load_json(cov_path) if cov_path else {}
            pass_data = test_pass.get((repo, method), {})
            completed = pass_data.get("completed", 0)
            passed = pass_data.get("passed", 0)
            rows.append(
                {
                    "repo": repo,
                    "method": method,
                    "coverage": pct(cov_data.get("coverage_ratio")),
                    "novelty": pct(cov_data.get("new_feature_ratio")),
                    "gt_paths": str(cov_data.get("num_gt_paths", "")),
                    "generated_paths": str(cov_data.get("num_repo_paths", "")),
                    "test_completed": str(completed or ""),
                    "test_passed": str(passed or ""),
                    "test_pass_rate": pct(passed / completed) if completed else "",
                    "coverage_source": str(cov_path or ""),
                }
            )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    cols = [c for c in rows[0] if c != "coverage_source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Repo0 Ablation Summary\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row[col] for col in cols) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-root", type=Path, default=Path("."))
    parser.add_argument("--repocraft-runs", type=Path, default=Path("outputs/repocraft_runs"))
    parser.add_argument("--repos", nargs="+", default=list(DEFAULT_REPOS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--csv", type=Path, default=Path("outputs/repocraft_runs/ablation_summary.csv"))
    parser.add_argument("--md", type=Path, default=Path("outputs/repocraft_runs/ablation_summary.md"))
    args = parser.parse_args()

    rows = build_rows(args)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    print(f"rows={len(rows)}")
    print(f"csv={args.csv}")
    print(f"md={args.md}")


if __name__ == "__main__":
    main()
