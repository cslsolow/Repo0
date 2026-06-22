#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
TEST_REPO_ROOT = Path(os.environ.get("REPO0_TEST_REPO_ROOT", ROOT / "tmp" / "test_repo")).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


FETCH_MOD = _load_module(TEST_REPO_ROOT / "fetch_pr_data.py", "fetch_pr_data_mod")
PR_REQ_MOD = _load_module(ROOT / "agents" / "ingest" / "pr_rqmts_paser.py", "pr_rqmts_parser_mod")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a requirement-iteration workspace from GitHub PRs.")
    parser.add_argument("--repo", required=True, help="Repository key, e.g. statsmodels/statsmodels")
    parser.add_argument(
        "--prnumbers-file",
        type=Path,
        default=TEST_REPO_ROOT / "prnumbers_human.json",
        help="JSON file mapping repositories to PR numbers.",
    )
    parser.add_argument(
        "--pr-data-file",
        type=Path,
        default=TEST_REPO_ROOT / "pr_data.jsonl",
        help="Existing PR data JSONL.",
    )
    parser.add_argument(
        "--baseline-repo-dir",
        type=Path,
        default=ROOT / "repos_gpt5mini_schema_fix" / "statsmodels",
        help="Existing generated repo directory to copy as the iteration baseline.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=ROOT / "repos_gpt5mini_requirement_iteration",
        help="Root directory where the copied iteration repo will be created.",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default="",
        help="Optional target repo directory name. Defaults to the repo leaf name under --target-root, e.g. 'statsmodels'.",
    )
    parser.add_argument("--fetch-missing", action="store_true", help="Fetch missing PR payloads from GitHub.")
    parser.add_argument("--base-url", type=str, default="", help="LLM base_url for PR->requirements extraction.")
    parser.add_argument("--api-key", type=str, default="", help="LLM api_key for PR->requirements extraction.")
    parser.add_argument("--reasoning-effort", type=str, default="", help="Optional reasoning_effort to pass through to the LLM API.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="LLM model for PR->requirements extraction.")
    return parser.parse_args()


def load_prnumbers(path: Path, repo: str) -> List[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get(repo, [])
    return [int(v) for v in values]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            items.append(payload)
    return items


def select_pr_items(repo: str, pr_numbers: List[int], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = {(repo, int(num)) for num in pr_numbers}
    selected: List[Dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in items:
        key = (str(item.get("repository", "")), int(item.get("pr_number", -1)))
        if key in wanted and key not in seen:
            selected.append(item)
            seen.add(key)
    return selected


def fetch_missing_items(repo: str, pr_numbers: List[int], existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    token = FETCH_MOD.get_github_token()
    owner, repo_name = FETCH_MOD.get_repo_info(repo)
    existing_keys = {(str(item.get("repository", "")), int(item.get("pr_number", -1))) for item in existing}
    fetched: List[Dict[str, Any]] = []
    for pr_number in pr_numbers:
        key = (repo, int(pr_number))
        if key in existing_keys:
            continue
        item = FETCH_MOD.fetch_pr_data(owner, repo_name, int(pr_number), token)
        if item:
            fetched.append(item)
    return fetched


def build_requirement_records(pr_items: List[Dict[str, Any]], *, output_dir: Path, model: str, base_url: str, api_key: str, reasoning_effort: str) -> List[Dict[str, Any]]:
    llm_client = PR_REQ_MOD.LLMClient(
        {"api_key": api_key, "base_url": base_url, "model": model, "reasoning_effort": reasoning_effort},
        str(output_dir),
        agent_name="pr_rqmts_parser",
    )
    records: List[Dict[str, Any]] = []
    for pr_item in pr_items:
        normalized_item = dict(pr_item)
        if normalized_item.get("description_with_link_context"):
            normalized_item["description"] = normalized_item["description_with_link_context"]
        requirement = PR_REQ_MOD.generate_requirement_for_pr(normalized_item, llm_client)
        if requirement is None:
            continue
        repo = pr_item.get("repository")
        pr_number = pr_item.get("pr_number")
        key = f"{repo}-{pr_number}" if repo and pr_number is not None else f"pr-{len(records)+1}"
        records.append(
            {
                key: requirement,
                "pr": {
                    "repository": repo,
                    "pr_number": pr_number,
                    "title": pr_item.get("title", ""),
                    "description": pr_item.get("description", ""),
                    "resolved_links": pr_item.get("resolved_links", []),
                },
            }
        )
    return records


def choose_target_dir(target_root: Path, repo: str, target_name: str) -> Path:
    repo_leaf = repo.split("/")[-1].strip()
    safe_repo = repo_leaf or repo.replace("/", "_")
    name = target_name.strip() or safe_repo
    return (target_root / name).resolve()


def copy_baseline_repo(src: Path, dest: Path) -> None:
    if dest.exists():
        raise FileExistsError(f"Target iteration repo already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    pr_numbers = load_prnumbers(args.prnumbers_file, args.repo)
    if not pr_numbers:
        raise SystemExit(f"No PR numbers found for {args.repo} in {args.prnumbers_file}")

    pr_items = select_pr_items(args.repo, pr_numbers, load_jsonl(args.pr_data_file))
    if args.fetch_missing:
        pr_items.extend(fetch_missing_items(args.repo, pr_numbers, pr_items))

    target_dir = choose_target_dir(args.target_root, args.repo, args.target_name)
    copy_baseline_repo(args.baseline_repo_dir.resolve(), target_dir)

    iteration_input_dir = target_dir / "iteration_input"
    raw_pr_path = iteration_input_dir / "selected_pr_data.jsonl"
    write_jsonl(raw_pr_path, pr_items)

    evolve_requirements_path = iteration_input_dir / "evolve_requirements.jsonl"
    records = build_requirement_records(
        pr_items,
        output_dir=iteration_input_dir,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        reasoning_effort=args.reasoning_effort,
    )
    write_jsonl(evolve_requirements_path, records)

    manifest = {
        "repo": args.repo,
        "baseline_repo_dir": str(args.baseline_repo_dir.resolve()),
        "target_repo_dir": str(target_dir),
        "selected_pr_count": len(pr_items),
        "selected_pr_numbers": pr_numbers,
        "selected_pr_data": str(raw_pr_path),
        "evolve_requirements_file": str(evolve_requirements_path),
    }
    manifest_path = iteration_input_dir / "iteration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
