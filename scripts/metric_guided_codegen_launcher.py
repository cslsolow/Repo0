#!/usr/bin/env python3
"""Prepare a single-repo metric-guided codegen run in a fresh output directory."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_README_REQ_ROOT = ROOT / "repo_input"
DEFAULT_PYTHON_BIN = Path(os.environ.get("PYTHON", sys.executable))


@dataclass(frozen=True)
class LauncherConfig:
    repo: str
    workspace_dir: Path
    baseline_root: Path
    output_root: Path
    run_label: str
    run_ts: str
    python_bin: Path
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str
    max_workers: int
    postcheck_max_workers: int
    log_level: str
    parent_codegen_dag_source: str
    action_refinement_rounds: int
    action_refinement_stop_on_stable: bool
    action_refinement_save_stops_component: bool
    use_processes: bool
    enable_component_metric_actions: bool
    enable_component_metric_merge_judge: bool
    component_metric_split_cohesion_threshold: float
    component_metric_split_min_subrequirements: int
    component_split_min_confidence: float
    component_metric_merge_max_small_subrequirements: int
    retry_empty_generated_components: bool
    readme_req_source: Path | None
    requirements_source: Path | None
    enable_gap_add_actions: bool
    gap_add_proposal_threshold: float
    gap_add_component_threshold: float
    gap_add_requirement_threshold: float
    stop_after_architecture_refinement: bool


@dataclass(frozen=True)
class RunPaths:
    baseline_repo_dir: Path
    run_dir: Path
    repo_root: Path
    output_dir: Path
    requirements_file: Path
    readme_req_source: Path
    local_readme_req: Path
    launch_script: Path
    manifest_file: Path
    status_file: Path
    log_dir: Path
    log_file: Path
    pid_file: Path


def _default_run_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_readme_req_source(config: LauncherConfig) -> Path:
    if config.readme_req_source is not None:
        return config.readme_req_source.expanduser().resolve()
    return (DEFAULT_README_REQ_ROOT / config.repo / "README.req").resolve()


def resolve_run_paths(config: LauncherConfig) -> RunPaths:
    baseline_repo_dir = (config.baseline_root / config.repo).expanduser().resolve()
    run_dir = (config.output_root / config.repo / f"{config.run_label}_{config.run_ts}").expanduser().resolve()
    repo_root = run_dir / config.repo
    output_dir = repo_root / "agents_output"
    requirements_file = repo_root / "readme_output" / "requirements.json"
    readme_req_source = resolve_readme_req_source(config)
    local_readme_req = repo_root / "README.req"
    launch_script = run_dir / "launch_run_agents.sh"
    manifest_file = run_dir / "launcher_manifest.json"
    status_file = run_dir / "status.txt"
    log_dir = (config.workspace_dir / "logs" / "metric_guided_codegen" / config.repo).resolve()
    log_file = log_dir / f"{config.run_label}_{config.repo}_{config.run_ts}.log"
    pid_file = run_dir / "pid.txt"
    return RunPaths(
        baseline_repo_dir=baseline_repo_dir,
        run_dir=run_dir,
        repo_root=repo_root,
        output_dir=output_dir,
        requirements_file=requirements_file,
        readme_req_source=readme_req_source,
        local_readme_req=local_readme_req,
        launch_script=launch_script,
        manifest_file=manifest_file,
        status_file=status_file,
        log_dir=log_dir,
        log_file=log_file,
        pid_file=pid_file,
    )


def validate_inputs(config: LauncherConfig, paths: RunPaths) -> None:
    if not config.repo.strip():
        raise ValueError("repo must be non-empty")
    if not paths.baseline_repo_dir.is_dir():
        raise FileNotFoundError(f"Missing baseline repo directory: {paths.baseline_repo_dir}")
    if not paths.readme_req_source.is_file():
        raise FileNotFoundError(f"Missing README.req source file: {paths.readme_req_source}")
    if config.requirements_source is not None and not config.requirements_source.is_file():
        raise FileNotFoundError(f"Missing requirements source file: {config.requirements_source}")
    if not config.python_bin.is_file():
        raise FileNotFoundError(f"Missing python binary: {config.python_bin}")
    if not config.api_key.strip():
        raise ValueError("api_key must be non-empty")
    if paths.run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {paths.run_dir}")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def copy_baseline_repo(config: LauncherConfig, paths: RunPaths) -> None:
    paths.run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(paths.baseline_repo_dir, paths.repo_root)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.readme_req_source, paths.local_readme_req)
    if config.requirements_source is not None:
        shutil.copy2(config.requirements_source, paths.requirements_file)


def prune_reused_artifacts(paths: RunPaths) -> None:
    for rel_path in (
        "requirements_merge_result.json",
        "requirements_for_dag.json",
        "edges_for_dag.json",
        "requirement_dag.json",
        "decomposed_dag.json",
        "plan.json",
        "architectures.json",
        "architectures_flattened.json",
        "component_merge_report.json",
        "component_merge_embedding_report.json",
        "dependency_graph.json",
        "layout_grouping_report.json",
        "package_api_plan.json",
        "module_plan.json",
        "module_assignment.json",
        "actions.json",
        "action_refinement_report.json",
        "component_metric_action_report.json",
        "gap_addition_report.json",
        "generated_files.json",
    ):
        target = paths.output_dir / rel_path
        if target.exists():
            _remove_path(target)

    for pattern in (
        "actions_round_*.json",
        "action_refinement_round_*.json",
        "architectures_round_*_before_pre_action_merge.json",
        "architectures_round_*_after_pre_action_merge.json",
        "component_refinement_report_round_*.json",
        "component_merge_report_round_*_pre_action.json",
    ):
        for target in paths.output_dir.glob(pattern):
            _remove_path(target)


def build_run_agents_command(config: LauncherConfig, paths: RunPaths) -> List[str]:
    command = [
        str(config.python_bin),
        str((config.workspace_dir / "run_agents.py").resolve()),
        "--repo",
        config.repo,
        "--workspace",
        str(config.workspace_dir),
        "--output",
        str(paths.output_dir),
        "--requirements-file",
        str(paths.requirements_file),
        "--req-path",
        str(paths.local_readme_req),
        "--base-url",
        config.base_url,
        "--api-key",
        config.api_key,
        "--model",
        config.model,
        "--reasoning-effort",
        config.reasoning_effort,
        "--max-workers",
        str(config.max_workers),
        "--postcheck-max-workers",
        str(config.postcheck_max_workers),
        "--log-level",
        config.log_level,
        "--parent-codegen-dag-source",
        config.parent_codegen_dag_source,
        "--action-refinement-rounds",
        str(config.action_refinement_rounds),
        "--component-metric-split-cohesion-threshold",
        str(config.component_metric_split_cohesion_threshold),
        "--component-metric-split-min-subrequirements",
        str(config.component_metric_split_min_subrequirements),
        "--component-split-min-confidence",
        str(config.component_split_min_confidence),
        "--component-metric-merge-max-small-subrequirements",
        str(config.component_metric_merge_max_small_subrequirements),
        "--gap-add-proposal-threshold",
        str(config.gap_add_proposal_threshold),
        "--gap-add-component-threshold",
        str(config.gap_add_component_threshold),
        "--gap-add-requirement-threshold",
        str(config.gap_add_requirement_threshold),
    ]
    if config.use_processes:
        command.append("--use-processes")
    if config.retry_empty_generated_components:
        command.append("--retry-empty-generated-components")
    if config.enable_component_metric_actions:
        command.append("--enable-component-metric-actions")
    if config.enable_component_metric_merge_judge:
        command.append("--enable-component-metric-merge-judge")
    if config.enable_gap_add_actions:
        command.append("--enable-gap-add-actions")
    if config.stop_after_architecture_refinement:
        command.append("--stop-after-architecture-refinement")
    if config.action_refinement_stop_on_stable:
        command.append("--action-refinement-stop-on-stable")
    if config.action_refinement_save_stops_component:
        command.append("--action-refinement-save-stops-component")
    return command


def build_redacted_command(command: List[str]) -> List[str]:
    redacted = list(command)
    for idx, token in enumerate(redacted):
        if token == "--api-key" and idx + 1 < len(redacted):
            redacted[idx + 1] = "<redacted>"
    return redacted


def write_launch_script(command: List[str], config: LauncherConfig, paths: RunPaths) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(config.workspace_dir))}",
        f"exec {shlex.join(command)}",
        "",
    ]
    paths.launch_script.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(paths.launch_script, 0o755)


def write_manifest(config: LauncherConfig, paths: RunPaths, command: List[str]) -> None:
    manifest = {
        "repo": config.repo,
        "run_label": config.run_label,
        "run_ts": config.run_ts,
        "workspace_dir": str(config.workspace_dir),
        "baseline_repo_dir": str(paths.baseline_repo_dir),
        "run_dir": str(paths.run_dir),
        "repo_root": str(paths.repo_root),
        "output_dir": str(paths.output_dir),
        "requirements_file": str(paths.requirements_file),
        "readme_req_source": str(paths.readme_req_source),
        "requirements_source": str(config.requirements_source.resolve()) if config.requirements_source else "",
        "local_readme_req": str(paths.local_readme_req),
        "launch_script": str(paths.launch_script),
        "log_file": str(paths.log_file),
        "pid_file": str(paths.pid_file),
        "command": build_redacted_command(command),
        "metric_config": {
            "action_refinement_rounds": config.action_refinement_rounds,
            "action_refinement_stop_on_stable": config.action_refinement_stop_on_stable,
            "action_refinement_save_stops_component": config.action_refinement_save_stops_component,
            "enable_component_metric_actions": config.enable_component_metric_actions,
            "enable_component_metric_merge_judge": config.enable_component_metric_merge_judge,
            "component_metric_split_cohesion_threshold": config.component_metric_split_cohesion_threshold,
            "component_metric_split_min_subrequirements": config.component_metric_split_min_subrequirements,
            "component_split_min_confidence": config.component_split_min_confidence,
            "component_metric_merge_max_small_subrequirements": config.component_metric_merge_max_small_subrequirements,
            "enable_gap_add_actions": config.enable_gap_add_actions,
            "gap_add_proposal_threshold": config.gap_add_proposal_threshold,
            "gap_add_component_threshold": config.gap_add_component_threshold,
            "gap_add_requirement_threshold": config.gap_add_requirement_threshold,
            "stop_after_architecture_refinement": config.stop_after_architecture_refinement,
        },
    }
    paths.manifest_file.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")


def write_status_file(paths: RunPaths) -> None:
    lines = [
        f"[prepared] run_dir={paths.run_dir}",
        f"[prepared] output_dir={paths.output_dir}",
        f"[prepared] launch_script={paths.launch_script}",
        f"[prepared] log_file={paths.log_file}",
    ]
    paths.status_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_launcher_run(config: LauncherConfig) -> Dict[str, str]:
    paths = resolve_run_paths(config)
    validate_inputs(config, paths)
    copy_baseline_repo(config, paths)
    prune_reused_artifacts(paths)
    command = build_run_agents_command(config, paths)
    write_launch_script(command, config, paths)
    write_manifest(config, paths, command)
    write_status_file(paths)
    return {
        "repo": config.repo,
        "run_dir": str(paths.run_dir),
        "repo_root": str(paths.repo_root),
        "output_dir": str(paths.output_dir),
        "requirements_file": str(paths.requirements_file),
        "readme_req": str(paths.local_readme_req),
        "launch_script": str(paths.launch_script),
        "manifest_file": str(paths.manifest_file),
        "status_file": str(paths.status_file),
        "log_file": str(paths.log_file),
        "pid_file": str(paths.pid_file),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a fresh single-repo metric-guided codegen run.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--workspace-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=ROOT / "repo_input",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "metric_guided_codegen")
    parser.add_argument("--run-label", default="metric_guided_codegen")
    parser.add_argument("--run-ts", default=_default_run_ts())
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--base-url", default="https://api.qingyuntop.top/v1")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--postcheck-max-workers", type=int, default=4)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--parent-codegen-dag-source", default="dependency")
    parser.add_argument("--action-refinement-rounds", type=int, default=5)
    parser.add_argument("--component-metric-split-cohesion-threshold", type=float, default=2.0 / 3.0)
    parser.add_argument("--component-metric-split-min-subrequirements", type=int, default=3)
    parser.add_argument("--component-split-min-confidence", type=float, default=0.70)
    parser.add_argument("--component-metric-merge-max-small-subrequirements", type=int, default=1)
    parser.add_argument("--readme-req-source", type=Path, default=None)
    parser.add_argument("--requirements-source", type=Path, default=None)
    parser.add_argument("--no-use-processes", action="store_true")
    parser.add_argument("--no-retry-empty-generated-components", action="store_true")
    parser.add_argument("--no-enable-component-metric-actions", action="store_true")
    parser.add_argument("--enable-component-metric-merge-judge", action="store_true")
    parser.add_argument("--action-refinement-stop-on-stable", action="store_true")
    parser.add_argument("--action-refinement-save-stops-component", action="store_true")
    parser.add_argument("--enable-gap-add-actions", action="store_true")
    parser.add_argument("--gap-add-proposal-threshold", type=float, default=0.55)
    parser.add_argument("--gap-add-component-threshold", type=float, default=0.74)
    parser.add_argument("--gap-add-requirement-threshold", type=float, default=0.82)
    parser.add_argument("--stop-after-architecture-refinement", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LauncherConfig:
    return LauncherConfig(
        repo=str(args.repo).strip(),
        workspace_dir=args.workspace_dir.expanduser().resolve(),
        baseline_root=args.baseline_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        run_label=str(args.run_label).strip(),
        run_ts=str(args.run_ts).strip(),
        python_bin=args.python_bin.expanduser().resolve(),
        base_url=str(args.base_url).strip(),
        api_key=str(args.api_key).strip(),
        model=str(args.model).strip(),
        reasoning_effort=str(args.reasoning_effort).strip(),
        max_workers=int(args.max_workers),
        postcheck_max_workers=int(args.postcheck_max_workers),
        log_level=str(args.log_level).strip(),
        parent_codegen_dag_source=str(args.parent_codegen_dag_source).strip(),
        action_refinement_rounds=max(1, int(args.action_refinement_rounds)),
        action_refinement_stop_on_stable=bool(args.action_refinement_stop_on_stable),
        action_refinement_save_stops_component=bool(args.action_refinement_save_stops_component),
        use_processes=not bool(args.no_use_processes),
        enable_component_metric_actions=not bool(args.no_enable_component_metric_actions),
        enable_component_metric_merge_judge=bool(args.enable_component_metric_merge_judge),
        component_metric_split_cohesion_threshold=float(args.component_metric_split_cohesion_threshold),
        component_metric_split_min_subrequirements=max(1, int(args.component_metric_split_min_subrequirements)),
        component_split_min_confidence=float(args.component_split_min_confidence),
        component_metric_merge_max_small_subrequirements=max(1, int(args.component_metric_merge_max_small_subrequirements)),
        retry_empty_generated_components=not bool(args.no_retry_empty_generated_components),
        readme_req_source=args.readme_req_source.expanduser().resolve() if args.readme_req_source else None,
        requirements_source=args.requirements_source.expanduser().resolve() if args.requirements_source else None,
        enable_gap_add_actions=bool(args.enable_gap_add_actions),
        gap_add_proposal_threshold=float(args.gap_add_proposal_threshold),
        gap_add_component_threshold=float(args.gap_add_component_threshold),
        gap_add_requirement_threshold=float(args.gap_add_requirement_threshold),
        stop_after_architecture_refinement=bool(args.stop_after_architecture_refinement),
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)
    summary = prepare_launcher_run(config)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
