#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import ComponentMergeAgent, ComponentSplitAgent, RequirementDAG
from run_agents import (
    _build_default_empty_actions,
    _build_metric_merge_judge,
    _rebuild_flattened_architectures,
    augment_actions_with_component_metrics,
    build_gap_add_stage_inputs,
    choose_actions_for_architectures,
    run_action_feedback_rounds,
    run_gap_addition_stage,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run action refinement directly from an existing architectures.json snapshot.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--input-dir", type=Path, required=True, help="agents_output directory containing architectures.json and decomposed_dag.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="target agents_output directory for refined artifacts")
    parser.add_argument("--requirements-file", type=Path, required=True)
    parser.add_argument("--req-path", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--postcheck-max-workers", type=int, default=4)
    parser.add_argument("--action-refinement-rounds", type=int, default=5)
    parser.add_argument("--action-refinement-stop-on-stable", action="store_true")
    parser.add_argument("--action-refinement-save-stops-component", action="store_true")
    parser.add_argument("--enable-component-metric-actions", action="store_true")
    parser.add_argument("--enable-component-metric-merge-judge", action="store_true")
    parser.add_argument("--component-metric-split-cohesion-threshold", type=float, default=2.0 / 3.0)
    parser.add_argument("--component-metric-split-min-subrequirements", type=int, default=3)
    parser.add_argument("--component-split-min-confidence", type=float, default=0.70)
    parser.add_argument("--component-metric-merge-max-small-subrequirements", type=int, default=1)
    parser.add_argument("--enable-gap-add-actions", action="store_true")
    parser.add_argument("--gap-add-proposal-threshold", type=float, default=0.55)
    parser.add_argument("--gap-add-component-threshold", type=float, default=0.74)
    parser.add_argument("--gap-add-requirement-threshold", type=float, default=0.82)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_from_existing_architecture(
    *,
    args: argparse.Namespace,
    choose_actions_fn: Callable[..., List[dict]] = choose_actions_for_architectures,
    feedback_rounds_fn: Callable[..., Any] = run_action_feedback_rounds,
    gap_add_stage_fn: Callable[..., Any] = run_gap_addition_stage,
) -> Dict[str, Any]:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Strict refine entrypoint starting: repo=%s input_dir=%s output_dir=%s", args.repo, input_dir, output_dir)

    architectures_path = input_dir / "architectures.json"
    decomposed_dag_path = input_dir / "decomposed_dag.json"
    if not architectures_path.exists():
        raise FileNotFoundError(f"Missing architectures snapshot: {architectures_path}")
    if not decomposed_dag_path.exists():
        raise FileNotFoundError(f"Missing decomposed DAG: {decomposed_dag_path}")

    architectures = load_json(architectures_path)
    decomposed_dag = RequirementDAG.from_dict(load_json(decomposed_dag_path))
    logging.info(
        "Loaded existing architecture inputs: parents=%d decomposed_nodes=%d",
        len(architectures) if isinstance(architectures, list) else 0,
        len(decomposed_dag.nodes),
    )

    api_config = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "repo": args.repo,
    }
    setattr(args, "api_config", api_config)

    component_merge_agent = ComponentMergeAgent(
        api_config=api_config,
        output_dir=str(output_dir),
    )
    component_split_agent = ComponentSplitAgent(
        api_config=api_config,
        output_dir=str(output_dir),
        enable_llm_split=True,
        split_min_confidence=float(args.component_split_min_confidence),
    )
    logging.info(
        "Action refinement config: rounds=%d metric_actions=%s gap_add=%s max_workers=%d",
        int(args.action_refinement_rounds),
        bool(args.enable_component_metric_actions),
        bool(args.enable_gap_add_actions),
        int(args.max_workers),
    )

    metric_merge_judge = None
    if bool(args.enable_component_metric_merge_judge):
        metric_merge_judge = _build_metric_merge_judge(component_merge_agent)

    if bool(args.enable_component_metric_actions):
        logging.info("Metric-guided structural mode enabled: skipping strategist baseline and building initial actions from metrics")
        all_actions = _build_default_empty_actions(architectures)
        all_actions, component_metric_action_report = augment_actions_with_component_metrics(
            architectures=architectures,
            actions=all_actions,
            decomposed_dag=decomposed_dag,
            split_cohesion_threshold=float(args.component_metric_split_cohesion_threshold),
            split_min_subrequirements=int(args.component_metric_split_min_subrequirements),
            merge_judge=metric_merge_judge,
            merge_max_small_subrequirements=int(args.component_metric_merge_max_small_subrequirements),
        )
        save_json(component_metric_action_report, output_dir / "component_metric_action_report.json")
    else:
        logging.info("Choosing initial actions from existing architectures via strategist")
        all_actions = choose_actions_fn(
            architectures=architectures,
            api_config=api_config,
            output_dir=str(output_dir),
            max_workers=int(args.max_workers),
        )

    save_json(all_actions, output_dir / "actions.json")
    non_empty_actions = sum(1 for row in all_actions if isinstance(row, dict) and row.get("actions"))
    logging.info("Initial actions saved: parents=%d non_empty_parents=%d", len(all_actions), non_empty_actions)

    logging.info("Starting action feedback rounds from existing architecture snapshot")
    architectures, all_actions, action_refinement_report = feedback_rounds_fn(
        architectures=architectures,
        initial_actions=all_actions,
        component_merge_agent=component_merge_agent,
        component_split_agent=component_split_agent,
        rounds=int(args.action_refinement_rounds),
        api_config=api_config,
        output_dir=output_dir,
        max_workers=int(args.max_workers),
        stop_on_stable=bool(args.action_refinement_stop_on_stable),
        save_stops_component=bool(args.action_refinement_save_stops_component),
        enable_cross_requirement_merge=False,
        save_round_artifacts=True,
        existing_generated_entries=[],
        tdd_rewrite_failure_threshold=3,
        decomposed_dag=decomposed_dag,
        enable_metric_actions=bool(args.enable_component_metric_actions),
        metric_split_cohesion_threshold=float(args.component_metric_split_cohesion_threshold),
        metric_split_min_subrequirements=int(args.component_metric_split_min_subrequirements),
        metric_merge_max_small_subrequirements=int(args.component_metric_merge_max_small_subrequirements),
        metric_merge_judge=metric_merge_judge,
    )
    save_json(all_actions, output_dir / "actions.json")
    save_json(action_refinement_report, output_dir / "action_refinement_report.json")
    logging.info(
        "Action refinement complete: components_after=%s merge_groups=%s split_groups=%s",
        action_refinement_report.get("stats", {}).get("components_after"),
        action_refinement_report.get("stats", {}).get("merge_group_count"),
        action_refinement_report.get("stats", {}).get("split_group_count"),
    )

    gap_inputs = build_gap_add_stage_inputs(
        requirements_file=args.requirements_file.expanduser().resolve(),
        req_path=args.req_path.expanduser().resolve(),
        generated_files_path=output_dir / "generated_files.json",
        realization_report_path=output_dir / "component_realization_report.json",
    )
    logging.info("Starting gap-add stage")
    architectures, gap_addition_report = gap_add_stage_fn(
        architectures=architectures,
        args=args,
        output_dir=output_dir,
        input_text=gap_inputs["input_text"],
        requirements_payload=gap_inputs["requirements_payload"],
        generated_entries=gap_inputs["generated_entries"],
        realization_report=gap_inputs["realization_report"],
        component_merge_agent=component_merge_agent,
        component_split_agent=component_split_agent,
    )
    if isinstance(gap_addition_report, dict):
        save_json(gap_addition_report, output_dir / "gap_addition_report.json")
        logging.info(
            "Gap-add stage complete: enabled=%s accepted_count=%s",
            gap_addition_report.get("enabled"),
            gap_addition_report.get("accepted_count"),
        )

    save_json(architectures, output_dir / "architectures.json")
    _rebuild_flattened_architectures(output_dir, architectures)
    logging.info("Saved refined architectures: parents=%d path=%s", len(architectures) if isinstance(architectures, list) else 0, output_dir / "architectures.json")

    return {
        "refined_parent_count": len(architectures) if isinstance(architectures, list) else 0,
        "output_dir": str(output_dir),
        "actions_path": str(output_dir / "actions.json"),
        "report_path": str(output_dir / "action_refinement_report.json"),
        "architectures_path": str(output_dir / "architectures.json"),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    summary = run_from_existing_architecture(args=args)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
