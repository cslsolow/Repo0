#!/usr/bin/env python3
"""Regenerate dependency_graph.json for an existing agents_output directory."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import DependencyGraphAgent, MemoryAgent, RequirementDAG  # noqa: E402
from run_agents import build_components_from_memory, build_requirement_edges, save_json  # noqa: E402


def resolve_artifact_paths(agents_output: Path) -> Dict[str, Path]:
    base = agents_output.expanduser().resolve()
    return {
        "agents_output": base,
        "memory": base / "memory.json",
        "requirement_dag": base / "requirement_dag.json",
        "dependency_graph": base / "dependency_graph.json",
    }


def resolve_output_path(agents_output: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()
    return agents_output.expanduser().resolve() / "dependency_graph.json"


def configure_logging(level: str) -> None:
    resolved = getattr(logging, str(level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_dependency_inputs(paths: Dict[str, Path], workspace_root: Path) -> Dict[str, object]:
    dependency_graph_path = paths["dependency_graph"]
    if dependency_graph_path.exists():
        try:
            cached = json.loads(dependency_graph_path.read_text(encoding="utf-8"))
        except Exception:
            cached = None
        if isinstance(cached, dict):
            components = cached.get("components")
            requirement_edges = cached.get("requirement_edges")
            if isinstance(components, list) and isinstance(requirement_edges, list):
                return {
                    "source": "cached_dependency_graph",
                    "components": components,
                    "requirement_edges": requirement_edges,
                }

    for label in ("memory", "requirement_dag"):
        if not paths[label].exists():
            raise FileNotFoundError(f"Missing required artifact: {paths[label]}")

    memory_agent = MemoryAgent(Path(workspace_root))
    memory_agent.load_snapshot(paths["memory"])
    requirement_dag = RequirementDAG.from_dict(
        json.loads(paths["requirement_dag"].read_text(encoding="utf-8"))
    )
    return {
        "source": "memory_and_requirement_dag",
        "components": build_components_from_memory(memory_agent),
        "requirement_edges": build_requirement_edges(requirement_dag),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate dependency_graph.json from existing agent artifacts.")
    parser.add_argument(
        "--agents-output",
        type=Path,
        required=True,
        help="Path to the agents_output directory.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT,
        help="Repo0 workspace root. Defaults to the current project root.",
    )
    parser.add_argument("--base-url", type=str, default="https://aihubmix.com/v1")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--model", type=str, default="deepseek-v3.2")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path for the regenerated dependency graph JSON. Defaults to <agents-output>/dependency_graph.json.",
    )
    return parser.parse_args()


def build_run_summary(
    *,
    loaded_inputs: Dict[str, object],
    output_path: Path,
    dependency_graph: Dict[str, object],
) -> Dict[str, object]:
    return {
        "input_source": loaded_inputs["source"],
        "components": len(loaded_inputs.get("components", []) or []),
        "input_requirement_edges": len(loaded_inputs.get("requirement_edges", []) or []),
        "dependency_graph_path": str(output_path),
        "component_edges": len((dependency_graph.get("component_edges") or []) if isinstance(dependency_graph, dict) else []),
        "requirement_edges": len((dependency_graph.get("edges") or []) if isinstance(dependency_graph, dict) else []),
        "uncertain_edges": len((dependency_graph.get("uncertain_edges") or []) if isinstance(dependency_graph, dict) else []),
    }


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    paths = resolve_artifact_paths(args.agents_output)
    output_path = resolve_output_path(paths["agents_output"], args.output_path)
    logging.info("Dependency graph regeneration started")
    logging.info("Agents output: %s", paths["agents_output"])
    logging.info("Output path: %s", output_path)
    loaded_inputs = load_dependency_inputs(paths, workspace_root=Path(args.workspace))
    logging.info(
        "Loaded dependency inputs from %s: components=%d requirement_edges=%d",
        loaded_inputs["source"],
        len(loaded_inputs.get("components", []) or []),
        len(loaded_inputs.get("requirement_edges", []) or []),
    )

    api_config = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
    }

    components = loaded_inputs["components"]
    requirement_edges = loaded_inputs["requirement_edges"]
    dependency_agent = DependencyGraphAgent(api_config=api_config, output_dir=str(paths["agents_output"]))
    logging.info("Calling DependencyGraphAgent.build_requirement_dependency_edges ...")
    dependency_graph = dependency_agent.build_requirement_dependency_edges(
        components,
        constraints={
            "must_use_only_component_ids": True,
            "disallow_self_dependency": True,
            "allow_same_requirement_edges": False,
        },
        requirement_edges=requirement_edges,
    )
    logging.info("Dependency graph inference completed")
    payload = {
        "components": components,
        "requirement_edges": requirement_edges,
        "dependency_graph": dependency_graph,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, output_path)
    summary = build_run_summary(
        loaded_inputs=loaded_inputs,
        output_path=output_path,
        dependency_graph=dependency_graph,
    )
    logging.info(
        "Saved dependency graph: component_edges=%d requirement_edges=%d uncertain_edges=%d",
        summary["component_edges"],
        summary["requirement_edges"],
        summary["uncertain_edges"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
