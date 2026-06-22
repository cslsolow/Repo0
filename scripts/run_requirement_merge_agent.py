#!/usr/bin/env python3
"""CLI runner for RequirementMergeAgent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import RequirementMergeAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge redundant requirements from a requirements JSON file.")
    parser.add_argument("--input", type=Path, required=True, help="Path to requirements JSON (list or {requirements:[...]})")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: <input_dir>/requirements_merged.json)")
    parser.add_argument("--base-url", type=str, default="", help="LLM base URL")
    parser.add_argument("--api-key", type=str, default="", help="LLM API key (optional; if missing, fallback merge is used)")
    parser.add_argument("--model", type=str, default="", help="LLM model")
    parser.add_argument("--reasoning-effort", type=str, default="", help="Optional reasoning_effort to pass through to the LLM API.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = (args.output.resolve() if args.output else input_path.parent / "requirements_merged.json")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    api_config = {}
    if args.api_key:
        api_config = {
            "base_url": args.base_url,
            "api_key": args.api_key,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
        }

    agent = RequirementMergeAgent(api_config=api_config, output_dir=str(output_path.parent))
    result = agent.merge_requirements(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Requirement Merge Result ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    stats = result.get("stats", {})
    print(
        "Stats: input={input_count}, merged_groups={merged_group_count}, merged_sources={merged_source_count}, output={output_count}".format(
            input_count=stats.get("input_count", 0),
            merged_group_count=stats.get("merged_group_count", 0),
            merged_source_count=stats.get("merged_source_count", 0),
            output_count=stats.get("output_count", 0),
        )
    )
    print("Merged mapping entries:", len(result.get("merged_name_mapping", [])))


if __name__ == "__main__":
    main()

