#!/usr/bin/env python3
"""Standalone CLI for the LLM-driven localization agent."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.localization_pipeline_agent import LLMToolLocalizationAgent  # noqa: E402
from agents.llm_client import LLMClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM localization agent")
    parser.add_argument("--repo", type=Path, required=True, help="Repository root")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--task", type=str, default="", help="Task description text")
    parser.add_argument("--task-file", type=Path, default=None, help="Path to task text file")
    parser.add_argument("--max-steps", type=int, default=12, help="Max tool steps")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per LLM call")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--base-url", type=str, required=True, help="LLM base URL")
    parser.add_argument("--api-key", type=str, required=True, help="LLM API key")
    parser.add_argument("--model", type=str, required=True, help="LLM model")
    parser.add_argument("--reasoning-effort", type=str, default="", help="Optional reasoning_effort to pass through to the LLM API.")
    return parser.parse_args()


def setup_logging(debug: bool) -> None:
    fmt = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.WARNING, format=fmt, datefmt=datefmt)

    agent_logger = logging.getLogger("localization_agent")
    agent_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    agent_logger.handlers = [handler]
    agent_logger.propagate = False

    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and name != "localization_agent":
            logger.setLevel(logging.WARNING)


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    if not args.task and not args.task_file:
        raise SystemExit("Provide --task or --task-file.")
    task_text = args.task
    if args.task_file:
        task_text = args.task_file.read_text(encoding="utf-8")

    llm_client = LLMClient(
        {"base_url": args.base_url, "api_key": args.api_key, "model": args.model, "reasoning_effort": args.reasoning_effort},
        str(args.output.parent),
        agent_name="localization_agent",
    )
    agent = LLMToolLocalizationAgent(
        repo_root=args.repo,
        llm_client=llm_client,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
    )
    results = agent.run(task_text)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
