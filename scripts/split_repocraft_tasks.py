#!/usr/bin/env python3
"""Split a RepoCraft tasks JSON file into deterministic shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split RepoCraft tasks into shards.")
    parser.add_argument("--input", required=True, help="Path to source tasks JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory to write shard files into")
    parser.add_argument("--num-shards", type=int, default=4, help="Number of shards to create")
    parser.add_argument(
        "--strategy",
        choices=["round_robin", "contiguous"],
        default="round_robin",
        help="How to assign tasks to shards",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(tasks, dict):
        tasks = [tasks]
    if not isinstance(tasks, list):
        raise TypeError(f"Expected list/dict tasks JSON, got {type(tasks).__name__}")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")

    shards: list[list[dict]] = [[] for _ in range(args.num_shards)]
    if args.strategy == "round_robin":
        for idx, task in enumerate(tasks):
            shards[idx % args.num_shards].append(task)
    else:
        chunk = (len(tasks) + args.num_shards - 1) // args.num_shards
        for shard_idx in range(args.num_shards):
            start = shard_idx * chunk
            end = start + chunk
            shards[shard_idx] = tasks[start:end]

    summary = {
        "input": str(input_path),
        "num_shards": args.num_shards,
        "strategy": args.strategy,
        "total_tasks": len(tasks),
        "shards": [],
    }

    stem = input_path.stem
    for shard_idx, shard_tasks in enumerate(shards):
        shard_path = output_dir / f"{stem}.shard{shard_idx}.json"
        shard_path.write_text(json.dumps(shard_tasks, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["shards"].append(
            {
                "shard_index": shard_idx,
                "path": str(shard_path),
                "task_count": len(shard_tasks),
            }
        )

    (output_dir / f"{stem}.shards.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
