#!/usr/bin/env python3
"""Run full-task function-level pytest audit inside a persistent Docker container."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-task function-level audit in Docker.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-root", required=True, help="Generated repo root mounted into container.")
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional function limit for smoke runs.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--runner",
        choices=["pytest", "django-runtests"],
        default="pytest",
        help="Execution backend inside container.",
    )
    parser.add_argument(
        "--django-settings",
        default="test_sqlite",
        help="Settings module used when --runner=django-runtests.",
    )
    return parser.parse_args()


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def iter_functions(tasks: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, str]]:
    for task in tasks:
        task_id = str(task.get("id") or "").strip()
        file_path = str(task.get("file") or "").strip()
        module = str(task.get("module") or "").strip()
        functions = task.get("functions", []) or []
        if module.startswith("class "):
            cls = module.split("class ", 1)[1].strip()
            for fn in functions:
                yield {
                    "task_id": task_id,
                    "file": file_path,
                    "module": module,
                    "function": str(fn),
                    "nodeid": f"{file_path}::{cls}::{fn}",
                }
        else:
            for fn in functions:
                yield {
                    "task_id": task_id,
                    "file": file_path,
                    "module": module,
                    "function": str(fn),
                    "nodeid": f"{file_path}::{fn}",
                }


def django_label(file_path: str, module: str, function: str) -> str:
    rel = file_path.replace("\\", "/")
    if rel.startswith("tests/"):
        rel = rel[len("tests/") :]
    if rel.endswith(".py"):
        rel = rel[:-3]
    label = rel.replace("/", ".")
    if module.startswith("class "):
        cls = module.split("class ", 1)[1].strip()
        return f"{label}.{cls}.{function}"
    return f"{label}.{function}"


def run(cmd: List[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def classify(returncode: int, output: str) -> str:
    text = output.lower()
    if "1 passed" in text or " passed" in text:
        if " failed" not in text and " error" not in text:
            if " skipped" in text and "passed" not in text:
                return "skipped"
            return "passed"
    if "xfailed" in text or "xfail" in text:
        return "xfailed"
    if "skipped" in text and "passed" not in text:
        return "skipped"
    if "found no collectors" in text:
        return "collector_error"
    if "modulenotfounderror" in text or "importerror" in text:
        return "import_error"
    if returncode == 0:
        return "passed"
    return "failed"


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    tasks_path = Path(args.tasks_jsonl).resolve()
    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(tasks_path)
    functions = list(iter_functions(tasks))
    if args.limit > 0:
        functions = functions[: args.limit]

    run(["docker", "rm", "-f", args.container_name], timeout=60)
    run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            args.container_name,
            "-v",
            f"{repo_root}:/repo",
            "-w",
            "/repo",
            args.image,
            "sleep",
            "infinity",
        ],
        timeout=60,
    )

    results: List[Dict[str, Any]] = []
    started = time.time()
    try:
        for idx, row in enumerate(functions, start=1):
            proc = run(
                (
                    [
                        "docker",
                        "exec",
                        "-e",
                        "PYTHONPATH=/repo",
                        args.container_name,
                        "python",
                        "tests/runtests.py",
                        f"--settings={args.django_settings}",
                        django_label(row["file"], row["module"], row["function"]),
                    ]
                    if args.runner == "django-runtests"
                    else [
                        "docker",
                        "exec",
                        args.container_name,
                        "python",
                        "-m",
                        "pytest",
                        "-q",
                        row["nodeid"],
                    ]
                ),
                timeout=args.timeout,
            )
            output = proc.stdout or ""
            result = dict(row)
            result["returncode"] = proc.returncode
            result["status"] = classify(proc.returncode, output)
            result["output_head"] = output[:4000]
            results.append(result)

            if idx % 50 == 0:
                out_path.write_text(
                    json.dumps(
                        {
                            "repo": args.repo,
                            "completed": idx,
                            "total": len(functions),
                            "elapsed_sec": round(time.time() - started, 2),
                            "results": results,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    finally:
        run(["docker", "rm", "-f", args.container_name], timeout=60)

    summary = {
        "repo": args.repo,
        "completed": len(functions),
        "total": len(functions),
        "elapsed_sec": round(time.time() - started, 2),
        "status_counts": {},
        "results": results,
    }
    counts: Dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary["status_counts"] = counts
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
