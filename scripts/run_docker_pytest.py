#!/usr/bin/env python3
"""Build a Docker image and run pytest automatically in a container."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Docker image and run pytest in the container",
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=Path.cwd(),
        help="Docker build context (default: current directory)",
    )
    parser.add_argument(
        "--dockerfile",
        type=Path,
        default=None,
        help="Path to Dockerfile (default: <context>/Dockerfile)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Docker image name, e.g. myproj:test",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default="",
        help="Optional container name",
    )
    parser.add_argument(
        "--mount-src",
        type=Path,
        default=None,
        help="Host path to mount into container (default: context)",
    )
    parser.add_argument(
        "--mount-dst",
        type=str,
        default="/workspace",
        help="Container mount destination (default: /workspace)",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="/workspace",
        help="Working directory in container (default: /workspace)",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="python",
        help="Python executable in container (default: python)",
    )
    parser.add_argument(
        "--pytest-target",
        type=str,
        default="tests",
        help="Pytest target path/module in container (default: tests)",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable for container, format KEY=VALUE (repeatable)",
    )
    parser.add_argument(
        "--build-arg",
        action="append",
        default=[],
        help="Docker build-arg, format KEY=VALUE (repeatable)",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="",
        help="Docker network for docker run",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Pass --no-cache to docker build",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pass --pull to docker build",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip docker build and directly run pytest in existing image",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build image, do not run pytest",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="",
        help="Container user, e.g. 1000:1000",
    )
    parser.add_argument(
        "--no-rm",
        action="store_true",
        help="Do not auto-remove container after test run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed commands",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to pytest. Put after '--'.",
    )
    args = parser.parse_args()

    if args.build_only and args.skip_build:
        parser.error("--build-only and --skip-build cannot be used together")

    return args


def _run(cmd: List[str], verbose: bool = False) -> int:
    if verbose:
        print("+", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def _ensure_docker(verbose: bool) -> None:
    rc = _run(["docker", "--version"], verbose=verbose)
    if rc != 0:
        raise SystemExit("docker is not available in current environment")


def _build_cmd(args: argparse.Namespace) -> List[str]:
    context = args.context.resolve()
    dockerfile = args.dockerfile.resolve() if args.dockerfile else context / "Dockerfile"

    cmd = ["docker", "build", "-t", args.image, "-f", str(dockerfile)]
    if args.no_cache:
        cmd.append("--no-cache")
    if args.pull:
        cmd.append("--pull")
    for item in args.build_arg:
        cmd.extend(["--build-arg", item])
    cmd.append(str(context))
    return cmd


def _normalize_pytest_args(raw: List[str]) -> List[str]:
    if not raw:
        return []
    if raw[0] == "--":
        return raw[1:]
    return raw


def _run_cmd(args: argparse.Namespace) -> List[str]:
    mount_src = (args.mount_src or args.context).resolve()

    cmd = ["docker", "run"]
    if not args.no_rm:
        cmd.append("--rm")
    if args.container_name:
        cmd.extend(["--name", args.container_name])
    if args.network:
        cmd.extend(["--network", args.network])
    if args.user:
        cmd.extend(["-u", args.user])

    cmd.extend(["-v", f"{mount_src}:{args.mount_dst}"])
    cmd.extend(["-w", args.workdir])

    for item in args.env:
        cmd.extend(["-e", item])

    cmd.append(args.image)

    pytest_args = _normalize_pytest_args(args.pytest_args)
    test_cmd = [args.python_bin, "-m", "pytest", args.pytest_target, *pytest_args]
    cmd.extend(test_cmd)
    return cmd


def main() -> None:
    args = _parse_args()
    _ensure_docker(args.verbose)

    if not args.skip_build:
        build_cmd = _build_cmd(args)
        rc = _run(build_cmd, verbose=args.verbose)
        if rc != 0:
            raise SystemExit(rc)

    if args.build_only:
        print(f"docker image built: {args.image}")
        return

    run_cmd = _run_cmd(args)
    rc = _run(run_cmd, verbose=args.verbose)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
