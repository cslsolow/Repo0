#!/usr/bin/env python3
"""End-to-end smoke test for the Docker-backed TDD pytest path."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from agents.coding.code_generator import CodeGeneratorAgent

    sandbox_root = Path(tempfile.mkdtemp(prefix="docker_tdd_smoke_root_"))
    project_root = Path(tempfile.mkdtemp(prefix="docker_tdd_smoke_proj_"))

    (project_root / "setup.py").write_text(
        "from setuptools import setup; setup(name='docker-tdd-smoke', version='0.0.0', packages=[])\n",
        encoding="utf-8",
    )

    (sandbox_root / "demo_component.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    tests_dir = sandbox_root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_demo_component.py").write_text(
        "from demo_component import add\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    agent = CodeGeneratorAgent(
        {
            "tdd_docker_image": "repo0-codegen-tdd:latest",
            "tdd_pip_project_root": str(project_root),
            "tdd_pytest_timeout": 60,
            "tdd_pip_timeout": 60,
        }
    )
    rc, out = agent._run_pytest_in_docker(
        sandbox_root,
        "tests/test_demo_component.py",
        heuristic_pip_specs=[],
    )
    print(f"docker_tdd_smoke rc={rc}")
    print(f"sandbox_root={sandbox_root}")
    print(f"project_root={project_root}")
    print("--- docker_tdd_smoke output ---")
    print(out.strip())
    print("--- end docker_tdd_smoke output ---")
    print("note: smoke artifacts are kept on disk because docker may create root-owned files in the bind mounts.")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
