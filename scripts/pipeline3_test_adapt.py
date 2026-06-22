#!/usr/bin/env python3
"""
Pipeline Stage 3: test adaptation and execution.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from eval_task_pipeline import Candidate

IGNORE_IMPORT_NAMES = {
    "pytest",
    "np",
    "numpy",
    "pd",
    "pandas",
    "os",
    "sys",
    "re",
    "math",
    "pathlib",
    "typing",
    "json",
}


def choose_import_target(
    test_file: Path,
    test_functions: List[str],
    class_name: Optional[str],
) -> Optional[str]:
    try:
        source = test_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports[alias.asname or alias.name] = alias.name
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name] = alias.name

    usage = Counter()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in test_functions:
            usage.update(find_name_usage(node))
        if isinstance(node, ast.ClassDef) and class_name and node.name == class_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in test_functions:
                    usage.update(find_name_usage(sub))

    for name, _ in usage.most_common():
        if name in imports and name not in IGNORE_IMPORT_NAMES:
            return name
    return None


def find_name_usage(node: ast.AST) -> Counter:
    counter = Counter()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            counter[sub.id] += 1
    return counter


def module_path_from_file(repo_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(repo_root)
    if rel.name == "__init__.py":
        rel = rel.parent
    module = ".".join(rel.with_suffix("").parts)
    return module


def adapt_test_code(
    test_file: Path,
    output_dir: Path,
    candidate: "Candidate",
    task: Dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = test_file.read_text(encoding="utf-8")
    class_name = None
    module_field = task.get("module", "")
    if module_field.startswith("class "):
        class_name = module_field.split(" ", 1)[1]

    target_name = choose_import_target(test_file, task.get("functions", []), class_name)
    if target_name:
        module_path = module_path_from_file(task["_repo_root"], candidate.file_path)
        import_re = re.compile(rf"^(from\s+[\w\.]+\s+import\s+.*\b{target_name}\b.*)$", re.M)
        source = import_re.sub(f"from {module_path} import {candidate.name}", source)
        source = re.sub(rf"\b{re.escape(target_name)}\b", candidate.name, source)

    adapted_path = output_dir / f"adapted_{test_file.name}"
    adapted_path.write_text(source, encoding="utf-8")
    return adapted_path


def run_pytest(repo_root: Path, test_path: Path, functions: List[str]) -> Dict[str, Any]:
    args = [sys.executable, "-m", "pytest", str(test_path)]
    if functions:
        expr = " or ".join(functions)
        args += ["-k", expr]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        args,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }
