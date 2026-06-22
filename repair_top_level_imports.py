#!/usr/bin/env python3
"""Standalone top-level import repair for existing Repo0 generated repos.

This script does NOT re-enter the generation pipeline. It only runs package-level
import smoke tests against existing ``agents_output/generated_code`` and applies
targeted import postcheck fixes through ``CodeGeneratorAgent.postcheck_package_modules``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair top-level package imports for an existing generated repo.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="agents_output directory")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--import-postcheck-max-fix-attempts", type=int, default=10)
    parser.add_argument("--package-postcheck-max-fix-attempts", type=int, default=10)
    parser.add_argument(
        "--init-export-lazy-imports",
        action="store_true",
        help="Rewrite generated eager __init__.py imports to lazy __getattr__ imports before postcheck.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def discover_top_level_package_modules(generated_root: Path) -> List[str]:
    modules: List[str] = []
    if not generated_root.is_dir():
        return modules
    for child in sorted(generated_root.iterdir()):
        if child.is_dir() and (child / "__init__.py").is_file():
            modules.append(child.name)
    return modules


def rewrite_generated_init_to_lazy(init_file: Path) -> bool:
    text = init_file.read_text(encoding="utf-8")
    if "# This file is generated to provide stable package/subpackage imports." not in text:
        return False
    if "def __getattr__(name):" in text:
        return False

    lazy_modules: Dict[str, str] = {}
    lazy_symbols: Dict[str, str] = {}
    kept_lines: List[str] = []
    changed = False

    for line in text.splitlines():
        stripped = line.strip()
        module_match = re.fullmatch(r"from \. import ([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)", stripped)
        symbol_match = re.fullmatch(r"from \.([A-Za-z_][A-Za-z0-9_]*) import (.+)", stripped)
        if module_match:
            for name in [part.strip() for part in module_match.group(1).split(",")]:
                if name:
                    lazy_modules[name] = f".{name}"
            changed = True
            continue
        if symbol_match:
            module = symbol_match.group(1)
            for name in [part.strip() for part in symbol_match.group(2).split(",")]:
                if name and " as " not in name:
                    lazy_symbols[name] = f".{module}"
            changed = True
            continue
        kept_lines.append(line)

    if not changed:
        return False

    insert_at = 0
    for idx, line in enumerate(kept_lines):
        if line.startswith("# This file is generated"):
            insert_at = idx + 1
            break

    lazy_block = [
        "",
        "import importlib as _importlib",
        "",
        "_LAZY_MODULES = {",
    ]
    for name, module in sorted(lazy_modules.items()):
        lazy_block.append(f'    "{name}": "{module}",')
    lazy_block.extend(["}", "_LAZY_SYMBOLS = {"])
    for name, module in sorted(lazy_symbols.items()):
        lazy_block.append(f'    "{name}": "{module}",')
    lazy_block.extend(
        [
            "}",
            "",
            "def __getattr__(name):",
            "    if name in _LAZY_MODULES:",
            "        value = _importlib.import_module(_LAZY_MODULES[name], __name__)",
            "        globals()[name] = value",
            "        return value",
            "    if name in _LAZY_SYMBOLS:",
            "        module = _importlib.import_module(_LAZY_SYMBOLS[name], __name__)",
            "        value = getattr(module, name)",
            "        globals()[name] = value",
            "        return value",
            "    raise AttributeError(f\"module {__name__!r} has no attribute {name!r}\")",
        ]
    )
    new_lines = kept_lines[:insert_at] + lazy_block + kept_lines[insert_at:]
    init_file.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    return True


def rewrite_generated_inits_to_lazy(generated_root: Path) -> List[str]:
    rewritten: List[str] = []
    for init_file in sorted(generated_root.rglob("__init__.py")):
        if rewrite_generated_init_to_lazy(init_file):
            rewritten.append(str(init_file))
    return rewritten


def load_implemented_context(workspace: Path, output_dir: Path) -> str:
    sys.path.insert(0, str(workspace))
    from agents.cognitive.memory import MemoryAgent  # imported lazily

    memory_path = output_dir / "memory.json"
    if not memory_path.exists():
        return ""
    try:
        agent = MemoryAgent(workspace)
        agent.load_snapshot(memory_path)
        return agent.format_implementations_for_prompt(status_filter="implemented")
    except Exception as exc:  # pragma: no cover - best effort context only
        logging.warning("Failed to load memory context from %s: %s", memory_path, exc)
        return ""


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    workspace = args.workspace.resolve()
    output_dir = args.output.resolve()
    generated_root = output_dir / "generated_code"
    if not generated_root.is_dir():
        raise SystemExit(f"generated_code directory does not exist: {generated_root}")

    sys.path.insert(0, str(workspace))
    from agents.coding.code_generator import CodeGeneratorAgent  # imported lazily

    package_modules = discover_top_level_package_modules(generated_root)
    if not package_modules:
        logging.warning("No top-level packages found under %s", generated_root)
        report = {
            "repo": args.repo,
            "generated_root": str(generated_root),
            "package_modules": [],
            "passed": True,
            "modules": [],
            "note": "No top-level packages discovered.",
        }
        report_path = output_dir / "top_level_import_repair_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(report_path)
        return 0

    lazy_rewritten: List[str] = []
    if args.init_export_lazy_imports:
        lazy_rewritten = rewrite_generated_inits_to_lazy(generated_root)
        logging.info("Lazy init rewrite completed: rewritten=%d", len(lazy_rewritten))

    api_config: Dict[str, Any] = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "repo": args.repo,
        "import_postcheck_max_fix_attempts": max(0, int(args.import_postcheck_max_fix_attempts)),
        "package_postcheck_max_fix_attempts": max(0, int(args.package_postcheck_max_fix_attempts)),
    }

    implemented_context = load_implemented_context(workspace, output_dir)
    code_generator = CodeGeneratorAgent(api_config=api_config, output_dir=str(output_dir))

    started_at = time.perf_counter()
    result = code_generator.postcheck_package_modules(
        package_modules=package_modules,
        repo_root=generated_root,
        implemented_components_context=implemented_context,
        max_fix_attempts=max(0, int(args.package_postcheck_max_fix_attempts)),
    )
    elapsed = time.perf_counter() - started_at

    report = {
        "repo": args.repo,
        "generated_root": str(generated_root),
        "package_modules": package_modules,
        "passed": bool(result.get("passed")) if isinstance(result, dict) else False,
        "elapsed_sec": round(elapsed, 3),
        "lazy_init_rewritten": lazy_rewritten,
        "result": result,
    }
    report_path = output_dir / "top_level_import_repair_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    logging.info(
        "Top-level import repair finished for %s: packages=%d passed=%s report=%s",
        args.repo,
        len(package_modules),
        report["passed"],
        report_path,
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
