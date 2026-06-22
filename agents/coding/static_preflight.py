"""Static preflight: syntax compile, first-party import paths, symbol presence, ruff F/E9.

Does not execute generated package code on the host (no import side effects).
"""

from __future__ import annotations

import ast
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

RUFF_RULES = "F401,F402,F403,F405,F406,F821,E9"


def _module_dotted_parts(file_path: Path, generated_root: Path) -> List[str]:
    rel = file_path.resolve().relative_to(generated_root.resolve())
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def _relative_import_base(generated_root: Path, file_path: Path, level: int) -> Optional[List[str]]:
    if level <= 0:
        return None
    mod_parts = _module_dotted_parts(file_path, generated_root)
    cur_pkg: List[str]
    if file_path.name == "__init__.py":
        cur_pkg = mod_parts
    else:
        cur_pkg = mod_parts[:-1] if len(mod_parts) >= 1 else []
    up = level - 1
    if up > len(cur_pkg):
        return None
    return cur_pkg[: len(cur_pkg) - up]


def _module_path_from_parts(root: Path, parts: Sequence[str]) -> Optional[Path]:
    if not parts:
        return None
    pkg_dir = root.joinpath(*parts)
    init_py = pkg_dir / "__init__.py"
    if init_py.is_file():
        return init_py
    single = pkg_dir.with_suffix(".py")
    if single.is_file():
        return single
    return None


def _discover_package_roots(generated_root: Path) -> List[str]:
    roots: List[str] = []
    if not generated_root.is_dir():
        return roots
    for child in sorted(generated_root.iterdir()):
        if child.is_dir() and (child / "__init__.py").is_file():
            roots.append(child.name)
    return roots


def _parse_module(path: Path) -> Optional[ast.Module]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("read failed %s: %s", path, exc)
        return None
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError:
        return None


def _resolve_import_from_target(root: Path, file_path: Path, node: ast.ImportFrom) -> Optional[Path]:
    level = node.level or 0
    mod = node.module
    if level == 0:
        if not mod:
            return None
        return _module_path_from_parts(root, mod.split("."))
    base = _relative_import_base(root, file_path, level)
    if not base:
        return None
    extra = mod.split(".") if mod else []
    return _module_path_from_parts(root, base + extra)


def _module_provides_symbol(
    root: Path,
    module_file: Path,
    symbol: str,
    chain: Set[Tuple[str, str]],
    depth: int,
) -> bool:
    if depth > 14:
        return False
    key = (str(module_file.resolve()), symbol)
    if key in chain:
        return False
    chain.add(key)

    tree = _parse_module(module_file)
    if not tree:
        return False

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                return True
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == symbol:
                    return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == symbol:
            return True

    mod_parts = _module_dotted_parts(module_file, root)
    if module_file.name == "__init__.py":
        parent_pkg: List[str] = mod_parts
    else:
        parent_pkg = mod_parts[:-1] if mod_parts else []
    sub = _module_path_from_parts(root, list(parent_pkg) + [symbol])
    if sub is not None and sub.is_file():
        return True

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        tgt = _resolve_import_from_target(root, module_file, node)
        if not tgt:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            if local != symbol:
                continue
            if _module_provides_symbol(root, tgt, alias.name, chain, depth + 1):
                return True

    return False


def _check_imports(
    file_path: Path,
    generated_root: Path,
    package_roots: Set[str],
    issues: List[Dict[str, Any]],
) -> None:
    tree = _parse_module(file_path)
    if not tree:
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] not in package_roots:
                    continue
                tgt = _module_path_from_parts(generated_root, parts)
                if not tgt:
                    issues.append(
                        {
                            "kind": "import_path",
                            "file": str(file_path),
                            "line": getattr(node, "lineno", 0) or 0,
                            "message": f"package path for import {alias.name!r} not found under generated root",
                        }
                    )

        elif isinstance(node, ast.ImportFrom):
            mod = node.module
            level = node.level or 0
            names_list = [a for a in node.names if a.name != "*"]
            if not names_list and node.names and node.names[0].name == "*":
                continue

            if level == 0 and mod:
                parts = mod.split(".")
                if parts[0] not in package_roots:
                    continue
                tgt = _module_path_from_parts(generated_root, parts)
                if not tgt:
                    issues.append(
                        {
                            "kind": "import_path",
                            "file": str(file_path),
                            "line": getattr(node, "lineno", 0) or 0,
                            "message": f"module {mod!r} not found under generated root",
                        }
                    )
                    continue
                for alias in names_list:
                    sym = alias.name
                    chain: Set[Tuple[str, str]] = set()
                    if not _module_provides_symbol(generated_root, tgt, sym, chain, 0):
                        issues.append(
                            {
                                "kind": "missing_symbol",
                                "file": str(file_path),
                                "line": getattr(node, "lineno", 0) or 0,
                                "message": f"cannot resolve {sym!r} from {mod!r} (static)",
                            }
                        )
            elif level > 0:
                base = _relative_import_base(generated_root, file_path, level)
                if not base:
                    issues.append(
                        {
                            "kind": "relative_import",
                            "file": str(file_path),
                            "line": getattr(node, "lineno", 0) or 0,
                            "message": f"invalid relative import level {level}",
                        }
                    )
                    continue
                extra = mod.split(".") if mod else []
                full = base + extra
                tgt = _module_path_from_parts(generated_root, full) if full else None
                if not tgt:
                    display = f"{'.' * level}{mod}" if mod else "." * level
                    issues.append(
                        {
                            "kind": "import_path",
                            "file": str(file_path),
                            "line": getattr(node, "lineno", 0) or 0,
                            "message": f"relative import target {display!r} not found",
                        }
                    )
                    continue
                for alias in names_list:
                    sym = alias.name
                    chain_rel: Set[Tuple[str, str]] = set()
                    if not _module_provides_symbol(generated_root, tgt, sym, chain_rel, 0):
                        display = f"{'.' * level}{mod}" if mod else "." * level
                        issues.append(
                            {
                                "kind": "missing_symbol",
                                "file": str(file_path),
                                "line": getattr(node, "lineno", 0) or 0,
                                "message": f"cannot resolve {sym!r} from relative {display!r} (static)",
                            }
                        )


def _run_ruff(generated_root: Path) -> Tuple[bool, str]:
    ruff = shutil.which("ruff")
    if not ruff:
        return False, ""
    proc = subprocess.run(
        [
            ruff,
            "check",
            str(generated_root),
            "--select",
            RUFF_RULES,
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return True, out.strip()


def run_static_preflight(
    generated_root: Path,
    py_files: List[Path],
) -> Dict[str, Any]:
    root = generated_root.resolve()
    package_roots_list = _discover_package_roots(root)
    pkg_set = set(package_roots_list)
    issues: List[Dict[str, Any]] = []
    compile_failures: List[Dict[str, Any]] = []

    normalized = sorted({p.resolve() for p in py_files if p.suffix == ".py" and p.is_file()}, key=str)

    for path in normalized:
        try:
            src = path.read_text(encoding="utf-8")
            compile(src, str(path), "exec")
        except SyntaxError as exc:
            compile_failures.append(
                {
                    "file": str(path),
                    "line": exc.lineno or 0,
                    "message": f"{exc.msg}",
                }
            )
        except OSError as exc:
            compile_failures.append({"file": str(path), "line": 0, "message": str(exc)})

    for path in normalized:
        _check_imports(path, root, pkg_set, issues)

    ruff_ok, ruff_out = _run_ruff(root)
    if ruff_ok and ruff_out:
        issues.append(
            {
                "kind": "ruff",
                "file": str(root),
                "line": 0,
                "message": ruff_out[:20000],
            }
        )

    issue_kind_counts: Dict[str, int] = {}
    for item in issues:
        kind = str(item.get("kind") or "other")
        issue_kind_counts[kind] = issue_kind_counts.get(kind, 0) + 1

    return {
        "generated_root": str(root),
        "package_roots": package_roots_list,
        "files_checked": len(normalized),
        "issue_count": len(issues),
        "issue_kind_counts": issue_kind_counts,
        "issues": issues,
        "compile_failure_count": len(compile_failures),
        "compile_failures": compile_failures,
        "ruff_available": ruff_ok,
    }


__all__ = ["run_static_preflight", "RUFF_RULES"]
