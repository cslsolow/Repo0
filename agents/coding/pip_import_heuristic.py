"""Shared import → PyPI heuristics for SetupPyAgent and TDD pytest pip installs."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Single source of truth: top-level import name -> PyPI distribution name (when they differ).
IMPORT_ROOT_TO_PYPI: Dict[str, str] = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "dateutil": "python-dateutil",
    "dns": "dnspython",
    "skimage": "scikit-image",
    "bs4": "beautifulsoup4",
    "Crypto": "pycryptodome",
    "jwt": "PyJWT",
    "lxml": "lxml",
    "odf": "odfpy",
    "googleapiclient": "google-api-python-client",
    "apiclient": "google-api-python-client",
    "faker": "Faker",
    "mx": "egenix-mx-base",
    "sqlalchemy": "SQLAlchemy",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "flask": "Flask",
    "django": "Django",
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "requests": "requests",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "redis": "redis",
    "psycopg2": "psycopg2-binary",
    "MySQLdb": "mysqlclient",
    "cvxpy": "cvxpy",
    "patsy": "patsy",
    "IPython": "ipython",
    "jinja2": "Jinja2",
    "markupsafe": "MarkupSafe",
    "werkzeug": "Werkzeug",
    "dotenv": "python-dotenv",
    "openai": "openai",
    "protobuf": "protobuf",
    "grpc": "grpcio",
}

_MOD_NOT_FOUND_RE = re.compile(
    r"(?:ModuleNotFoundError|ImportError):\s*No module named\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE | re.MULTILINE,
)


def stdlib_names() -> Set[str]:
    names = getattr(sys, "stdlib_module_names", None)
    if isinstance(names, (set, frozenset)):
        return set(names)
    return {
        "abc",
        "argparse",
        "array",
        "ast",
        "asyncio",
        "atexit",
        "base64",
        "bisect",
        "builtins",
        "bz2",
        "calendar",
        "collections",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "gc",
        "glob",
        "hashlib",
        "heapq",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "multiprocessing",
        "operator",
        "os",
        "pathlib",
        "pickle",
        "pkgutil",
        "platform",
        "queue",
        "random",
        "re",
        "secrets",
        "shlex",
        "shutil",
        "signal",
        "socket",
        "sqlite3",
        "string",
        "struct",
        "subprocess",
        "sys",
        "tempfile",
        "textwrap",
        "threading",
        "time",
        "tokenize",
        "traceback",
        "types",
        "typing",
        "unicodedata",
        "unittest",
        "urllib",
        "uuid",
        "warnings",
        "weakref",
        "xml",
        "zipfile",
        "zlib",
    }


def distribution_for_import_root(import_root: str) -> str:
    """Map a top-level import name to a PyPI distribution name (heuristic)."""
    raw = str(import_root).strip()
    if raw in IMPORT_ROOT_TO_PYPI:
        return IMPORT_ROOT_TO_PYPI[raw]
    lo = raw.lower()
    for k, v in IMPORT_ROOT_TO_PYPI.items():
        if k.lower() == lo:
            return v
    return raw


def ast_top_level_import_roots(tree: ast.AST) -> Set[str]:
    """Top-level names from ``import`` / ``from pkg import`` (absolute only)."""
    tops: Set[str] = set()
    froms: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = (alias.name or "").split(".")[0]
                if name:
                    tops.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if not node.module:
                continue
            mod = node.module.split(".")[0]
            if mod:
                froms.add(mod)
    tops.update(froms)
    return tops


def collect_import_roots_from_sources(*sources: str) -> Set[str]:
    """Parse arbitrary Python source strings and union top-level import roots."""
    roots: Set[str] = set()
    for src in sources:
        if not src or not str(src).strip():
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        roots.update(ast_top_level_import_roots(tree))
    return roots


def filter_third_party_import_roots(
    import_roots: Sequence[str],
    *,
    local_top_level: Set[str],
    stdlib: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Split into third-party import roots vs skipped (stdlib / local), matching SetupPyAgent rules."""
    stdlib_set = stdlib if stdlib is not None else stdlib_names()
    local_lower = {x.lower() for x in local_top_level}
    skipped: List[str] = []
    external: List[str] = []

    for raw in sorted(set(import_roots)):
        if not raw or raw == "__future__":
            skipped.append(raw)
            continue
        if raw.startswith("_") and raw != "_pytest":
            if raw in stdlib_set:
                skipped.append(raw)
                continue
        if raw in stdlib_set:
            skipped.append(raw)
            continue
        if raw in local_top_level or raw.lower() in local_lower:
            skipped.append(raw)
            continue
        external.append(raw)

    return external, skipped


def third_party_roots_to_pip_install_specs(third_party_roots: Sequence[str]) -> List[str]:
    """Dedupe mapped PyPI distribution names for ``pip install``."""
    seen: Set[str] = set()
    out: List[str] = []
    for r in sorted(set(third_party_roots)):
        dist = distribution_for_import_root(r)
        if dist not in seen:
            seen.add(dist)
            out.append(dist)
    return out


def third_party_roots_to_heuristic_rows(third_party_roots: Sequence[str]) -> List[Dict[str, str]]:
    """Rows for SetupPyAgent / LLM payload (import_root → suggested_distribution)."""
    rows: List[Dict[str, str]] = []
    for root_name in sorted(set(third_party_roots)):
        pypi = distribution_for_import_root(root_name)
        rows.append(
            {
                "import_root": root_name,
                "suggested_distribution": pypi,
                "version_spec": "",
            }
        )
    return rows


def import_roots_to_pip_specs(
    roots: Iterable[str],
    *,
    stdlib: Set[str],
    local: Set[str],
) -> List[str]:
    """TDD helper: filter then map to pip install arguments."""
    ext, _ = filter_third_party_import_roots(list(roots), local_top_level=local, stdlib=stdlib)
    return third_party_roots_to_pip_install_specs(ext)


def _package_parts_for_file(root: Path, file_path: Path) -> List[str]:
    rel = file_path.relative_to(root)
    parts = list(rel.parts)
    if not parts:
        return []
    if parts[-1] == "__init__.py":
        return parts[:-1]
    if parts[-1].endswith(".py"):
        return parts[:-1]
    return parts


def _resolve_local_module_file(root: Path, module_parts: Sequence[str]) -> Optional[Path]:
    if not module_parts:
        return None
    base = root.joinpath(*module_parts)
    file_candidate = base.with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate
    init_candidate = base / "__init__.py"
    if init_candidate.is_file():
        return init_candidate
    return None


def collect_transitive_project_import_roots(
    project_root: Path,
    rel_paths: Iterable[str],
) -> Set[str]:
    """Collect third-party import roots reachable through local project imports.

    Starts from the given project-relative entry files, recursively follows local imports,
    and returns the union of external top-level import roots encountered.
    """
    root = Path(project_root)
    local_top = sandbox_top_level_names(root, rel_paths)
    stdlib = stdlib_names()
    visited: Set[Path] = set()
    external_roots: Set[str] = set()

    def visit_file(file_path: Path) -> None:
        file_path = file_path.resolve()
        if file_path in visited or not file_path.is_file():
            return
        visited.add(file_path)
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            return

        package_parts = _package_parts_for_file(root.resolve(), file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    full = [p for p in str(alias.name or "").split(".") if p]
                    if not full:
                        continue
                    top = full[0]
                    if top in stdlib:
                        continue
                    if top in local_top:
                        target = _resolve_local_module_file(root, full)
                        if target is not None:
                            visit_file(target)
                        continue
                    external_roots.add(top)
            elif isinstance(node, ast.ImportFrom):
                module_name = str(node.module or "").strip()
                if node.level and node.level > 0:
                    keep = len(package_parts) - (node.level - 1)
                    if keep < 0:
                        keep = 0
                    target_parts = list(package_parts[:keep])
                    if module_name:
                        target_parts.extend([p for p in module_name.split(".") if p])
                    target = _resolve_local_module_file(root, target_parts)
                    if target is not None:
                        visit_file(target)
                        for alias in node.names:
                            child_name = str(alias.name or "").strip()
                            if not child_name or child_name == "*":
                                continue
                            child_target = _resolve_local_module_file(root, [*target_parts, child_name])
                            if child_target is not None:
                                visit_file(child_target)
                    continue

                if not module_name:
                    continue
                full = [p for p in module_name.split(".") if p]
                if not full:
                    continue
                top = full[0]
                if top in stdlib:
                    continue
                if top in local_top:
                    target = _resolve_local_module_file(root, full)
                    if target is not None:
                        visit_file(target)
                        for alias in node.names:
                            child_name = str(alias.name or "").strip()
                            if not child_name or child_name == "*":
                                continue
                            child_target = _resolve_local_module_file(root, [*full, child_name])
                            if child_target is not None:
                                visit_file(child_target)
                    continue
                external_roots.add(top)

    for rel in rel_paths:
        rel_norm = str(rel or "").strip().replace("\\", "/")
        if not rel_norm:
            continue
        candidate = (root / rel_norm).resolve()
        if candidate.is_file():
            visit_file(candidate)

    return external_roots


def sandbox_top_level_names(root: Path, rel_paths: Iterable[str]) -> Set[str]:
    """Top-level package/dir names under ``root`` (layout + ``rel_paths`` first segment)."""
    names: Set[str] = set()
    for rel in rel_paths:
        parts = Path(str(rel).replace("\\", "/")).parts
        if parts:
            top = parts[0]
            if top.endswith(".py"):
                top = Path(top).stem
            names.add(top)
    if root.is_dir():
        try:
            for child in root.iterdir():
                if child.is_dir():
                    names.add(child.name)
                elif child.suffix == ".py":
                    names.add(child.stem)
        except OSError:
            pass
    return names


def missing_import_roots_from_pytest_log(log: str) -> List[str]:
    """Ordered unique top-level module names from pytest ``ModuleNotFoundError`` lines."""
    found: List[str] = []
    seen: Set[str] = set()
    for m in _MOD_NOT_FOUND_RE.findall(log or ""):
        top = str(m).split(".")[0].strip()
        if not top:
            continue
        low = top.lower()
        if low in seen:
            continue
        seen.add(low)
        found.append(top)
    return found
