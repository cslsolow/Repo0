#!/usr/bin/env python3
"""
Evaluation tasks collection + agent pipeline runner.

Implements:
  - D.2: Test function harvesting, hierarchical categorization, sampling/filtering
  - D.3: Localization -> majority-vote validation -> test adaptation/execution
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Tuple

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "__pycache__",
}

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

from pipeline3_test_adapt import adapt_test_code, run_pytest


@dataclass(frozen=True)
class TestItem:
    file_path: Path
    module_name: str
    class_name: Optional[str]
    function_name: str
    docstring: str


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    file_path: Path
    lineno: int
    docstring: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation task collector + pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect and sample evaluation tasks")
    collect.add_argument("--repo", type=Path, required=True, help="Repository root")
    collect.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    collect.add_argument(
        "--taxonomy-output",
        type=Path,
        default=None,
        help="Output path for hierarchical taxonomy JSON",
    )
    collect.add_argument(
        "--sampled-output",
        type=Path,
        default=None,
        help="Output path for sampled/filtered tasks (JSON/JSONL)",
    )
    collect.add_argument("--max-tasks", type=int, default=100, help="Max tasks to sample")
    collect.add_argument("--seed", type=int, default=13, help="Random seed")
    collect.add_argument(
        "--exclude-keywords",
        type=str,
        default="format,repr,str,version,warning,logging,cli,help,docstring,typing",
        help="Comma-separated keywords to exclude",
    )
    collect.add_argument(
        "--root-tests-imports",
        action="store_true",
        help="Use imports to categorize tests when tests/ is at repo root",
    )

    evaluate = subparsers.add_parser("evaluate", help="Run the agent pipeline on tasks")
    evaluate.add_argument("--repo", type=Path, required=True, help="Generated repository root")
    evaluate.add_argument("--tasks", type=Path, required=True, help="Tasks JSONL/JSON path")
    evaluate.add_argument("--tests-root", type=Path, required=True, help="Source tests repo root")
    evaluate.add_argument("--output", type=Path, required=True, help="Output JSON path")
    evaluate.add_argument("--max-candidates", type=int, default=8, help="Max candidates to validate")
    evaluate.add_argument("--llm-votes", type=int, default=5, help="LLM votes per candidate")
    evaluate.add_argument("--max-attempts", type=int, default=3, help="Localization retries")
    evaluate.add_argument("--no-llm", action="store_true", help="Disable LLM validation")
    evaluate.add_argument("--base-url", type=str, default="", help="LLM base URL")
    evaluate.add_argument("--api-key", type=str, default="", help="LLM API key")
    evaluate.add_argument("--model", type=str, default="", help="LLM model")
    evaluate.add_argument("--output-dir", type=Path, default=None, help="Working directory")
    return parser.parse_args()


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in DEFAULT_EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def discover_test_files(root: Path) -> List[Path]:
    test_files = []
    for path in iter_python_files(root):
        if path.name.startswith("test_") or path.name.endswith("_test.py") or "tests" in path.parts:
            test_files.append(path)
    return test_files


def parse_test_items(file_path: Path) -> List[TestItem]:
    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    items: List[TestItem] = []
    module_name = file_path.stem

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            doc = ast.get_docstring(node) or ""
            items.append(
                TestItem(file_path, module_name, None, node.name, doc)
            )
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            class_doc = ast.get_docstring(node) or ""
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name.startswith("test"):
                    doc = ast.get_docstring(sub) or class_doc
                    items.append(
                        TestItem(file_path, module_name, node.name, sub.name, doc)
                    )
    return items


def detect_category_from_imports(file_path: Path) -> Optional[str]:
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None

    counts: Counter = Counter()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[-1]
                if module and module not in IGNORE_IMPORT_NAMES:
                    counts[module] += 1
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            module = node.module.split(".")[-1]
            if module and module not in IGNORE_IMPORT_NAMES:
                counts[module] += 1

    if not counts:
        return None
    return counts.most_common(1)[0][0]


def category_from_path(
    test_root: Path,
    file_path: Path,
    use_imports_for_root_tests: bool,
) -> str:
    rel = file_path.relative_to(test_root)
    if "tests" in rel.parts:
        idx = rel.parts.index("tests")
        if idx == 0 and use_imports_for_root_tests:
            detected = detect_category_from_imports(file_path)
            if detected:
                return detected
        if idx > 0:
            return rel.parts[idx - 1]
        if idx + 1 < len(rel.parts):
            return rel.parts[idx + 1]
    return rel.parts[0] if rel.parts else "misc"


def build_task_taxonomy(
    test_root: Path,
    items: List[TestItem],
    use_imports_for_root_tests: bool,
) -> Dict[str, Any]:
    taxonomy: Dict[str, Any] = {}
    for item in items:
        category = category_from_path(test_root, item.file_path, use_imports_for_root_tests)
        category_node = taxonomy.setdefault(category, {})
        module_node = category_node.setdefault(item.module_name, {"classes": {}, "functions": []})
        if item.class_name:
            class_node = module_node["classes"].setdefault(item.class_name, [])
            class_node.append(item.function_name)
        else:
            module_node["functions"].append(item.function_name)
    return taxonomy


def snake_to_words(name: str) -> str:
    return " ".join(part for part in re.split(r"[_\W]+", name) if part)


def build_task_query(cap: str, functions: List[str], docstrings: List[str]) -> str:
    doc = next((d for d in docstrings if d), "")
    if doc:
        first_line = doc.strip().splitlines()[0].strip()
        return f"You are testing {cap} behavior: {first_line}"
    keywords = ", ".join(snake_to_words(fn) for fn in functions[:5])
    return f"You are testing {cap} behavior covering: {keywords}."


def build_tasks(
    repo_name: str,
    test_root: Path,
    items: List[TestItem],
    exclude_keywords: List[str],
    use_imports_for_root_tests: bool,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    taxonomy = build_task_taxonomy(test_root, items, use_imports_for_root_tests)

    for category, modules in taxonomy.items():
        for module_name, module_node in modules.items():
            module_file = next(
                (item.file_path for item in items if item.module_name == module_name),
                None,
            )
            if not module_file:
                continue
            rel_file = module_file.relative_to(test_root)

            if module_node["functions"]:
                functions = sorted(set(module_node["functions"]))
                cap = module_name
                if should_exclude_task(cap, functions, exclude_keywords):
                    continue
                docstrings = [
                    item.docstring
                    for item in items
                    if item.module_name == module_name and item.class_name is None
                ]
                tasks.append(
                    {
                        "category": category,
                        "file": str(rel_file),
                        "module": module_name,
                        "cap": cap,
                        "functions": functions,
                        "task_query": build_task_query(cap, functions, docstrings),
                        "id": "",
                    }
                )

            for class_name, methods in module_node["classes"].items():
                functions = sorted(set(methods))
                cap = class_name
                if should_exclude_task(cap, functions, exclude_keywords):
                    continue
                docstrings = [
                    item.docstring
                    for item in items
                    if item.module_name == module_name and item.class_name == class_name
                ]
                tasks.append(
                    {
                        "category": category,
                        "file": str(rel_file),
                        "module": f"class {class_name}",
                        "cap": cap,
                        "functions": functions,
                        "task_query": build_task_query(cap, functions, docstrings),
                        "id": "",
                    }
                )

    for idx, task in enumerate(tasks, 1):
        task["id"] = f"{repo_name}-{idx:04d}"
    return tasks


def normalize_test_name(name: str) -> str:
    return re.sub(r"^test_+", "", name).strip("_")


def tokenize_test_name(name: str) -> List[str]:
    return [t for t in re.split(r"[_\W]+", name) if t]


def build_ngram_counts(names: List[str]) -> Counter:
    counts: Counter = Counter()
    for name in names:
        tokens = tokenize_test_name(normalize_test_name(name))
        seen = set()
        for n in (3, 2):
            for i in range(len(tokens) - n + 1):
                gram = "_".join(tokens[i : i + n])
                seen.add(gram)
        for gram in seen:
            counts[gram] += 1
    return counts


def choose_group_key(name: str, counts: Counter, default_group: str) -> str:
    tokens = tokenize_test_name(normalize_test_name(name))
    if not tokens:
        return default_group or "misc"
    candidates = []
    for n in (3, 2):
        for i in range(len(tokens) - n + 1):
            gram = "_".join(tokens[i : i + n])
            count = counts.get(gram, 0)
            if count >= 2:
                candidates.append((count, n, i, gram))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
        return candidates[0][3]
    if default_group:
        return default_group
    return tokens[0]


def group_test_functions(functions: List[str], default_group: str) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    if not functions:
        return grouped
    counts = build_ngram_counts(functions)
    for func in sorted(set(functions)):
        key = choose_group_key(func, counts, default_group)
        grouped.setdefault(key, []).append(func)
    return grouped


def build_taxonomy(
    test_root: Path,
    items: List[TestItem],
    use_imports_for_root_tests: bool,
) -> Dict[str, Any]:
    taxonomy: Dict[str, Any] = {}
    module_functions: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    class_functions: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    for item in items:
        category = category_from_path(test_root, item.file_path, use_imports_for_root_tests)
        if item.class_name:
            class_functions[(category, item.module_name, item.class_name)].append(item.function_name)
        else:
            module_functions[(category, item.module_name)].append(item.function_name)

    for (category, module_name), funcs in module_functions.items():
        category_node = taxonomy.setdefault(category, {})
        module_key = module_name
        default_group = normalize_test_name(module_name) if module_name.startswith("test") else module_name
        module_node = category_node.setdefault(module_key, {"classes": {}, "functions": {}})
        module_node["functions"] = group_test_functions(funcs, default_group)

    for (category, module_name, class_name), funcs in class_functions.items():
        category_node = taxonomy.setdefault(category, {})
        module_node = category_node.setdefault(module_name, {"classes": {}, "functions": {}})
        class_default = normalize_test_name(class_name) if class_name.lower().startswith("test") else class_name
        module_node["classes"][class_name] = {
            "functions": group_test_functions(funcs, class_default)
        }

    return taxonomy


def should_exclude_task(cap: str, functions: List[str], keywords: List[str]) -> bool:
    haystack = " ".join([cap] + functions).lower()
    return any(kw in haystack for kw in keywords)


def balanced_sample(tasks: List[Dict[str, Any]], max_tasks: int, seed: int) -> List[Dict[str, Any]]:
    if len(tasks) <= max_tasks:
        return tasks
    random.seed(seed)
    tasks_by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        tasks_by_cat[task["category"]].append(task)

    for task_list in tasks_by_cat.values():
        random.shuffle(task_list)

    categories = list(tasks_by_cat.keys())
    random.shuffle(categories)

    sampled: List[Dict[str, Any]] = []
    while len(sampled) < max_tasks:
        progressed = False
        for cat in categories:
            if tasks_by_cat[cat]:
                sampled.append(tasks_by_cat[cat].pop(0))
                progressed = True
                if len(sampled) >= max_tasks:
                    break
        if not progressed:
            break
    return sampled


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        tasks = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tasks.append(json.loads(line))
        return tasks
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("tasks", [])
    return data


def write_tasks(path: Path, tasks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_taxonomy(path: Path, taxonomy: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")


def derive_sampled_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.sampled{output_path.suffix}")


def derive_taxonomy_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.taxonomy.json")


def build_candidate_index(repo_root: Path) -> List[Candidate]:
    candidates: List[Candidate] = []
    for path in iter_python_files(repo_root):
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                candidates.append(
                    Candidate(
                        name=node.name,
                        kind="function",
                        file_path=path,
                        lineno=node.lineno,
                        docstring=ast.get_docstring(node) or "",
                    )
                )
            if isinstance(node, ast.ClassDef):
                candidates.append(
                    Candidate(
                        name=node.name,
                        kind="class",
                        file_path=path,
                        lineno=node.lineno,
                        docstring=ast.get_docstring(node) or "",
                    )
                )
    return candidates


def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9_]+", text.lower()) if t]


def score_candidate(task: Dict[str, Any], candidate: Candidate) -> float:
    tokens = set(tokenize(task.get("task_query", "")) + tokenize(task.get("cap", "")))
    tokens.update(tokenize(" ".join(task.get("functions", []))))
    if not tokens:
        return 0.0
    cand_tokens = set(tokenize(candidate.name) + tokenize(candidate.docstring))
    overlap = len(tokens & cand_tokens) / max(len(tokens), 1)
    name_sim = 0.0
    if task.get("cap"):
        name_sim = sequence_ratio(task["cap"].lower(), candidate.name.lower())
    return overlap + name_sim


def sequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return len(os.path.commonprefix([a, b])) / max(len(a), len(b))


def candidate_key(candidate: Candidate) -> Tuple[str, str, str, int]:
    return (candidate.kind, candidate.name, str(candidate.file_path), candidate.lineno)


def localize_candidates(
    task: Dict[str, Any],
    candidates: List[Candidate],
    max_candidates: int,
    exclude: Optional[set] = None,
) -> List[Candidate]:
    exclude = exclude or set()
    scored = []
    for cand in candidates:
        if candidate_key(cand) in exclude:
            continue
        scored.append((score_candidate(task, cand), cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cand for _, cand in scored[:max_candidates]]


def load_llm_client(base_url: str, api_key: str, model: str, output_dir: Path, reasoning_effort: str = ""):
    if not (base_url and api_key and model):
        return None
    try:
        from agents.llm_client import LLMClient
    except Exception:
        return None
    return LLMClient({"base_url": base_url, "api_key": api_key, "model": model, "reasoning_effort": reasoning_effort}, str(output_dir), "validator")


def validate_candidate_llm(
    llm_client,
    task: Dict[str, Any],
    candidate: Candidate,
    votes: int,
) -> Tuple[bool, List[str]]:
    prompt = (
        "Task description:\n"
        f"{task.get('task_query','')}\n\n"
        "Candidate:\n"
        f"- name: {candidate.name}\n"
        f"- type: {candidate.kind}\n"
        f"- file: {candidate.file_path}\n"
        f"- docstring: {candidate.docstring[:400]}\n\n"
        "Question: Does the candidate implement the described algorithm? Answer yes or no."
    )
    results = []
    for _ in range(votes):
        response = llm_client.call(
            [
                {"role": "system", "content": "You are a strict code reviewer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=64,
        )
        verdict = parse_yes_no(response)
        results.append(verdict)
    yes_votes = sum(1 for r in results if r == "yes")
    return yes_votes > votes // 2, results


def parse_yes_no(text: str) -> str:
    text = text.strip().lower()
    if "yes" in text and "no" not in text:
        return "yes"
    if "no" in text and "yes" not in text:
        return "no"
    return "no"




def collect_command(args: argparse.Namespace) -> None:
    repo_root = args.repo.resolve()
    test_files = discover_test_files(repo_root)
    items = []
    for file_path in test_files:
        items.extend(parse_test_items(file_path))

    taxonomy = build_taxonomy(repo_root, items, args.root_tests_imports)
    exclude_keywords = [kw.strip().lower() for kw in args.exclude_keywords.split(",") if kw.strip()]
    tasks = build_tasks(repo_root.name, repo_root, items, exclude_keywords, args.root_tests_imports)
    tasks = balanced_sample(tasks, args.max_tasks, args.seed)
    write_tasks(args.output, tasks)
    taxonomy_output = args.taxonomy_output or derive_taxonomy_output_path(args.output)
    write_taxonomy(taxonomy_output, taxonomy)
    sampled_output = args.sampled_output or derive_sampled_output_path(args.output)
    if sampled_output != args.output:
        write_tasks(sampled_output, tasks)


def evaluate_command(args: argparse.Namespace) -> None:
    repo_root = args.repo.resolve()
    tests_root = args.tests_root.resolve()
    output_dir = args.output_dir or (repo_root / "eval_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.tasks)
    candidates = build_candidate_index(repo_root)
    llm_client = None
    if not args.no_llm:
        llm_client = load_llm_client(args.base_url, args.api_key, args.model, output_dir)

    results = []
    for task in tasks:
        task["_repo_root"] = repo_root
        test_file = tests_root / task["file"]
        localized = []
        validated = None
        attempts = 0
        validated_votes = []
        tried_candidates = set()

        while attempts < args.max_attempts and validated is None:
            attempts += 1
            localized = localize_candidates(
                task,
                candidates,
                args.max_candidates,
                exclude=tried_candidates,
            )
            if not localized:
                break
            for cand in localized:
                if llm_client:
                    ok, votes = validate_candidate_llm(llm_client, task, cand, args.llm_votes)
                    validated_votes = votes
                else:
                    ok = True
                if ok:
                    validated = cand
                    break
                tried_candidates.add(candidate_key(cand))

        if validated:
            adapted_path = adapt_test_code(test_file, output_dir, validated, task)
            test_result = run_pytest(repo_root, adapted_path, task.get("functions", []))
        else:
            adapted_path = None
            test_result = {"returncode": -1, "stdout": "", "stderr": "No validated candidate"}

        results.append(
            {
                "task_id": task.get("id"),
                "validated_candidate": candidate_to_dict(validated) if validated else None,
                "validation_votes": validated_votes,
                "localized_candidates": [candidate_to_dict(c) for c in localized],
                "adapted_test": str(adapted_path) if adapted_path else None,
                "test_result": test_result,
                "attempts": attempts,
            }
        )

    args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")


def candidate_to_dict(candidate: Optional[Candidate]) -> Optional[Dict[str, Any]]:
    if not candidate:
        return None
    return {
        "name": candidate.name,
        "kind": candidate.kind,
        "file": str(candidate.file_path),
        "lineno": candidate.lineno,
    }


def main() -> None:
    args = parse_args()
    if args.command == "collect":
        collect_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)


if __name__ == "__main__":
    main()
